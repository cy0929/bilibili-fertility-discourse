# -*- coding: utf-8 -*-
"""
LLM-based comment sentiment analysis against a locally deployed,
OpenAI-compatible Chat Completions endpoint (vLLM, SGLang, LM Studio, Ollama, ...).

Example:
  export LLM_API_BASE_URL=http://localhost:8000/v1     # local server; no key needed
  python sentiment_analysis_llm.py --config config/sentiment_config.toml --dry-run --limit 20
  python sentiment_analysis_llm.py --config config/sentiment_config.toml

Privacy:
  - Only the comment text plus a minimal video context (title) is sent to the model.
    Commenter names / user IDs are never read or transmitted.
  - The script writes a new workbook by default and saves progress regularly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import openpyxl


# Presentation mappings are local to this original script. No source cell values
# are translated, and the source workbook is never saved by these helpers.
EXPORT_HEADER_NAMES = {
    "说明": "description",
    "项目": "item",
    "评论内容": "comment_content",
    "评论文本": "comment_text",
    "评论正文": "comment_body",
    "用户评论": "user_comment",
    "评论": "comment",
    "标题": "video_title",
    "视频ID": "video_identifier",
    "主题名称": "video_theme_name",
    "学术分类": "analytic_category",
    "四分区": "video_partition_name",
    "已失效视频": "invalid_video_flag",
    "评论ID": "comment_id",
    "根评论ID": "root_comment_id",
    "点赞数": "like_count",
    "情绪标签": "sentiment_label",
    "情绪分数": "sentiment_score"
}
EXPORT_SHEET_NAMES = {
    "说明": "Notes",
    "情绪结果": "Sentiment_results",
    "情绪分析": "Sentiment_analysis",
    "视频信息": "Video_information",
    "评论明细": "Comment_details"
}


def _contains_chinese(value):
    return isinstance(value, str) and any("\u3400" <= c <= "\u9fff" for c in value)


def _export_plan(workbook):
    """Choose English display names without changing column or sheet order.

    Unknown source labels use positional English identifiers, avoiding an
    invented translation. Original labels are recovered from the source workbook
    before any resumed model input is constructed.
    """
    reserved_sheets = {ws.title.lower() for ws in workbook if not _contains_chinese(ws.title)}
    plan = []
    for index, ws in enumerate(workbook, 1):
        title = ws.title
        if _contains_chinese(title):
            base = EXPORT_SHEET_NAMES.get(title, f"Sheet{index}")
            title = base
            suffix = 1
            while title.lower() in reserved_sheets:
                title = f"{base[:25]}_{suffix}"
                suffix += 1
            reserved_sheets.add(title.lower())
        original = [cell.value for cell in ws[1]]
        reserved_headers = {v for v in original if isinstance(v, str) and not _contains_chinese(v)}
        translated = []
        for column, value in enumerate(original, 1):
            name = value
            if _contains_chinese(value):
                name = EXPORT_HEADER_NAMES.get(value, f"source_field_{column}")
                while name in reserved_headers:
                    name += f"_source_{column}"
                reserved_headers.add(name)
            translated.append(name)
        plan.append((title, translated))
    return plan


def save_english_workbook(workbook, path):
    """Save English display names, then restore the in-memory source schema.

    Only first-row header cells and sheet titles are changed during saving.
    All result values, row/column positions and the model-input schema are kept.
    """
    originals = [(ws.title, [cell.value for cell in ws[1]]) for ws in workbook]
    plan = _export_plan(workbook)
    try:
        for ws, (title, headers) in zip(workbook, plan):
            ws.title = title
            for cell, value in zip(ws[1], headers):
                cell.value = value
        workbook.save(path)
    finally:
        for ws, (title, headers) in zip(workbook, originals):
            ws.title = title
            for cell, value in zip(ws[1], headers):
                cell.value = value


def restore_source_schema(workbook, input_path):
    """Restore exported header/sheet names from the unchanged original input.

    This happens before selecting the configured worksheet or constructing model
    context. Result columns and every data cell remain untouched.
    """
    source = openpyxl.load_workbook(input_path, read_only=True, data_only=False)
    try:
        plan = _export_plan(source)
        for current, original, (title, headers) in zip(workbook, source, plan):
            if current.title not in {original.title, title}:
                continue
            current.title = original.title
            for cell, original_cell, exported in zip(current[1], original[1], headers):
                if cell.value == exported:
                    cell.value = original_cell.value
    finally:
        source.close()



# Default: a locally deployed OpenAI-compatible server. Override with LLM_API_BASE_URL or [api].base_url.
API_BASE_URL = "http://localhost:8000/v1"
# Local servers usually ignore the bearer token; a placeholder keeps the header well-formed.
LOCAL_PLACEHOLDER_API_KEY = "EMPTY"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"


def _resolve_default_config() -> Path:
    """Config file lookup order:
    1. Environment variable BFD_SENTIMENT_CONFIG
    2. <repo_root>/config/sentiment_config.toml        (user's own config, already gitignored)
    3. <repo_root>/config/sentiment_config.example.toml (example template shipped with the repo)
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    candidates = [
        Path(os.environ["BFD_SENTIMENT_CONFIG"]) if os.environ.get("BFD_SENTIMENT_CONFIG") else None,
        repo_root / "config" / "sentiment_config.toml",
        repo_root / "config" / "sentiment_config.example.toml",
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path
    return candidates[1]  # When none exist, return the most obvious expected path; the loader reports the error later


DEFAULT_CONFIG_PATH = _resolve_default_config()
LOG = logging.getLogger("llm_sentiment")

COMMENT_COLUMN_CANDIDATES = [
    "评论内容",
    "评论文本",
    "评论正文",
    "用户评论",
    "评论",
    "content",
    "comment_text",
    "comment",
    "text",
    "message",
]

# Commenter name / user-id columns are deliberately NOT read: usernames are never sent to the model.

CONTEXT_COLUMN_CANDIDATES = [
    "Title",
    "标题",
    "Video_ID",
    "视频ID",
    "URL",
    "主题名称",
    "Search_item",
]

RESULT_COLUMNS = [
    "sentiment_label",
    "sentiment_score",
    "sentiment_model",
    "sentiment_status",
    "sentiment_processed_at",
]

SYSTEM_PROMPT = """你是一个严谨的中文社交媒体评论情感分析员。

任务：分析用户评论表达的情感，而不是分析视频标题、作者立场或事实真伪。

判定规则：
1. 只根据评论文本判断；标题、视频主题等上下文只能用于消歧，不能替代评论本身。
2. 情感标签只能从 positive、negative、neutral、mixed、uncertain 中选择。
3. sentiment_score 使用 -1 到 1：-1 极负面，0 中性，1 极正面。
4. comment 过短、只有表情/乱码/无法判断时，label 用 uncertain，score 接近 0。
5. 讽刺、反问、阴阳怪气要按真实情绪判断；如果无法可靠判断，用 mixed 或 uncertain。
6. 必须严格输出 JSON，不要输出 Markdown，不要添加解释文字。"""

USER_PROMPT_TEMPLATE = """请对下面这批评论逐条做情感分析。

输出 JSON 格式：
{{
  "items": [
    {{
      "row_id": 2,
      "sentiment_label": "positive|negative|neutral|mixed|uncertain",
      "sentiment_score": 0.0
    }}
  ]
}}

评论数据：
{comments_json}
"""

JSON_SCHEMA = {
    "name": "comment_sentiment_batch",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "row_id",
                        "sentiment_label",
                        "sentiment_score",
                    ],
                    "properties": {
                        "row_id": {"type": "integer"},
                        "sentiment_label": {
                            "type": "string",
                            "enum": [
                                "positive",
                                "negative",
                                "neutral",
                                "mixed",
                                "uncertain",
                            ],
                        },
                        "sentiment_score": {
                            "type": "number",
                            "minimum": -1,
                            "maximum": 1,
                        },
                    },
                },
            }
        },
    },
}


@dataclass
class CommentItem:
    row_id: int
    comment: str
    user: str | None
    context: dict[str, str]


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ResponseParseError(RuntimeError):
    pass


def prefix_timestamp(path: Path, timestamp: str) -> Path:
    return path.with_name(f"{timestamp}_{path.name}")


def find_latest_timestamped_file(base_path: Path) -> Path | None:
    pattern = re.compile(rf"^\d{{14}}_{re.escape(base_path.name)}$")
    if not base_path.parent.exists():
        return None

    candidates = [
        path
        for path in base_path.parent.iterdir()
        if path.is_file() and pattern.match(path.name)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def setup_logging(level: str, log_file: str | None, timestamp: str) -> Path | None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    actual_log_path: Path | None = None
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        actual_log_path = prefix_timestamp(log_path, timestamp)
        actual_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(actual_log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        LOG.info('Log file: %s', actual_log_path)

    return actual_log_path


def config_path_for_display(config_path: str | None) -> Path:
    return Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH


def safe_api_key_label(value: str | None) -> str:
    if not value:
        return 'not supplied'
    text = value.strip()
    if len(text) <= 10:
        return "***"
    return f"{text[:6]}...{text[-4:]}"


def rough_token_count(text: str) -> int:
    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii_count = len(text) - ascii_count
    return int(non_ascii_count + ascii_count / 4)


def log_duration(label: str, start_time: float) -> None:
    LOG.info('%s completed in %.2f seconds', label, time.perf_counter() - start_time)


class MinuteRateLimiter:
    def __init__(
        self,
        max_rpm: int,
        max_tpm: int,
        safety_factor: float,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.max_rpm = max(1, int(max_rpm))
        self.max_tpm = max(1, int(max_tpm))
        self.safety_factor = min(max(float(safety_factor), 0.1), 1.0)
        self.effective_rpm = max(1, int(self.max_rpm * self.safety_factor))
        self.effective_tpm = max(1, int(self.max_tpm * self.safety_factor))
        self._request_times: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - 60
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def acquire(self, estimated_tokens: int, label: str) -> None:
        if not self.enabled:
            LOG.debug('Rate limiter disabled: %s', label)
            return

        token_cost = max(1, min(int(estimated_tokens), self.effective_tpm))
        if estimated_tokens > self.effective_tpm:
            LOG.warning(
                'Estimated request tokens=%s exceed effective TPM=%s; count effective TPM for rate limiting: %s',
                estimated_tokens,
                self.effective_tpm,
                label,
            )

        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                rpm_used = len(self._request_times)
                tpm_used = sum(tokens for _, tokens in self._token_events)

                if rpm_used + 1 <= self.effective_rpm and tpm_used + token_cost <= self.effective_tpm:
                    self._request_times.append(now)
                    self._token_events.append((now, token_cost))
                    LOG.debug(
                        'Rate limit passed: %s rpm=%s/%s tpm_est=%s/%s request_tokens=%s',
                        label,
                        rpm_used + 1,
                        self.effective_rpm,
                        tpm_used + token_cost,
                        self.effective_tpm,
                        token_cost,
                    )
                    return

                waits: list[float] = []
                if rpm_used + 1 > self.effective_rpm and self._request_times:
                    waits.append(60 - (now - self._request_times[0]) + 0.05)
                if tpm_used + token_cost > self.effective_tpm and self._token_events:
                    waits.append(60 - (now - self._token_events[0][0]) + 0.05)
                wait_seconds = max(0.2, min(waits) if waits else 1.0)

                LOG.info(
                    'Rate limit reached; waiting %.1f seconds: %s rpm=%s/%s tpm_est=%s/%s request_tokens=%s',
                    wait_seconds,
                    label,
                    rpm_used,
                    self.effective_rpm,
                    tpm_used,
                    self.effective_tpm,
                    token_cost,
                )

            time.sleep(wait_seconds)


def load_config(config_path: str | None) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        if config_path:
            raise ValueError(f"Configuration file not found: {path}")
        return {}

    with path.open("rb") as file:
        config = tomllib.load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration format: {path}")
    return config


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return default


def as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def is_placeholder_api_key(value: str | None) -> bool:
    if not value:
        return True
    text = value.strip()
    return (
        not text
        or text.startswith("sk-xxxxxxxx")
        or "请填" in text
        or "YOUR_" in text.upper()
    )


def normalize_header(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def normalize_name(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value.strip().lower())


def is_meaningful_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if len(text) < 2:
        return False
    try:
        float(text.replace(",", ""))
        return False
    except ValueError:
        return True


def find_column(headers: list[str], requested: str | None, candidates: list[str]) -> int | None:
    if requested:
        for idx, header in enumerate(headers, 1):
            if header == requested:
                return idx
        raise ValueError(f"Specified column not found: {requested}")

    normalized_candidates = {normalize_name(name) for name in candidates}
    for idx, header in enumerate(headers, 1):
        if normalize_name(header) in normalized_candidates:
            return idx
    return None


def find_comment_column(ws: openpyxl.worksheet.worksheet.Worksheet, requested: str | None) -> int:
    headers = [normalize_header(cell.value) for cell in ws[1]]
    if requested:
        found = find_column(headers, requested, COMMENT_COLUMN_CANDIDATES)
        if found is None:
            raise ValueError(f"Specified comment column not found: {requested}")
        return found

    normalized_candidates = {normalize_name(name) for name in COMMENT_COLUMN_CANDIDATES}
    best_idx: int | None = None
    best_score = 0

    for idx, header in enumerate(headers, 1):
        if normalize_name(header) not in normalized_candidates:
            continue

        values = [ws.cell(row=row, column=idx).value for row in range(2, min(ws.max_row, 102) + 1)]
        text_count = sum(1 for value in values if is_meaningful_text(value))
        if text_count > best_score:
            best_idx = idx
            best_score = text_count

    if best_idx and best_score > 0:
        return best_idx

    header_text = "、".join(header for header in headers if header)
    raise ValueError(
        "No comment text column found for sentiment analysis. The Comments column appears "
        "to contain counts. Add or export a comment text column, such as Comment_Content.\n"
        f"Current headers: {header_text}"
    )


def ensure_result_columns(ws: openpyxl.worksheet.worksheet.Worksheet) -> dict[str, int]:
    headers = [normalize_header(cell.value) for cell in ws[1]]
    positions: dict[str, int] = {}
    added: list[str] = []

    for col_name in RESULT_COLUMNS:
        if col_name in headers:
            positions[col_name] = headers.index(col_name) + 1
            LOG.debug('Result column exists: %s -> column %s', col_name, positions[col_name])
            continue
        new_col = ws.max_column + 1
        ws.cell(row=1, column=new_col, value=col_name)
        positions[col_name] = new_col
        headers.append(col_name)
        added.append(f"{col_name}(column {new_col})")

    if added:
        LOG.info('Added result columns: %s', "、".join(added))
    else:
        LOG.info('All result columns already exist')

    return positions


def backfill_completed_markers(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    result_cols: dict[str, int],
    processed_at: str,
) -> int:
    filled = 0
    label_col = result_cols["sentiment_label"]
    status_col = result_cols["sentiment_status"]
    processed_col = result_cols["sentiment_processed_at"]

    for row_id in range(2, ws.max_row + 1):
        label_value = ws.cell(row=row_id, column=label_col).value
        status_value = ws.cell(row=row_id, column=status_col).value
        if label_value and normalize_header(status_value).lower() != "done":
            ws.cell(row=row_id, column=status_col, value="done")
            if not ws.cell(row=row_id, column=processed_col).value:
                ws.cell(row=row_id, column=processed_col, value=processed_at)
            filled += 1

    return filled


def collect_context(ws: openpyxl.worksheet.worksheet.Worksheet, row_id: int, headers: list[str]) -> dict[str, str]:
    context: dict[str, str] = {}
    wanted = {normalize_name(name) for name in CONTEXT_COLUMN_CANDIDATES}
    for idx, header in enumerate(headers, 1):
        if normalize_name(header) not in wanted:
            continue
        value = ws.cell(row=row_id, column=idx).value
        if value is not None and str(value).strip():
            context[header] = str(value).strip()[:300]
    return context


def build_items(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    comment_col: int,
    user_col: int | None,
    result_cols: dict[str, int],
    limit: int | None,
    resume: bool,
) -> list[CommentItem]:
    headers = [normalize_header(cell.value) for cell in ws[1]]
    items: list[CommentItem] = []
    skipped_done = 0
    skipped_non_text = 0
    scanned = 0
    total_rows = max(0, ws.max_row - 1)
    start_time = time.perf_counter()
    LOG.info(
        'Scanning comments: data rows=%s resume=%s limit=%s',
        total_rows,
        resume,
        limit if limit is not None else "none",
    )

    for row_id in range(2, ws.max_row + 1):
        scanned += 1
        status_value = ws.cell(row=row_id, column=result_cols["sentiment_status"]).value
        label_value = ws.cell(row=row_id, column=result_cols["sentiment_label"]).value
        if resume and (normalize_header(status_value).lower() == "done" or label_value):
            skipped_done += 1
            continue

        comment_value = ws.cell(row=row_id, column=comment_col).value
        if not is_meaningful_text(comment_value):
            skipped_non_text += 1
            continue

        items.append(
            CommentItem(
                row_id=row_id,
                comment=str(comment_value).strip(),
                user=None,  # usernames are never read or transmitted
                context=collect_context(ws, row_id, headers),
            )
        )
        if len(items) <= 3:
            LOG.debug('Example comment row=%s chars=%s', row_id, len(str(comment_value)))
        if scanned % 10000 == 0:
            LOG.info(
                'Scan progress: %s/%s rows, pending=%s, completed skipped=%s, empty/non-text skipped=%s',
                scanned,
                total_rows,
                len(items),
                skipped_done,
                skipped_non_text,
            )
        if limit is not None and len(items) >= limit:
            LOG.info('Reached limit=%s; stopping scan', limit)
            break

    LOG.info(
        'Scan complete: rows=%s pending=%s completed skipped=%s empty/non-text skipped=%s elapsed=%.2f seconds',
        scanned,
        len(items),
        skipped_done,
        skipped_non_text,
        time.perf_counter() - start_time,
    )
    return items


def chunked(items: list[CommentItem], size: int) -> list[list[CommentItem]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def compact_item(item: CommentItem) -> dict[str, Any]:
    return {
        "row_id": item.row_id,
        "comment": item.comment[:1200],
        "context": item.context,
    }


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def make_payload(
    batch: list[CommentItem],
    model: str,
    response_format: str,
    system_prompt: str,
    user_prompt_template: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    comments_json = json.dumps([compact_item(item) for item in batch], ensure_ascii=False, indent=2)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template.format(comments_json=comments_json)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    if response_format == "json_schema":
        payload["response_format"] = {"type": "json_schema", "json_schema": JSON_SCHEMA}
    elif response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}

    # For reasoning-capable models, sentiment classification is more stable and cheaper without thinking output.
    payload["enable_thinking"] = enable_thinking
    return payload


def estimate_batch_tokens(
    batch: list[CommentItem],
    model: str,
    response_format: str,
    system_prompt: str,
    user_prompt_template: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> int:
    payload = make_payload(
        batch,
        model=model,
        response_format=response_format,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    payload_tokens = rough_token_count(json.dumps(payload, ensure_ascii=False))
    expected_output = {
        "items": [
            {"row_id": item.row_id, "sentiment_label": "neutral", "sentiment_score": 0.0}
            for item in batch
        ]
    }
    output_tokens = rough_token_count(json.dumps(expected_output, ensure_ascii=False, separators=(",", ":")))
    return payload_tokens + output_tokens


def post_chat_completion(api_key: str, api_base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    LOG.debug('Sending API request: url=%s body_bytes=%s timeout=%s', url, len(data), timeout)
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiError(f"HTTP {exc.code}: {body}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise ApiError(str(exc), status=None) from exc


def analyze_batch(
    batch: list[CommentItem],
    api_key: str,
    api_base_url: str,
    model: str,
    response_format: str,
    system_prompt: str,
    user_prompt_template: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    timeout: int,
    max_retries: int,
    rate_limiter: MinuteRateLimiter | None,
    estimated_tokens: int,
    batch_label: str,
) -> dict[int, dict[str, Any]]:
    active_format = response_format
    last_error: Exception | None = None
    parse_failed = False
    batch_start = time.perf_counter()
    batch_chars = sum(len(item.comment) for item in batch)

    for attempt in range(1, max_retries + 1):
        try:
            payload = make_payload(
                batch,
                model=model,
                response_format=active_format,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            payload_text = json.dumps(payload, ensure_ascii=False)
            LOG.debug(
                'API request prepared: attempt=%s/%s response_format=%s comments=%s comment_chars=%s payload_chars=%s rough_tokens=%s',
                attempt,
                max_retries,
                active_format,
                len(batch),
                batch_chars,
                len(payload_text),
                rough_token_count(payload_text),
            )
            if rate_limiter:
                rate_limiter.acquire(estimated_tokens, f"{batch_label} attempt={attempt}")
            request_start = time.perf_counter()
            response = post_chat_completion(api_key, api_base_url, payload, timeout=timeout)
            LOG.debug('API response time: %.2f seconds', time.perf_counter() - request_start)
            content = response["choices"][0]["message"]["content"]
            LOG.debug(
                'API response: content_chars=%s rough_tokens=%s',
                len(content),
                rough_token_count(content),
            )
            parsed = extract_json(content)
            items = parsed.get("items", [])
            result = {int(item["row_id"]): item for item in items}
            LOG.info(
                'Batch parsed: returned %s/%s items, attempt=%s, elapsed=%.2f seconds',
                len(result),
                len(batch),
                attempt,
                time.perf_counter() - batch_start,
            )
            if len(result) != len(batch):
                missing = sorted({item.row_id for item in batch} - set(result))
                LOG.warning('Batch result count mismatch; missing row examples: %s', missing[:10])
            return result
        except ApiError as exc:
            last_error = exc
            if exc.status == 400 and active_format == "json_schema":
                active_format = "json_object"
                LOG.warning('json_schema request failed; subsequent retries use json_object')
            sleep_seconds = min(60, 2**attempt)
            LOG.warning(
                'API call failed: attempt=%s/%s status=%s; retrying after %s seconds. Error: %s',
                attempt,
                max_retries,
                exc.status if exc.status is not None else 'no HTTP status',
                sleep_seconds,
                exc,
            )
            time.sleep(sleep_seconds)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            parse_failed = True
            sleep_seconds = min(30, 2**attempt)
            LOG.warning(
                'Result parsing failed: attempt=%s/%s; retrying after %s seconds. Error: %s',
                attempt,
                max_retries,
                sleep_seconds,
                exc,
            )
            time.sleep(sleep_seconds)

    if parse_failed and not isinstance(last_error, ApiError):
        raise ResponseParseError(f"Batch response parsing failed: {last_error}") from last_error
    raise RuntimeError(f"Batch processing failed: {last_error}")


def analyze_batch_resilient(
    batch: list[CommentItem],
    api_key: str,
    api_base_url: str,
    model: str,
    response_format: str,
    system_prompt: str,
    user_prompt_template: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    timeout: int,
    max_retries: int,
    rate_limiter: MinuteRateLimiter | None,
    estimated_tokens: int,
    batch_label: str,
    min_split_batch_size: int,
) -> dict[int, dict[str, Any]]:
    try:
        return analyze_batch(
            batch=batch,
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            response_format=response_format,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
            estimated_tokens=estimated_tokens,
            batch_label=batch_label,
        )
    except ResponseParseError as exc:
        if len(batch) <= max(1, min_split_batch_size):
            LOG.error(
                'Batch parsing failed at minimum split size: %s comments=%s min_split_batch_size=%s',
                batch_label,
                len(batch),
                min_split_batch_size,
            )
            raise

        midpoint = len(batch) // 2
        left = batch[:midpoint]
        right = batch[midpoint:]
        LOG.warning(
            'Batch parsing failed; splitting and retrying: %s comments=%s -> %s + %s. Reason: %s',
            batch_label,
            len(batch),
            len(left),
            len(right),
            exc,
        )

        combined: dict[int, dict[str, Any]] = {}
        for part_name, part in (("A", left), ("B", right)):
            part_estimated_tokens = estimate_batch_tokens(
                part,
                model=model,
                response_format=response_format,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            combined.update(
                analyze_batch_resilient(
                    batch=part,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    model=model,
                    response_format=response_format,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    timeout=timeout,
                    max_retries=max_retries,
                    rate_limiter=rate_limiter,
                    estimated_tokens=part_estimated_tokens,
                    batch_label=f"{batch_label}.{part_name}",
                    min_split_batch_size=min_split_batch_size,
                )
            )
        return combined


def write_result(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    row_id: int,
    result_cols: dict[str, int],
    model: str,
    result: dict[str, Any],
) -> None:
    ws.cell(row=row_id, column=result_cols["sentiment_label"], value=result.get("sentiment_label"))
    ws.cell(row=row_id, column=result_cols["sentiment_score"], value=result.get("sentiment_score"))
    ws.cell(row=row_id, column=result_cols["sentiment_model"], value=model)
    ws.cell(row=row_id, column=result_cols["sentiment_status"], value="done")
    ws.cell(
        row=row_id,
        column=result_cols["sentiment_processed_at"],
        value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze spreadsheet comment sentiment with a locally deployed OpenAI-compatible LLM endpoint.")
    parser.add_argument("--config", help="TOML config path. Defaults to config/sentiment_config.toml (or the .example template).")
    parser.add_argument("--input", help="Input .xlsx path. Overrides config run.input.")
    parser.add_argument("--output", help="Output .xlsx path. Defaults to *_sentiment.xlsx.")
    parser.add_argument("--sheet", help="Worksheet name. Defaults to the first sheet.")
    parser.add_argument("--comment-column", help="Column header containing comment text.")
    parser.add_argument("--model", help="Model name as served by the local endpoint. Overrides config model.name.")
    parser.add_argument("--batch-size", type=int, help="Comments per API request. Overrides config run.batch_size.")
    parser.add_argument("--limit", type=int, help="Only process first N pending comments.")
    parser.add_argument("--timeout", type=int, help="API timeout seconds. Overrides config run.timeout.")
    parser.add_argument("--max-retries", type=int, help="Retry count. Overrides config run.max_retries.")
    parser.add_argument("--save-every", type=int, help="Save after every N batches. Overrides config run.save_every.")
    parser.add_argument(
        "--response-format",
        choices=["json_schema", "json_object", "none"],
        help="Response format mode. Overrides config model.response_format.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Do not skip rows that already have labels.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect columns and pending rows without API calls.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    parser.add_argument("--log-file", help="Optional UTF-8 log file path.")
    parser.add_argument("--max-workers", type=int, help="Concurrent API worker threads.")
    parser.add_argument("--min-split-batch-size", type=int, help="Smallest batch size after parse-failure splitting.")
    parser.add_argument("--max-rpm", type=int, help="Rate limit: requests per minute.")
    parser.add_argument("--max-tpm", type=int, help="Rate limit: estimated tokens per minute.")
    parser.add_argument("--rate-safety-factor", type=float, help="Use this fraction of RPM/TPM, e.g. 0.9.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    try:
        config = load_config(args.config)
    except ValueError as exc:
        setup_logging(args.log_level or "INFO", args.log_file, run_timestamp)
        LOG.error("%s", exc)
        return 2

    api_config = section(config, "api")
    model_config = section(config, "model")
    run_config = section(config, "run")
    prompts_config = section(config, "prompts")
    logging_config = section(config, "logging")
    rate_limit_config = section(config, "rate_limit")

    log_level = str(first_present(args.log_level, logging_config.get("level"), default="INFO"))
    log_file = first_present(args.log_file, logging_config.get("file"))
    actual_log_path = setup_logging(log_level, str(log_file) if log_file else None, run_timestamp)
    program_start = time.perf_counter()
    LOG.info('Starting LLM comment sentiment analysis (local OpenAI-compatible endpoint)')
    LOG.info('Run timestamp: %s', run_timestamp)
    LOG.info('Configuration file: %s (exists=%s)', config_path_for_display(args.config), bool(config))

    input_value = first_present(args.input, run_config.get("input"))
    if not input_value:
        LOG.error('Specify the input Excel file using --input or run.input in the configuration.')
        return 2

    input_path = Path(str(input_value)).expanduser().resolve()
    output_value = first_present(args.output, run_config.get("output"))
    base_output_path = Path(str(output_value)).expanduser().resolve() if output_value else input_path.with_name(
        f"{input_path.stem}_sentiment.xlsx"
    )
    resume_from_latest = as_bool(run_config.get("resume_from_latest"), default=True) and not args.no_resume
    resume_output_path = find_latest_timestamped_file(base_output_path) if resume_from_latest else None
    workbook_path = resume_output_path if resume_output_path else input_path
    output_path = resume_output_path if resume_output_path else prefix_timestamp(base_output_path, run_timestamp)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sheet_name = first_present(args.sheet, run_config.get("sheet"))
    comment_column = first_present(args.comment_column, run_config.get("comment_column"))
    limit = first_present(args.limit, run_config.get("limit"))
    batch_size = int(first_present(args.batch_size, run_config.get("batch_size"), default=20))
    max_workers = int(first_present(args.max_workers, run_config.get("max_workers"), default=16))
    min_split_batch_size = int(
        first_present(args.min_split_batch_size, run_config.get("min_split_batch_size"), default=25)
    )
    timeout = int(first_present(args.timeout, run_config.get("timeout"), default=120))
    max_retries = int(first_present(args.max_retries, run_config.get("max_retries"), default=4))
    save_every = int(first_present(args.save_every, run_config.get("save_every"), default=5))
    response_format = str(
        first_present(args.response_format, model_config.get("response_format"), default="json_object")
    )
    model = str(first_present(args.model, os.getenv("LLM_MODEL"), model_config.get("name"), default=DEFAULT_MODEL))
    temperature = float(first_present(model_config.get("temperature"), default=0.1))
    max_tokens = int(first_present(model_config.get("max_tokens"), default=4096))
    enable_thinking = as_bool(model_config.get("enable_thinking"), default=False)
    api_base_url = str(first_present(os.getenv("LLM_API_BASE_URL"), api_config.get("base_url"), default=API_BASE_URL))
    system_prompt = str(first_present(prompts_config.get("system_prompt"), default=SYSTEM_PROMPT))
    user_prompt_template = str(first_present(prompts_config.get("user_prompt_template"), default=USER_PROMPT_TEMPLATE))
    rate_limit_enabled = as_bool(rate_limit_config.get("enabled"), default=True)
    max_rpm = int(first_present(args.max_rpm, rate_limit_config.get("max_rpm"), default=500))
    max_tpm = int(first_present(args.max_tpm, rate_limit_config.get("max_tpm"), default=2_000_000))
    rate_safety_factor = float(
        first_present(args.rate_safety_factor, rate_limit_config.get("safety_factor"), default=0.9)
    )

    LOG.info('Input file: %s', input_path)
    LOG.info('Output filename template: %s', base_output_path)
    LOG.info('Resume from latest result file: %s', 'enabled' if resume_from_latest else 'disabled')
    if resume_output_path:
        LOG.info('Resumable result file found: %s', resume_output_path)
    else:
        LOG.info('No resumable result file found; creating a timestamped result file')
    LOG.info('Workbook to read: %s', workbook_path)
    LOG.info('Output file: %s', output_path)
    if actual_log_path:
        LOG.info('Log file for this run: %s', actual_log_path)
    LOG.info('Worksheet: %s', sheet_name if sheet_name else 'first worksheet')
    LOG.info('Comment column configuration: %s', comment_column if comment_column else 'automatic detection')
    LOG.info('User columns: neither read nor sent (privacy protection)')
    LOG.info(
        'Model configuration: model=%s response_format=%s temperature=%s max_tokens=%s enable_thinking=%s',
        model,
        response_format,
        temperature,
        max_tokens,
        enable_thinking,
    )
    LOG.info(
        'Run settings: batch_size=%s max_workers=%s min_split_batch_size=%s limit=%s timeout=%s max_retries=%s save_every=%s resume=%s dry_run=%s',
        batch_size,
        max_workers,
        min_split_batch_size,
        limit if limit is not None else "none",
        timeout,
        max_retries,
        save_every,
        not args.no_resume,
        args.dry_run,
    )
    LOG.info("API base_url:%s", api_base_url)
    LOG.info(
        'Rate limit settings: enabled=%s max_rpm=%s max_tpm=%s safety_factor=%s effective_rpm=%s effective_tpm=%s',
        rate_limit_enabled,
        max_rpm,
        max_tpm,
        rate_safety_factor,
        max(1, int(max_rpm * min(max(rate_safety_factor, 0.1), 1.0))),
        max(1, int(max_tpm * min(max(rate_safety_factor, 0.1), 1.0))),
    )
    LOG.debug("System prompt chars=%s rough_tokens=%s", len(system_prompt), rough_token_count(system_prompt))
    LOG.debug(
        "User prompt template chars=%s rough_tokens=%s",
        len(user_prompt_template),
        rough_token_count(user_prompt_template),
    )

    if not input_path.exists():
        LOG.error('Input file not found: %s', input_path)
        return 2
    if not workbook_path.exists():
        LOG.error('Workbook to read not found: %s', workbook_path)
        return 2

    load_start = time.perf_counter()
    LOG.info('Reading workbook')
    wb = openpyxl.load_workbook(workbook_path)
    restore_source_schema(wb, input_path)
    ws = wb[str(sheet_name)] if sheet_name else wb[wb.sheetnames[0]]
    headers = [normalize_header(cell.value) for cell in ws[1]]
    LOG.info('Workbook loaded: sheets=%s', wb.sheetnames)
    LOG.info('Current worksheet: %s, rows=%s, columns=%s', ws.title, ws.max_row, ws.max_column)
    LOG.info('Headers: %s', " | ".join(headers))
    log_duration('Read workbook', load_start)

    try:
        comment_col = find_comment_column(ws, str(comment_column) if comment_column else None)
        user_col = None
    except ValueError as exc:
        LOG.error("%s", exc)
        return 2
    LOG.info('Detected comment column: %s -> column %s', ws.cell(row=1, column=comment_col).value, comment_col)
    result_cols = ensure_result_columns(ws)
    LOG.info('Result column positions: %s', result_cols)
    if not args.no_resume:
        marker_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        backfilled = backfill_completed_markers(ws, result_cols, marker_time)
        LOG.info('Restored resume markers for %s rows', backfilled)

    build_start = time.perf_counter()
    items = build_items(
        ws=ws,
        comment_col=comment_col,
        user_col=user_col,
        result_cols=result_cols,
        limit=int(limit) if limit is not None else None,
        resume=not args.no_resume,
    )
    log_duration('Build pending comment list', build_start)

    LOG.info('Pending comments: %s', len(items))

    if args.dry_run:
        preview = [compact_item(item) for item in items[:3]]
        LOG.info('Dry-run mode: no API calls or result files')
        LOG.info('First three pending examples:\n%s', json.dumps(preview, ensure_ascii=False, indent=2))
        log_duration('Full pipeline dry-run', program_start)
        return 0

    api_key = first_present(os.getenv("LLM_API_KEY"), api_config.get("api_key"))
    if is_placeholder_api_key(str(api_key) if api_key is not None else None):
        # Local OpenAI-compatible servers normally accept any bearer token.
        api_key = LOCAL_PLACEHOLDER_API_KEY
        LOG.info('No API key supplied; using local endpoint placeholder %r (local servers usually do not validate it)', api_key)
    else:
        LOG.info('API key: %s (redacted)', safe_api_key_label(str(api_key)))

    batches = chunked(items, max(1, batch_size))
    LOG.info('Batch plan: %s batches, at most %s comments per batch', len(batches), batch_size)
    if not batches:
        LOG.info('No pending batches; saving output file')
    else:
        batch_estimates = [
            estimate_batch_tokens(
                batch,
                model=model,
                response_format=response_format,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            for batch in batches
        ]
        total_estimated_tokens = sum(batch_estimates)
        LOG.info(
            'Estimated batch tokens: total=%s mean/batch=%.0f max/batch=%s',
            total_estimated_tokens,
            total_estimated_tokens / len(batch_estimates),
            max(batch_estimates),
        )
        LOG.info(
            'Concurrency plan: max_workers=%s; rate-limited theoretical maximum approximately %.1f batches/minute (estimated from mean tokens)',
            max_workers,
            min(
                max_rpm * rate_safety_factor,
                (max_tpm * rate_safety_factor) / max(1, total_estimated_tokens / len(batch_estimates)),
            )
            if rate_limit_enabled
            else float(max_workers),
        )

        rate_limiter = MinuteRateLimiter(
            max_rpm=max_rpm,
            max_tpm=max_tpm,
            safety_factor=rate_safety_factor,
            enabled=rate_limit_enabled,
        )

        completed_batches = 0
        completed_comments = 0
        failed = False
        futures = {}
        pool_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for batch_index, batch in enumerate(batches, 1):
                estimated_tokens = batch_estimates[batch_index - 1]
                batch_label = f"batch={batch_index}/{len(batches)} rows={batch[0].row_id}-{batch[-1].row_id}"
                LOG.info(
                    'Submitting batch %s/%s: rows %s-%s, comments=%s, estimated_tokens=%s',
                    batch_index,
                    len(batches),
                    batch[0].row_id,
                    batch[-1].row_id,
                    len(batch),
                    estimated_tokens,
                )
                future = executor.submit(
                    analyze_batch_resilient,
                    batch=batch,
                    api_key=str(api_key),
                    api_base_url=api_base_url,
                    model=model,
                    response_format=response_format,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    timeout=timeout,
                    max_retries=max_retries,
                    rate_limiter=rate_limiter,
                    estimated_tokens=estimated_tokens,
                    batch_label=batch_label,
                    min_split_batch_size=min_split_batch_size,
                )
                futures[future] = (batch_index, batch, estimated_tokens, time.perf_counter())

            for future in as_completed(futures):
                batch_index, batch, estimated_tokens, batch_start = futures[future]
                try:
                    results = future.result()
                except Exception as exc:
                    failed = True
                    LOG.exception('Batch %s/%s failed; saving progress and stopping: %s', batch_index, len(batches), exc)
                    cancelled = 0
                    for pending_future in futures:
                        if pending_future is not future and pending_future.cancel():
                            cancelled += 1
                    LOG.warning('Attempted to cancel %s pending batch tasks; waiting for running tasks to finish', cancelled)
                    break

                missing_count = 0
                for item in batch:
                    result = results.get(item.row_id)
                    if result is None:
                        missing_count += 1
                        result = {
                            "row_id": item.row_id,
                            "sentiment_label": "uncertain",
                            "sentiment_score": 0,
                        }
                    write_result(ws, item.row_id, result_cols, model, result)

                completed_batches += 1
                completed_comments += len(batch)
                elapsed = time.perf_counter() - pool_start
                LOG.info(
                    'Batch %s/%s written: results=%s missing filled as uncertain=%s estimated_tokens=%s elapsed=%.2f seconds; progress %s/%s batches, %s/%s comments, overall %.1f comments/minute',
                    batch_index,
                    len(batches),
                    len(results),
                    missing_count,
                    estimated_tokens,
                    time.perf_counter() - batch_start,
                    completed_batches,
                    len(batches),
                    completed_comments,
                    len(items),
                    (completed_comments / elapsed * 60) if elapsed > 0 else 0,
                )

                if completed_batches % max(1, save_every) == 0:
                    save_start = time.perf_counter()
                    save_english_workbook(wb, output_path)
                    LOG.info(
                        'Progress saved: %s (%s/%s batches completed, elapsed=%.2f seconds)',
                        output_path,
                        completed_batches,
                        len(batches),
                        time.perf_counter() - save_start,
                    )

        if failed:
            save_start = time.perf_counter()
            save_english_workbook(wb, output_path)
            LOG.info('Progress before failure saved: %s (elapsed=%.2f seconds)', output_path, time.perf_counter() - save_start)
            return 1

    save_start = time.perf_counter()
    save_english_workbook(wb, output_path)
    LOG.info('Final results saved: %s (elapsed=%.2f seconds)', output_path, time.perf_counter() - save_start)
    log_duration('Full pipeline', program_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
