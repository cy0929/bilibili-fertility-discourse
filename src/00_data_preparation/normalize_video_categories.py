# -*- coding: utf-8 -*-
"""Normalise the hand-coded video category column to English labels.

Early exports of the video table use a legacy Chinese category header and Chinese
label values. The analysis scripts accept both, but this utility rewrites a workbook
into the canonical form:

    analytic_category = Micro Individual Experience and Lifestyle | Gender Issues and
    Rights Discussion | Macro Policy and Social Development | Economic Foundation and
    Class Survival | Pan-Entertainment and Cultural Projection

Only the category column is touched; every other cell keeps its original value. Values
that are neither a known Chinese label nor a known English label are left untouched and
reported at the end.

Usage:
    # write <stem>_en.xlsx next to the source file
    python src/00_data_preparation/normalize_video_categories.py data/raw/video_info.xlsx

    # choose the output path
    python src/00_data_preparation/normalize_video_categories.py INPUT.xlsx --output OUT.xlsx

    # rewrite the source file (a timestamped .bak.xlsx copy is written first)
    python src/00_data_preparation/normalize_video_categories.py INPUT.xlsx --in-place

See data/README.md for the five category definitions.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook

# Reuse the canonical vocabulary instead of restating it here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_categories import (  # noqa: E402
    CATEGORY_COLUMN,
    CATEGORY_COLUMNS,
    CATEGORY_LABELS,
    CATEGORY_LABELS_ZH_TO_EN,
)


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


_ENGLISH_LABELS = set(CATEGORY_LABELS)


def find_category_column(worksheet) -> int:
    """Return the 1-based index of the category column in the header row."""
    headers = [cell.value for cell in worksheet[1]]
    for name in CATEGORY_COLUMNS:
        if name in headers:
            return headers.index(name) + 1
    raise SystemExit(
        f"[error] Video category column not found; expected one of {list(CATEGORY_COLUMNS)}; headers: {headers}"
    )


def convert(src: Path, dst: Path, rename_column: bool = True) -> dict:
    """Rewrite the category values (and optionally the header) into ``dst``."""
    workbook = load_workbook(src)
    sheet = workbook[workbook.sheetnames[0]]
    if len(workbook.sheetnames) > 1:
        print(f"[warn] Workbook has multiple sheets; processing only the first: {workbook.sheetnames[0]}")

    index = find_category_column(sheet)
    stats = {"sheet": sheet.title, "rows": 0, "converted": 0, "already_en": 0, "unknown": set()}

    for row in range(2, sheet.max_row + 1):
        cell = sheet.cell(row=row, column=index)
        stats["rows"] += 1
        if not isinstance(cell.value, str):
            continue
        key = cell.value.strip()
        mapped = CATEGORY_LABELS_ZH_TO_EN.get(key)
        if mapped is not None:
            if mapped != cell.value:
                cell.value = mapped
                stats["converted"] += 1
            else:
                stats["already_en"] += 1
        elif key in _ENGLISH_LABELS:
            stats["already_en"] += 1
        else:
            stats["unknown"].add(key)

    if rename_column:
        sheet.cell(row=1, column=index).value = CATEGORY_COLUMN

    save_english_workbook(workbook, dst)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Normalize video analytic category values to English (writes a new file by default)'
    )
    parser.add_argument("input", type=Path, help='Source video information workbook')
    parser.add_argument("--output", type=Path, default=None, help='Output path; default: <stem>_en.xlsx')
    parser.add_argument("--in-place", action="store_true", help='Overwrite the source after creating a .bak.xlsx backup')
    parser.add_argument(
        "--keep-column-name",
        action="store_true",
        help='Accepted for CLI compatibility; exported category headers are always English',
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"[error] File not found: {args.input}", file=sys.stderr)
        return 1

    rename = not args.keep_column_name

    if args.in_place:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup = args.input.with_suffix("")
        backup = backup.parent / f"{backup.name}.{stamp}.bak.xlsx"
        backup.write_bytes(args.input.read_bytes())
        print(f"[backup] {backup}")
        target = args.input
    else:
        target = args.output or args.input.with_name(f"{args.input.stem}_en.xlsx")

    stats = convert(args.input, target, rename_column=rename)

    print(f"[sheet]     {stats['sheet']}")
    print(f"[rows]      {stats['rows']}")
    print(f"[converted] {stats['converted']}  (Chinese -> English)")
    print(f"[already]   {stats['already_en']}  (already English)")
    print(f"[out]       {target}")
    if stats["unknown"]:
        print(f"[warn] Unrecognized values (preserved): {sorted(stats['unknown'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
