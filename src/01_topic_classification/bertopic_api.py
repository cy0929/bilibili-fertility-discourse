"""
Comment topic classification - embedding-API version
============================================================================
Embeddings are requested from a locally deployed, OpenAI-compatible embedding
endpoint (e.g. vLLM / SGLang / Ollama serving Qwen3-Embedding-8B). Comment text is
sent only to that local endpoint; nothing leaves the machine unless you point
LLM_API_BASE_URL at a remote server yourself.
The rest of the pipeline (comment cleaning / optional UMAP+HDBSCAN clustering /
c-TF-IDF keywords / noise labeling) is unchanged.

Highlights:
    - Concurrent requests (ThreadPoolExecutor)
    - Automatic retry (exponential backoff; handles 429/5xx/timeouts/network jitter)
    - Disk cache (keyed by text hash, so re-tuning does not cost money twice)
    - Long-text truncation (stops one very long comment from dragging down a batch)
    - Progress bar

Usage:
    # Point the script at the local embedding server (default http://localhost:8000/v1)
    export LLM_API_BASE_URL=http://localhost:8000/v1
    python src/01_topic_classification/bertopic_api.py --config config/bertopic_config.yaml

Dependencies:
    pip install pyyaml openai bertopic sentence-transformers \\
                hdbscan umap-learn jieba pandas openpyxl tqdm
============================================================================
"""

# Deferred evaluation of type annotations, so `X | None` style also works on 3.9
from __future__ import annotations

# ---- standard library ----
import argparse                                # command-line parsing (here only for --config)
import glob                                    # wildcard matching of input files
import hashlib                                 # SHA-1 text hash, used as the cache key
import logging                                 # structured logging
import os                                      # path joining / existence checks / mkdir
import re                                      # regex: text normalization + token filtering
import time                                    # sleep during retries
from contextlib import contextmanager          # _timer context manager
from datetime import datetime                  # timestamps
from concurrent.futures import ThreadPoolExecutor, as_completed   # concurrent requests
from types import SimpleNamespace              # wrap dicts into dot-accessible objects
from typing import Any, Callable, Dict, Generator, List, Sequence, Set, Tuple

# ---- third-party libraries ----
import jieba                                   # Chinese word segmentation
import numpy as np                             # numerical computation
import pandas as pd                            # table reading / comment cleaning
import yaml                                    # load the YAML config
from sklearn.cluster import MiniBatchKMeans    # second-level clustering inside a first-level topic


class BaseEmbedder:
    """Lightweight fallback: with run_bertopic=false the script need not import BERTopic/UMAP."""
    def __init__(self, *args, **kwargs) -> None:
        pass

# openai SDK: local servers (vLLM / SGLang / Ollama / LM Studio) speak the OpenAI-compatible protocol
try:
    from openai import OpenAI
    # both old and new SDKs expose these exception classes; import them defensively
    from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError
except ImportError as e:
    raise ImportError(
        "需要安装 openai SDK：pip install openai\n"
        "原始错误: " + str(e)
    )

# tqdm progress bar; fall back to a no-op implementation if it is not installed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):   # noqa: D401
        return it


# Default config lookup order:
#   1. environment variable BFD_BERTOPIC_CONFIG
#   2. <repo_root>/config/bertopic_config.yaml        (the user's own config, gitignored)
#   3. <repo_root>/config/bertopic_config.example.yaml(example template shipped with the repo)
#   4. bertopic_config_api.yaml next to this script    (compat with the old flat layout)
def _resolve_default_config() -> str:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
    _candidates = [
        os.environ.get("BFD_BERTOPIC_CONFIG", ""),
        os.path.join(_repo_root, "config", "bertopic_config.yaml"),
        os.path.join(_repo_root, "config", "bertopic_config.example.yaml"),
        os.path.join(_script_dir, "bertopic_config_api.yaml"),
    ]
    for _path in _candidates:
        if _path and os.path.isfile(_path):
            return _path
    return _candidates[1]  # if none exist, give the most intuitive expected path and let loading fail later


DEFAULT_CONFIG_PATH = _resolve_default_config()

# =============================================================================
# Logging configuration
# =============================================================================

logger = logging.getLogger("bertopic_api")


def setup_logging(log_file: str | None = None) -> None:
    """Init logging: console at INFO, file at DEBUG (if a path is given)."""
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    if log_file:
        ensure_dir(os.path.dirname(log_file) or ".")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"日志同步写入文件: {log_file}")


@contextmanager
def _timer(label: str) -> Generator[None, None, None]:
    """Timing context manager; logs step start/end and elapsed time."""
    logger.info(f"▶ {label}")
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    mins, secs = divmod(elapsed, 60)
    if mins >= 1:
        logger.info(f"✓ {label} 完成，耗时 {int(mins)}m {secs:.1f}s")
    else:
        logger.info(f"✓ {label} 完成，耗时 {elapsed:.1f}s")


# =============================================================================
# Config loading
# =============================================================================

def _dict_to_ns(obj: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace, so cfg.api.model works as attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_ns(v) for v in obj]
    return obj


def load_config(path: str) -> SimpleNamespace:
    """
    Read YAML -> SimpleNamespace. Also:
      1. convert the three vocabulary lists into sets
      2. when api.api_key is blank, try to read it from an environment variable
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到配置文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _dict_to_ns(raw)

    # turn the vocabularies into sets (O(1) membership tests)
    cfg.forced_irrelevant_keywords = set(getattr(cfg, "forced_irrelevant_keywords", []) or [])
    cfg.social_chatter_tokens = set(getattr(cfg, "social_chatter_tokens", []) or [])
    cfg.research_signal_tokens = set(getattr(cfg, "research_signal_tokens", []) or [])

    # Endpoint: env var LLM_API_BASE_URL overrides YAML; default is a local server.
    base_url = (os.environ.get("LLM_API_BASE_URL") or getattr(cfg.api, "base_url", "") or "").strip()
    cfg.api.base_url = base_url or "http://localhost:8000/v1"

    # API Key: env var LLM_API_KEY, then YAML. Local OpenAI-compatible servers normally
    # accept any bearer token, so an empty value falls back to a placeholder.
    api_key = (os.environ.get("LLM_API_KEY") or getattr(cfg.api, "api_key", "") or "").strip()
    cfg.api.api_key = api_key or "EMPTY"
    return cfg


# =============================================================================
# File reading / vocabulary loading
# =============================================================================

def read_table(file_path: str) -> pd.DataFrame:
    """Dispatch on the file extension to the matching pandas reader. Only xlsx/xls/csv/tsv."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if ext == ".csv":
        return pd.read_csv(file_path)
    if ext == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    raise ValueError(f"不支持的文件格式: {ext}，仅支持 xlsx/xls/csv/tsv")


def load_stopwords(path: str) -> Set[str]:
    """Load the stopword list: one word per line, lines starting with # are comments;
    an invalid path yields an empty set."""
    stopwords: Set[str] = set()
    if not path or not os.path.exists(path):
        return stopwords
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith("#"):
                stopwords.add(w)
    return stopwords


def load_keywords(path: str) -> Set[str]:
    """Load the keyword list; same logic as load_stopwords."""
    keywords: Set[str] = set()
    if not path or not os.path.exists(path):
        return keywords
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith("#"):
                keywords.add(w)
    return keywords


# =============================================================================
# Text normalization / tokenization
# =============================================================================

def normalize_text(text: str) -> str:
    """Clean raw comment text: strip @-mentions, links, emoji markers, and squeeze whitespace."""
    t = str(text)
    t = re.sub(r"回复\s*@[^:：\n]+[:：]?", " ", t)         # Bilibili `回复 @xx:` prefix
    t = re.sub(r"https?://\S+", " ", t)                    # http(s) links
    t = re.sub(r"\[[^\]]{1,20}\]", " ", t)                 # `[doge]` `[笑哭]` emoji markers
    t = re.sub(r"\s+", " ", t)                             # squeeze repeated whitespace
    return t.strip()


def should_drop_token(token: str, stopwords: Set[str]) -> bool:
    """Decide whether to drop a token; True = drop. Rules: empty/stopword/platform noise/digit-letter garbage codes/pure punctuation."""
    t = token.strip()
    if not t:                                                       return True
    if t in stopwords:                                              return True
    if re.fullmatch(r"(?i)up[a-z0-9]{1,}", t):                      return True   # up123/67up
    if re.fullmatch(r"(?i)bv[a-z0-9]{6,}", t):                      return True   # BV1xx...
    if re.fullmatch(r"(?i)(?:\d+[a-z]|[a-z]\d+|\d+[a-z]\d+|\d+[a-z]{1,2}|[a-z]{1,2}\d+)", t):  return True
    if re.fullmatch(r"[0-9]+([.:/-][0-9]+)*", t):                   return True   # timestamps/versions
    if re.fullmatch(r"[A-Za-z_]+", t):                              return True   # pure English
    if re.fullmatch(r"[A-Za-z0-9]{7,}", t):                         return True   # long hashes
    if re.search(r"[A-Za-z]", t) and re.search(r"[0-9]", t) and len(t) >= 5:  return True
    if not re.search(r"[\u4e00-\u9fff]", t) and len(t) <= 5:        return True   # short non-Chinese
    if re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):                return True   # pure punctuation
    return False


def build_tokenizer(stopwords: Set[str]) -> Callable[[str], List[str]]:
    """Tokenizer for the filtering step: jieba precise mode + should_drop_token filtering."""
    def _tokenize(text: str) -> List[str]:
        words = jieba.lcut(text, cut_all=False)
        return [w for w in words if not should_drop_token(w, stopwords)]
    return _tokenize


def build_vectorizer_tokenizer(stopwords: Set[str]) -> Callable[[str], List[str]]:
    """For CountVectorizer: adds one more constraint, "must contain Chinese characters"."""
    def _tokenize(text: str) -> List[str]:
        words = jieba.lcut(text, cut_all=False)
        out: List[str] = []
        for w in words:
            w = w.strip()
            if not w:                                            continue
            if w in stopwords:                                   continue
            if not re.search(r"[\u4e00-\u9fff]", w):             continue
            if re.fullmatch(r"[\W_]+", w, flags=re.UNICODE):     continue
            if re.fullmatch(r"[0-9]+([.:/-][0-9]+)*", w):        continue
            out.append(w)
        return out
    return _tokenize


def build_vectorizer_tokenizer_fallback(stopwords: Set[str]) -> Callable[[str], List[str]]:
    """Vectorizer fallback: relaxes "must contain Chinese characters", used when the vocabulary is too sparse."""
    def _tokenize(text: str) -> List[str]:
        words = jieba.lcut(text, cut_all=False)
        out: List[str] = []
        for w in words:
            w = w.strip()
            if not w:                                            continue
            if w in stopwords:                                   continue
            if re.fullmatch(r"[\W_]+", w, flags=re.UNICODE):     continue
            if re.fullmatch(r"[0-9]+([.:/-][0-9]+)*", w):        continue
            if re.fullmatch(r"(?i)up[a-z0-9]{1,}", w):           continue
            if re.fullmatch(r"(?i)bv[a-z0-9]{6,}", w):           continue
            out.append(w)
        return out
    return _tokenize


def clean_topic_keyword(phrase: str, stopwords: Set[str]) -> str:
    """Clean the keywords produced by BERTopic (split bigrams + token filtering)."""
    parts = [p.strip() for p in re.split(r"[\s,，、/|]+", str(phrase)) if p.strip()]
    kept = [p for p in parts if not should_drop_token(p, stopwords)]
    cleaned = " ".join(kept).strip()
    if not cleaned or not re.search(r"[\u4e00-\u9fff]", cleaned):
        return ""
    return cleaned


# =============================================================================
# Comment collection
# =============================================================================

def collect_comments(
    files: Sequence[str],
    include_sub_replies: bool,
    min_likes: int,
) -> pd.DataFrame:
    """Walk the input files and assemble main comments + (optional) second-level replies into a DataFrame with a unified schema."""
    records: List[Dict[str, object]] = []

    for file_idx, f in enumerate(files, start=1):
        logger.info(f"[读取 {file_idx}/{len(files)}] {os.path.basename(f)}")
        df = read_table(f)
        logger.debug(f"  → 原始行数: {len(df)}")
        # fill in missing required columns so row.get never raises KeyError
        needed_cols = [
            "Video_ID", "Title", "Comment_ID", "Comment_Content", "Comment_Likes",
            "Sub_C_Reply_ID", "Sub_C_Reply_Content", "Sub_C_Reply_Likes",
        ]
        for col in needed_cols:
            if col not in df.columns:
                df[col] = ""

        for _, row in df.iterrows():
            # main comment
            main_text = normalize_text(row.get("Comment_Content", ""))
            main_likes = int(pd.to_numeric(row.get("Comment_Likes", 0), errors="coerce") or 0)
            if main_text and main_likes >= min_likes:
                records.append({
                    "source_file": os.path.basename(f),
                    "video_id": str(row.get("Video_ID", "")),
                    "title": str(row.get("Title", "")),
                    "root_comment_id": str(row.get("Comment_ID", "")),
                    "reply_id": "",
                    "comment_type": "main",
                    "likes": main_likes,
                    "text": main_text,
                })

            # second-level reply (optional)
            if include_sub_replies:
                reply_id = str(row.get("Sub_C_Reply_ID", "")).strip()
                if reply_id and reply_id != "无":
                    reply_text = normalize_text(row.get("Sub_C_Reply_Content", ""))
                    reply_likes = int(pd.to_numeric(row.get("Sub_C_Reply_Likes", 0), errors="coerce") or 0)
                    if reply_text and reply_likes >= min_likes:
                        records.append({
                            "source_file": os.path.basename(f),
                            "video_id": str(row.get("Video_ID", "")),
                            "title": str(row.get("Title", "")),
                            "root_comment_id": str(row.get("Comment_ID", "")),
                            "reply_id": reply_id,
                            "comment_type": "reply",
                            "likes": reply_likes,
                            "text": reply_text,
                        })

    out = pd.DataFrame(records)
    if out.empty:
        return out
    before_dedup = len(out)
    out = out.drop_duplicates(subset=["comment_type", "root_comment_id", "reply_id", "text"])
    out = out.reset_index(drop=True)
    out.insert(0, "doc_id", out.index)
    logger.info(f"[收集] 原始记录 {before_dedup} 条，去重后 {len(out)} 条")
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_dataframe_excel(df: pd.DataFrame, path: str) -> None:
    """Export a DataFrame to xlsx in one consistent way."""
    ensure_dir(os.path.dirname(path) or ".")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)


def parse_nr_topics(value: Any) -> object:
    """Parse nr_topics: 'auto' / int / numeric string."""
    if isinstance(value, int):
        return value
    v = str(value).strip().lower()
    if v == "auto":
        return "auto"
    return int(v)


# =============================================================================
# Filtering helpers
# =============================================================================

def should_drop_social_chatter(tokens, drop_social_chatter, social_chatter_tokens, research_signal_tokens):
    """Social-chatter detection: drop only if a chatter token hits, no research signal, and <= 8 tokens."""
    if not drop_social_chatter:
        return False
    token_set = set(tokens)
    chatter_hits = len(token_set & social_chatter_tokens)
    signal_hits = len(token_set & research_signal_tokens)
    return chatter_hits >= 1 and signal_hits == 0 and len(tokens) <= 8


def contains_research_signal(text: str, research_signal_tokens: Set[str]) -> bool:
    return any(keyword in text for keyword in research_signal_tokens)


def contains_any_keyword(text: str, keywords: Set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def should_keep_research_doc(text, research_keywords, research_signal_tokens):
    hard_reject_keywords = {"皇室", "耀祖", "宫内", "日本", "皇位", "爱子", "公主", "皇太子", "储君"}
    if contains_any_keyword(text, hard_reject_keywords):
        return False
    return contains_research_signal(text, research_signal_tokens)


def classify_confidence(p: float) -> str:
    if p >= 0.5:   return "high"
    if p >= 0.15:  return "mid"
    return "low"


def extract_top2_from_probs(probs: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract the top-1 / top-2 topics and probabilities from an (N, K) probability matrix."""
    if probs is None or probs.ndim != 2 or probs.shape[1] == 0:
        row_count = 0 if probs is None or probs.ndim == 0 else int(probs.shape[0])
        empty = np.full((row_count,), np.nan)
        return {
            "top1_topic": np.full((row_count,), -1, dtype=int),
            "top1_prob": empty,
            "top2_topic": np.full((row_count,), -1, dtype=int),
            "top2_prob": empty,
        }
    top_indices = np.argsort(-probs, axis=1)
    top1_idx = top_indices[:, 0]
    top1_prob = probs[np.arange(probs.shape[0]), top1_idx]
    if probs.shape[1] >= 2:
        top2_idx = top_indices[:, 1]
        top2_prob = probs[np.arange(probs.shape[0]), top2_idx]
        top2_topic = top2_idx.astype(int)
    else:
        top2_prob = np.full((probs.shape[0],), np.nan)
        top2_topic = np.full((probs.shape[0],), -1, dtype=int)
    return {
        "top1_topic": top1_idx.astype(int),
        "top1_prob": top1_prob,
        "top2_topic": top2_topic,
        "top2_prob": top2_prob,
    }


# =============================================================================
# API embeddings (with cache / concurrency / retry)
# =============================================================================

def _text_hash(text: str, model: str) -> str:
    """SHA-1 of (model, text), first 16 hex chars, used as the cache key.
    The model is mixed in so the cache is invalidated automatically when the model changes."""
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def _load_cache(cache_path: str) -> Dict[str, np.ndarray]:
    """Read the local cache. Missing or unreadable -> empty dict."""
    if not os.path.exists(cache_path):
        return {}
    try:
        data = np.load(cache_path, allow_pickle=False)
        return {k: data[k] for k in data.files}
    except Exception as e:
        logger.warning(f"[cache] 读取失败({e})，按空缓存继续")
        return {}


def _save_cache(cache_path: str, cache: Dict[str, np.ndarray]) -> None:
    """Persist the cache to disk."""
    ensure_dir(os.path.dirname(cache_path) or ".")
    np.savez(cache_path, **cache)


def _call_embedding_api(
    client: OpenAI,
    model: str,
    batch: List[str],
    request_timeout: int,
    max_retries: int,
) -> List[np.ndarray]:
    """
    Request embeddings for one batch, with exponential backoff retries.
    The returned list order strictly matches the batch order.
    """
    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.embeddings.create(
                model=model,
                input=batch,
                timeout=request_timeout,
            )
            # sort by index so the returned order matches the batch
            ordered: List[np.ndarray | None] = [None] * len(batch)
            for item in resp.data:
                idx = int(getattr(item, "index", -1))
                if 0 <= idx < len(batch):
                    ordered[idx] = np.asarray(item.embedding, dtype=np.float32)
            if any(v is None for v in ordered):
                raise RuntimeError(f"API 返回数量不匹配：期望 {len(batch)}，实际 {sum(v is not None for v in ordered)}")
            return [v for v in ordered if v is not None]
        # transient errors that are worth retrying
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_err = e
            wait = min(delay, 30.0)
            logger.warning(f"[api] 第 {attempt}/{max_retries} 次失败 ({type(e).__name__})，等 {wait:.1f}s 后重试")
            time.sleep(wait)
            delay *= 2
        # APIError: retry on 5xx, raise immediately on 4xx
        except APIError as e:
            status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
            if status and 500 <= int(status) < 600:
                last_err = e
                wait = min(delay, 30.0)
                logger.warning(f"[api] 第 {attempt}/{max_retries} 次 {status} 服务端错误，等 {wait:.1f}s 后重试")
                time.sleep(wait)
                delay *= 2
            else:
                raise
    # retries exhausted
    raise RuntimeError(f"API 请求重试 {max_retries} 次仍失败：{last_err}")


def compute_api_embeddings(texts: List[str], cfg: SimpleNamespace, cache_path: str) -> np.ndarray:
    """
    Main entry point: turn a batch of texts into an (N, dim) embedding matrix.
    Flow:
      1. truncate long texts (so one very long text cannot drag down a batch)
      2. look up the cache and pick out the misses
      3. split the misses into batches and request them concurrently with ThreadPoolExecutor
      4. merge the results, update the cache, and return in the original order
    """
    # 1. truncation
    max_chars = int(cfg.api.max_chars_per_text)
    safe_texts = [t[:max_chars] if len(t) > max_chars else t for t in texts]

    # 2. load the cache
    cache: Dict[str, np.ndarray] = {}
    if cfg.api.cache_enabled:
        cache = _load_cache(cache_path)
        logger.info(f"[cache] 命中 {len(cache)} 条历史 embedding")

    # hash every text and find the cache misses
    hashes = [_text_hash(t, cfg.api.model) for t in safe_texts]
    missing_indices = [i for i, h in enumerate(hashes) if h not in cache]
    logger.info(f"[api] 总计 {len(safe_texts)} 条，需请求 {len(missing_indices)} 条，已缓存 {len(safe_texts) - len(missing_indices)} 条")

    # 3. split into batches (misses only)
    if missing_indices:
        client = OpenAI(api_key=cfg.api.api_key, base_url=cfg.api.base_url)
        batch_size = int(cfg.api.batch_size)
        # each batch is (start offset, global index list of this batch, actual texts of this batch)
        batches: List[Tuple[int, List[int], List[str]]] = []
        for s in range(0, len(missing_indices), batch_size):
            idx_chunk = missing_indices[s : s + batch_size]
            texts_chunk = [safe_texts[i] for i in idx_chunk]
            batches.append((s, idx_chunk, texts_chunk))

        # 4. run concurrently
        max_workers = max(1, int(cfg.api.max_concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _call_embedding_api,
                    client,
                    cfg.api.model,
                    texts_chunk,
                    int(cfg.api.request_timeout),
                    int(cfg.api.max_retries),
                ): (batch_no, idx_chunk)
                for batch_no, idx_chunk, texts_chunk in batches
            }
            # wrap as_completed in tqdm to show progress
            for fut in tqdm(as_completed(futures), total=len(futures), desc="API embedding"):
                batch_no, idx_chunk = futures[fut]
                vecs = fut.result()
                for global_i, vec in zip(idx_chunk, vecs):
                    cache[hashes[global_i]] = vec

        # 5. write the cache back
        if cfg.api.cache_enabled:
            _save_cache(cache_path, cache)
            logger.info(f"[cache] 已落盘到 {cache_path}")

    # 6. reassemble the full matrix in the original order
    return np.vstack([cache[h] for h in hashes])


class APIEmbedder(BaseEmbedder):
    """BERTopic-compatible embedding wrapper.

    This wrapper supports both kinds of calls:
      1. document-level API embeddings (for `fit_transform(..., embeddings=...)`)
      2. BERTopic's internal seed-topic / keyword embeddings (via `embed_documents` / `embed_words`)

    By inheriting `BaseEmbedder`, BERTopic uses this wrapper directly in guided topic
    modeling and similar paths, rather than falling back to the sentence-transformers backend.
    """

    def __init__(self, client: OpenAI, model: str, batch_size: int,
                 request_timeout: int, max_retries: int) -> None:
        super().__init__()
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.request_timeout = request_timeout
        self.max_retries = max_retries

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        if not documents:
            return np.empty((0,), dtype=np.float32)
        vecs: List[np.ndarray] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            vecs.extend(_call_embedding_api(
                self.client, self.model, batch,
                self.request_timeout, self.max_retries,
            ))
        return np.vstack(vecs)

    def embed_documents(self, document: List[str], verbose: bool = False) -> np.ndarray:
        return self.embed(document, verbose=verbose)

    def embed_words(self, words: List[str], verbose: bool = False) -> np.ndarray:
        return self.embed(words, verbose=verbose)


# =============================================================================
# Merging into the research macro categories: semantic codebook classification
# =============================================================================

DEFAULT_MACRO_TOPIC_NAMES = [
    "生育身体与医疗技术风险",
    "照护劳动与母职处境",
    "婚育成本与生活机会",
    "婚恋制度与生育选择",
    "性别权力与身体自主",
    "代际家庭伦理与养育关系",
]


def _as_list(value: Any) -> List[Any]:
    """Safely coerce an optional YAML list into a Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Config read that works for both SimpleNamespace and dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _join_codebook_parts(parts: Sequence[str]) -> str:
    return "；".join(str(p).strip() for p in parts if str(p).strip())


def build_macro_topic_defs(cfg: SimpleNamespace) -> List[Dict[str, object]]:
    """Build the final research macro categories from config. Prefers the labels form of the semantic codebook."""
    macro_cfg = getattr(cfg.topic_modeling, "macro_topicing", None)
    if not macro_cfg or not bool(_cfg_get(macro_cfg, "enabled", False)):
        return []

    labels = _as_list(_cfg_get(macro_cfg, "labels", None))
    macro_defs: List[Dict[str, object]] = []
    if labels:
        for i, label in enumerate(labels):
            name = str(_cfg_get(label, "name", DEFAULT_MACRO_TOPIC_NAMES[i] if i < len(DEFAULT_MACRO_TOPIC_NAMES) else f"研究大类{i + 1}"))
            definition = str(_cfg_get(label, "definition", ""))
            include = [str(x).strip() for x in _as_list(_cfg_get(label, "include", [])) if str(x).strip()]
            exclude = [str(x).strip() for x in _as_list(_cfg_get(label, "exclude", [])) if str(x).strip()]
            examples = [str(x).strip() for x in _as_list(_cfg_get(label, "examples", [])) if str(x).strip()]
            priority_keywords = [str(x).strip() for x in _as_list(_cfg_get(label, "priority_keywords", [])) if str(x).strip()]
            codebook_text = (
                f"类别名称：{name}\n"
                f"定义：{definition}\n"
                f"纳入：{_join_codebook_parts(include)}\n"
                f"排除：{_join_codebook_parts(exclude)}\n"
                f"典型表达：{_join_codebook_parts(examples)}"
            )
            macro_defs.append({
                "id": i,
                "name": name,
                "definition": definition,
                "include": include,
                "exclude": exclude,
                "examples": examples,
                "priority_keywords": priority_keywords,
                "codebook_text": codebook_text,
            })
        return macro_defs

    # backwards compatibility: the old names + keywords config still works, but is wrapped into a simple codebook
    names = _as_list(_cfg_get(macro_cfg, "names", None)) or DEFAULT_MACRO_TOPIC_NAMES
    keyword_groups = _as_list(_cfg_get(macro_cfg, "keywords", None))
    if not keyword_groups:
        keyword_groups = _as_list(getattr(cfg.topic_modeling, "seed_topic_list", []))
    min_keyword_len = int(_cfg_get(macro_cfg, "min_keyword_len", 2))
    for i, keywords in enumerate(keyword_groups):
        name = str(names[i]) if i < len(names) else f"研究大类{i + 1}"
        cleaned_keywords = []
        for kw in _as_list(keywords):
            kw = str(kw).strip()
            if len(kw) >= min_keyword_len:
                cleaned_keywords.append(kw)
        cleaned_keywords = sorted(set(cleaned_keywords), key=len, reverse=True)
        macro_defs.append({
            "id": i,
            "name": name,
            "definition": name,
            "include": cleaned_keywords,
            "exclude": [],
            "examples": [],
            "priority_keywords": cleaned_keywords,
            "codebook_text": f"类别名称：{name}\n关键词：{'、'.join(cleaned_keywords)}",
        })
    return macro_defs


def _safe_l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


def _macro_keyword_weight(keyword: str) -> float:
    """Longer phrases define topic boundaries better than generic words, so they get more weight."""
    if len(keyword) >= 5:
        return 2.5
    if len(keyword) >= 4:
        return 2.0
    if len(keyword) >= 3:
        return 1.5
    return 1.0


def classify_macro_topics_semantic(
    texts: Sequence[str],
    doc_embeddings: np.ndarray,
    macro_defs: List[Dict[str, object]],
    api_embedder: "APIEmbedder",
    macro_cfg: SimpleNamespace,
) -> pd.DataFrame:
    """Final research macro-category assignment based on comment-embedding vs codebook-embedding similarity."""
    if not macro_defs:
        return pd.DataFrame(index=range(len(texts)))

    min_similarity = float(_cfg_get(macro_cfg, "min_similarity", 0.28))
    min_margin = float(_cfg_get(macro_cfg, "min_margin", 0.025))
    priority_boost = float(_cfg_get(macro_cfg, "priority_boost", 0.025))
    max_priority_boost = float(_cfg_get(macro_cfg, "max_priority_boost", 0.075))
    unassigned_name = str(_cfg_get(macro_cfg, "unassigned_name", "边界不清/其他"))

    codebook_texts = [str(item["codebook_text"]) for item in macro_defs]
    logger.info("  语义编码本分类: %d 个大类，计算类别 embedding", len(codebook_texts))
    codebook_embeddings = api_embedder.embed_documents(codebook_texts)

    doc_norm = _safe_l2_normalize(np.asarray(doc_embeddings, dtype=np.float32))
    codebook_norm = _safe_l2_normalize(np.asarray(codebook_embeddings, dtype=np.float32))
    raw_scores = doc_norm @ codebook_norm.T
    adjusted_scores = raw_scores.copy()

    matched_keywords: List[List[str]] = [[] for _ in range(len(texts))]
    for row_idx, text in enumerate(texts):
        text = str(text)
        for col_idx, item in enumerate(macro_defs):
            hits = [kw for kw in item.get("priority_keywords", []) if kw and kw in text]
            if not hits:
                continue
            boost_units = sum(_macro_keyword_weight(kw) for kw in hits)
            boost = min(max_priority_boost, priority_boost * boost_units)
            adjusted_scores[row_idx, col_idx] += boost
            if boost > 0:
                matched_keywords[row_idx].extend(hits[:8])

    rebalance_notes = [""] * len(texts)
    rebalance_cfg = _cfg_get(macro_cfg, "first_topic_rebalance", None)
    if rebalance_cfg and bool(_cfg_get(rebalance_cfg, "enabled", False)):
        source_name = str(_cfg_get(rebalance_cfg, "source_topic_name", ""))
        target_names = {str(x) for x in _as_list(_cfg_get(rebalance_cfg, "target_topic_names", [])) if str(x).strip()}
        max_extra_boost = float(_cfg_get(rebalance_cfg, "max_extra_boost", 0.0))
        min_target_similarity = float(_cfg_get(rebalance_cfg, "min_target_similarity", min_similarity))
        source_idx = next((j for j, item in enumerate(macro_defs) if str(item["name"]) == source_name), None)

        if source_idx is not None and max_extra_boost > 0:
            initial_order = np.argsort(-adjusted_scores, axis=1)
            for row_idx, text in enumerate(texts):
                if int(initial_order[row_idx, 0]) != source_idx:
                    continue

                source_score = float(adjusted_scores[row_idx, source_idx])
                candidates: List[Tuple[float, int, List[str], float]] = []
                text = str(text)
                for col_idx, item in enumerate(macro_defs):
                    if col_idx == source_idx:
                        continue
                    if target_names and str(item["name"]) not in target_names:
                        continue
                    hits = [kw for kw in item.get("priority_keywords", []) if kw and kw in text]
                    if not hits:
                        continue

                    target_similarity = float(raw_scores[row_idx, col_idx])
                    if target_similarity < min_target_similarity:
                        continue
                    target_score = float(adjusted_scores[row_idx, col_idx])
                    required_boost = source_score - target_score + min_margin + 1e-6
                    if required_boost <= 0 or required_boost > max_extra_boost:
                        continue
                    keyword_strength = sum(_macro_keyword_weight(kw) for kw in hits)
                    candidates.append((target_score + keyword_strength * 0.001, col_idx, hits, required_boost))

                if not candidates:
                    continue
                _, target_idx, hits, required_boost = max(candidates, key=lambda x: x[0])
                adjusted_scores[row_idx, target_idx] += required_boost
                rebalance_notes[row_idx] = (
                    f"from={source_name};to={macro_defs[target_idx]['name']};"
                    f"hits={'、'.join(dict.fromkeys(hits).keys())}"
                )

    order = np.argsort(-adjusted_scores, axis=1)
    top_idx = order[:, 0]
    second_idx = order[:, 1] if adjusted_scores.shape[1] > 1 else np.full(len(texts), -1, dtype=int)
    top_score = adjusted_scores[np.arange(len(texts)), top_idx]
    top_similarity = raw_scores[np.arange(len(texts)), top_idx]
    second_score = np.where(second_idx >= 0, adjusted_scores[np.arange(len(texts)), second_idx], np.nan)
    margin = top_score - np.nan_to_num(second_score, nan=0.0)

    rows: List[Dict[str, object]] = []
    for i in range(len(texts)):
        best = macro_defs[int(top_idx[i])]
        second_name = macro_defs[int(second_idx[i])]["name"] if int(second_idx[i]) >= 0 else ""
        if float(top_similarity[i]) < min_similarity:
            macro_id = -1
            macro_name = unassigned_name
            reason = "codebook_low_similarity"
        elif float(margin[i]) < min_margin:
            macro_id = -1
            macro_name = unassigned_name
            reason = "codebook_tie"
        else:
            macro_id = int(best["id"])
            macro_name = str(best["name"])
            reason = "semantic_codebook_rebalanced_from_first" if rebalance_notes[i] else "semantic_codebook"
        rows.append({
            "macro_topic": macro_id,
            "macro_topic_name": macro_name,
            "macro_score": float(top_score[i]),
            "macro_similarity": float(top_similarity[i]),
            "macro_score_gap": float(margin[i]),
            "macro_second_topic": int(second_idx[i]),
            "macro_second_name": second_name,
            "macro_matched_keywords": "、".join(dict.fromkeys(matched_keywords[i]).keys()),
            "macro_rebalance_note": rebalance_notes[i],
            "macro_reason": reason,
        })
    return pd.DataFrame(rows)


def build_macro_summary(comments_out: pd.DataFrame) -> pd.DataFrame:
    """Summarize the final research macro categories, so the main analysis can read the first-level topic distribution directly."""
    if "macro_topic" not in comments_out.columns:
        return pd.DataFrame()
    agg = {
        "Count": ("doc_id", "count"),
        "Avg_Macro_Score": ("macro_score", "mean"),
        "Avg_Macro_Similarity": ("macro_similarity", "mean") if "macro_similarity" in comments_out.columns else ("macro_score", "mean"),
        "Avg_Topic_Probability": ("topic_probability", "mean"),
        "Micro_Topic_Count": ("micro_topic", "nunique"),
        "Manual_Review_Count": ("needs_manual_review", "sum"),
    }
    summary = (
        comments_out
        .groupby(["macro_topic", "macro_topic_name"], dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(["macro_topic"], ascending=True)
    )
    total = max(1, int(summary["Count"].sum()))
    summary.insert(2, "Share", summary["Count"] / total)
    return summary

GENERIC_SUBTOPIC_TERMS = {
    "这个", "那个", "就是", "不是", "没有", "觉得", "感觉", "因为", "所以", "如果", "但是", "然后",
    "现在", "真的", "可以", "不能", "不会", "不要", "一个", "一种", "很多", "什么", "时候", "问题",
    "自己", "他们", "我们", "你们", "大家", "应该", "可能", "还是", "其实", "这种", "这样", "为了",
    "有人", "东西", "事情", "评论", "视频", "现实", "社会", "人生",
    "孩子", "小孩", "生孩子", "父母", "爸妈", "家长", "家庭", "结婚", "生育", "女性", "男性",
    "女生", "女人", "男方", "女方", "妈妈", "幸福", "为什么",
}


def _choose_subtopic_count(n_docs: int, sub_cfg: SimpleNamespace) -> int:
    """Pick the number of second-level topics from the first-level sample size, to avoid fragmenting again."""
    min_cluster_size = max(1, int(_cfg_get(sub_cfg, "min_cluster_size", 80)))
    min_subtopics = max(1, int(_cfg_get(sub_cfg, "min_subtopics", 2)))
    max_subtopics = max(min_subtopics, int(_cfg_get(sub_cfg, "max_subtopics", 5)))
    target_docs = max(min_cluster_size, int(_cfg_get(sub_cfg, "target_docs_per_subtopic", 900)))
    if n_docs < min_cluster_size * 2:
        return 1
    k = int(np.ceil(n_docs / target_docs))
    k = max(min_subtopics, k)
    k = min(max_subtopics, k, max(1, n_docs // min_cluster_size))
    return max(1, k)


def _clean_subtopic_token(token: str, min_keyword_len: int) -> str:
    t = str(token).strip()
    if len(t) < min_keyword_len:
        return ""
    if t in GENERIC_SUBTOPIC_TERMS:
        return ""
    if re.fullmatch(r"[0-9]+([.:/-][0-9]+)*", t):
        return ""
    if re.fullmatch(r"[A-Za-z0-9_]+", t):
        return ""
    return t


def _short_text(text: str, limit: int = 120) -> str:
    t = re.sub(r"\s+", " ", str(text)).strip()
    return t if len(t) <= limit else t[:limit] + "..."


def _collect_cluster_tokens(
    texts: Sequence[str],
    tokenizer: Callable[[str], List[str]],
    min_keyword_len: int,
) -> Dict[str, int]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for text in texts:
        for token in tokenizer(str(text)):
            token = _clean_subtopic_token(token, min_keyword_len)
            if token:
                counter[token] += 1
    return dict(counter)


def _rank_subtopic_keywords(
    counts_by_label: Dict[int, Dict[str, int]],
    top_n: int,
) -> Dict[int, List[str]]:
    from collections import Counter

    doc_freq: Counter[str] = Counter()
    for counts in counts_by_label.values():
        for term in counts.keys():
            doc_freq[term] += 1
    n_clusters = max(1, len(counts_by_label))

    ranked: Dict[int, List[str]] = {}
    for label, counts in counts_by_label.items():
        total_terms = max(1, sum(counts.values()))
        scored: List[Tuple[str, float]] = []
        for term, cnt in counts.items():
            idf = np.log((1 + n_clusters) / (1 + doc_freq[term])) + 1.0
            score = (cnt / total_terms) * idf * np.log1p(cnt)
            scored.append((term, float(score)))
        scored.sort(key=lambda x: (-x[1], x[0]))
        ranked[label] = [term for term, _ in scored[:top_n]]
    return ranked


def _merge_small_subtopic_clusters(
    vectors: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int,
) -> np.ndarray:
    """Merge overly small second-level clusters into the nearest large cluster, so second-level topics are easier to interpret in the paper."""
    unique_labels = sorted(int(x) for x in np.unique(labels))
    if len(unique_labels) <= 1:
        return np.zeros(len(labels), dtype=int)

    counts = {label: int(np.sum(labels == label)) for label in unique_labels}
    large_labels = [label for label in unique_labels if counts[label] >= min_cluster_size]
    if not large_labels:
        return np.zeros(len(labels), dtype=int)

    if len(large_labels) < len(unique_labels):
        large_centers: List[np.ndarray] = []
        for label in large_labels:
            center = vectors[labels == label].mean(axis=0, keepdims=True)
            large_centers.append(_safe_l2_normalize(center)[0])
        centers = np.vstack(large_centers)
        small_mask = ~np.isin(labels, large_labels)
        if small_mask.any():
            nearest = np.argmax(vectors[small_mask] @ centers.T, axis=1)
            labels = labels.copy()
            labels[small_mask] = np.asarray([large_labels[int(i)] for i in nearest], dtype=int)

    ordered = sorted((int(x) for x in np.unique(labels)), key=lambda lab: (-int(np.sum(labels == lab)), lab))
    remap = {old: new for new, old in enumerate(ordered)}
    return np.asarray([remap[int(label)] for label in labels], dtype=int)


def build_hierarchical_subtopics(
    comments_df: pd.DataFrame,
    doc_embeddings: np.ndarray,
    tokenizer: Callable[[str], List[str]],
    sub_cfg: SimpleNamespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Re-cluster inside each first-level topic to produce strictly nested second-level topics."""
    if not bool(_cfg_get(sub_cfg, "enabled", False)):
        return comments_df, pd.DataFrame()
    if "macro_topic" not in comments_df.columns or "macro_topic_name" not in comments_df.columns:
        return comments_df, pd.DataFrame()

    out = comments_df.copy()
    total_docs = max(1, len(out))
    min_cluster_size = max(1, int(_cfg_get(sub_cfg, "min_cluster_size", 80)))
    top_n_words = max(3, int(_cfg_get(sub_cfg, "top_n_words", 10)))
    rep_docs = max(1, int(_cfg_get(sub_cfg, "representative_docs", 3)))
    min_keyword_len = max(1, int(_cfg_get(sub_cfg, "min_keyword_len", 2)))
    random_state = int(_cfg_get(sub_cfg, "random_state", 42))

    out["primary_topic"] = out["macro_topic"]
    out["primary_topic_no"] = np.where(out["macro_topic"].fillna(-1).astype(int) >= 0, out["macro_topic"].astype(int) + 1, -1)
    out["primary_topic_name"] = out["macro_topic_name"]
    out["secondary_topic"] = -1
    out["secondary_topic_no"] = ""
    out["secondary_topic_name"] = np.where(out["macro_topic"].fillna(-1).astype(int) >= 0, "", out["macro_topic_name"].astype(str))
    out["secondary_topic_keywords"] = ""
    out["secondary_topic_similarity"] = np.nan

    embeddings = _safe_l2_normalize(np.asarray(doc_embeddings, dtype=np.float32))
    valid_mask = out["macro_topic"].fillna(-1).astype(int).ge(0)
    if "is_noise" in out.columns:
        valid_mask = valid_mask & (~out["is_noise"].fillna(False).astype(bool))

    summary_rows: List[Dict[str, object]] = []
    for macro_id in sorted(out.loc[valid_mask, "macro_topic"].dropna().astype(int).unique().tolist()):
        macro_mask = valid_mask & out["macro_topic"].astype(int).eq(macro_id)
        idx = out.index[macro_mask].to_numpy()
        if len(idx) == 0:
            continue

        positions = out.index.get_indexer(idx)
        group_vectors = embeddings[positions]
        group_texts = out.loc[idx, "text"].astype(str).tolist()
        macro_name = str(out.loc[idx, "macro_topic_name"].iloc[0])
        k = _choose_subtopic_count(len(idx), sub_cfg)
        logger.info("  一级主题[%s] %s: %d 条，拆分为 %d 个二级主题", macro_id + 1, macro_name, len(idx), k)

        if k <= 1:
            labels = np.zeros(len(idx), dtype=int)
        else:
            batch_size = min(4096, max(256, len(idx)))
            cluster_model = MiniBatchKMeans(
                n_clusters=k,
                random_state=random_state,
                batch_size=batch_size,
                n_init=10,
            )
            labels = cluster_model.fit_predict(group_vectors).astype(int)
            labels = _merge_small_subtopic_clusters(group_vectors, labels, min_cluster_size)

        counts_by_label: Dict[int, Dict[str, int]] = {}
        for label in sorted(int(x) for x in np.unique(labels)):
            label_texts = [group_texts[i] for i in np.where(labels == label)[0]]
            counts_by_label[label] = _collect_cluster_tokens(label_texts, tokenizer, min_keyword_len)
        keywords_by_label = _rank_subtopic_keywords(counts_by_label, top_n_words)

        for label in sorted(int(x) for x in np.unique(labels)):
            local_mask = labels == label
            sub_idx = idx[local_mask]
            sub_vectors = group_vectors[local_mask]
            centroid = _safe_l2_normalize(sub_vectors.mean(axis=0, keepdims=True))[0]
            sims = sub_vectors @ centroid
            order = np.argsort(-sims)
            rep_texts = [_short_text(out.loc[sub_idx[int(i)], "text"], 140) for i in order[:rep_docs]]

            keywords = keywords_by_label.get(label, [])
            keyword_text = "、".join(keywords)
            title_terms = "、".join(keywords[:4]) if keywords else f"子主题{label + 1}"
            subtopic_no = f"{macro_id + 1}.{label + 1}"
            subtopic_name = f"{title_terms}"

            out.loc[sub_idx, "secondary_topic"] = int(label)
            out.loc[sub_idx, "secondary_topic_no"] = subtopic_no
            out.loc[sub_idx, "secondary_topic_name"] = subtopic_name
            out.loc[sub_idx, "secondary_topic_keywords"] = keyword_text
            out.loc[sub_idx, "secondary_topic_similarity"] = sims

            row: Dict[str, object] = {
                "primary_topic": int(macro_id),
                "一级主题编号": int(macro_id + 1),
                "一级主题名称": macro_name,
                "secondary_topic": int(label),
                "二级主题编号": subtopic_no,
                "二级主题名称": subtopic_name,
                "评论数": int(local_mask.sum()),
                "占本一级主题比例": float(local_mask.sum() / len(idx)),
                "占全部评论比例": float(local_mask.sum() / total_docs),
                "关键词": keyword_text,
                "代表评论": " || ".join(rep_texts),
                "平均二级主题相似度": float(np.mean(sims)),
                "平均大类相似度": float(out.loc[sub_idx, "macro_similarity"].mean()) if "macro_similarity" in out.columns else np.nan,
                "平均小类概率": float(out.loc[sub_idx, "topic_probability"].mean()) if "topic_probability" in out.columns else np.nan,
                "需人工复核数": int(out.loc[sub_idx, "needs_manual_review"].sum()) if "needs_manual_review" in out.columns else 0,
            }
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["一级主题编号", "评论数"], ascending=[True, False]).reset_index(drop=True)
    return out, summary


# =============================================================================
# Model construction / fitting
# =============================================================================

def build_topic_model(
    cfg: SimpleNamespace,
    tokenizer: Callable[[str], List[str]],
    nr_topics: object,
    effective_min_df: int,
    vectorizer_tokenizer: Callable[[str], List[str]] | None = None,
    api_embedder: "APIEmbedder | None" = None,
) -> BERTopic:
    """Assemble a BERTopic instance.
    Document embeddings are precomputed externally and passed in via fit_transform(embeddings=...),
    so BERTopic will not process documents with embedding_model again; api_embedder is only called
    when MMR needs word-level embeddings for candidate keywords.
    """
    from bertopic import BERTopic
    from bertopic.backend import BaseEmbedder as BERTopicBaseEmbedder
    from bertopic.representation import MaximalMarginalRelevance
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    class BERTopicAPIEmbedder(BERTopicBaseEmbedder):
        """Wrap the lightweight APIEmbedder as a BaseEmbedder that BERTopic accepts."""
        def __init__(self, wrapped: "APIEmbedder") -> None:
            super().__init__()
            self.wrapped = wrapped

        def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
            return self.wrapped.embed(documents, verbose=verbose)

        def embed_documents(self, document: List[str], verbose: bool = False) -> np.ndarray:
            return self.wrapped.embed_documents(document, verbose=verbose)

        def embed_words(self, words: List[str], verbose: bool = False) -> np.ndarray:
            return self.wrapped.embed_words(words, verbose=verbose)

    umap_model = UMAP(
        n_neighbors=cfg.umap.n_neighbors,
        n_components=cfg.umap.n_components,
        min_dist=cfg.umap.min_dist,
        metric="cosine",
        random_state=cfg.umap.random_state,
    )
    cluster_selection_method = getattr(cfg.hdbscan, "cluster_selection_method", "eom")
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        min_samples=cfg.hdbscan.min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        tokenizer=vectorizer_tokenizer or tokenizer,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=effective_min_df,
    )
    representation_model = MaximalMarginalRelevance(diversity=cfg.topic_modeling.mmr_diversity)

    # ⭐ Seed topics (semi-supervised): cfg.topic_modeling.seed_topic_list in the YAML is list[list[str]]
    #    unset / empty list -> pass None and fall back to pure unsupervised
    seed_topic_list = getattr(cfg.topic_modeling, "seed_topic_list", None) or None
    if seed_topic_list:
        logger.info(
            "启用 seed_topic_list: %d 个种子主题，每个词数=%s",
            len(seed_topic_list), [len(s) for s in seed_topic_list],
        )

    return BERTopic(
        embedding_model=BERTopicAPIEmbedder(api_embedder) if api_embedder is not None else None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        seed_topic_list=seed_topic_list,
        calculate_probabilities=True,
        min_topic_size=cfg.hdbscan.min_topic_size,
        nr_topics=nr_topics,
        top_n_words=cfg.topic_modeling.top_n_words,
        verbose=True,
    )


def evaluate_topic_quality(topic_model, topics, top_n_words, target_topic_count, topic_count_penalty):
    """Custom topic quality score. score = 0.45*coverage + 0.35*diversity + 0.20*balance - penalty."""
    topic_info = topic_model.get_topic_info()
    valid = topic_info[topic_info["Topic"] != -1].copy()
    if len(valid) == 0:
        return {"topic_count": 0.0, "noise_ratio": 1.0, "coverage": 0.0,
                "diversity": 0.0, "balance": 0.0, "score": 0.0}
    topic_count = float(len(valid))
    topics_arr = np.asarray(topics)
    noise_ratio = float((topics_arr == -1).mean())
    coverage = 1.0 - noise_ratio
    words: List[str] = []
    for topic_id in valid["Topic"].tolist():
        topic_words = topic_model.get_topic(int(topic_id)) or []
        words.extend([w for w, _ in topic_words[:top_n_words]])
    diversity = float(len(set(words)) / max(1, len(words)))
    counts = valid["Count"].to_numpy(dtype=float)
    probs = counts / counts.sum()
    if len(probs) <= 1:
        balance = 1.0
    else:
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        balance = float(entropy / np.log(len(probs)))
    topic_count_gap = abs(topic_count - float(target_topic_count)) / max(1.0, float(target_topic_count))
    penalty = topic_count_penalty * topic_count_gap
    score = 0.45 * coverage + 0.35 * diversity + 0.20 * balance - penalty
    return {
        "topic_count": topic_count, "noise_ratio": noise_ratio,
        "coverage": coverage, "diversity": diversity, "balance": balance,
        "topic_count_gap": topic_count_gap, "penalty": penalty, "score": score,
    }


def fit_topics_with_fallback(
    cfg, tokenizer, docs, embeddings, nr_topics,
    effective_min_df, vectorizer_tokenizer, fallback_vectorizer_tokenizer,
    api_embedder: "APIEmbedder | None" = None,
):
    """Fault-tolerant fitting: empty vocabulary -> lower min_df + fallback tokenizer; too few samples -> nr_topics=None."""
    topic_model = build_topic_model(
        cfg=cfg, tokenizer=tokenizer, nr_topics=nr_topics,
        effective_min_df=effective_min_df, vectorizer_tokenizer=vectorizer_tokenizer,
        api_embedder=api_embedder,
    )
    try:
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        return topic_model, topics, probs
    except ValueError as e:
        err_msg = str(e)
        if "empty vocabulary" in err_msg or "only contain stop words" in err_msg:
            topic_model = build_topic_model(
                cfg=cfg, tokenizer=tokenizer, nr_topics=nr_topics,
                effective_min_df=1, vectorizer_tokenizer=fallback_vectorizer_tokenizer,
                api_embedder=api_embedder,
            )
            topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
            return topic_model, topics, probs
        if "Found array with 0 sample(s)" in err_msg and nr_topics is not None:
            topic_model = build_topic_model(
                cfg=cfg, tokenizer=tokenizer, nr_topics=None,
                effective_min_df=effective_min_df, vectorizer_tokenizer=vectorizer_tokenizer,
                api_embedder=api_embedder,
            )
            try:
                topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
                return topic_model, topics, probs
            except ValueError as inner_e:
                inner_msg = str(inner_e)
                if "empty vocabulary" in inner_msg or "only contain stop words" in inner_msg:
                    topic_model = build_topic_model(
                        cfg=cfg, tokenizer=tokenizer, nr_topics=None,
                        effective_min_df=1, vectorizer_tokenizer=fallback_vectorizer_tokenizer,
                        api_embedder=api_embedder,
                    )
                    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
                    return topic_model, topics, probs
                raise
        raise


# =============================================================================
# Main pipeline
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 BERTopic + 硅基流动 API 对 B 站评论做主题分类")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help=f"YAML 配置文件路径（默认 {DEFAULT_CONFIG_PATH}）")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    _model_slug = re.sub(r"[^\w.-]", "_", cfg.api.model).strip("_")
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _run_tag = f"{_model_slug}_{_timestamp}"

    ensure_dir(cfg.io.output_dir)
    log_file = os.path.join(cfg.io.output_dir, f"bertopic_run_{_run_tag}.log")
    setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("BERTopic 评论主题分类 (API 版) 启动")
    logger.info(f"配置文件: {args.config}")
    logger.info("=" * 60)

    # ---- 1. locate the input files ----
    with _timer("步骤 1/9  查找输入文件"):
        files: List[str] = []
        if cfg.io.input_file and os.path.exists(cfg.io.input_file):
            files = [cfg.io.input_file]
        else:
            files = sorted(glob.glob(cfg.io.input_glob))
        if not files:
            raise FileNotFoundError(
                f"未找到输入文件。input_file={cfg.io.input_file}, input_glob={cfg.io.input_glob}"
            )
        logger.info(f"  找到 {len(files)} 个文件: {[os.path.basename(f) for f in files]}")

    # ---- 2. dictionaries / vocabularies ----
    with _timer("步骤 2/9  加载词典 / 词表"):
        if cfg.io.user_dict and os.path.exists(cfg.io.user_dict):
            jieba.load_userdict(cfg.io.user_dict)
            logger.info(f"  用户词典: {cfg.io.user_dict}")
        stopwords = load_stopwords(cfg.io.stopwords)
        logger.info(f"  停用词数: {len(stopwords)}")
        irrelevant_keywords = load_keywords(cfg.io.irrelevant_keywords)
        irrelevant_keywords.update(cfg.forced_irrelevant_keywords)
        logger.info(f"  不相关关键词数: {len(irrelevant_keywords)}")
        tokenizer = build_tokenizer(stopwords)
        vectorizer_tokenizer = build_vectorizer_tokenizer(stopwords)
        fallback_vectorizer_tokenizer = build_vectorizer_tokenizer_fallback(stopwords)

    # ---- 3. collect + filter comments ----
    with _timer("步骤 3/9  收集评论"):
        comments_df = collect_comments(
            files=files,
            include_sub_replies=cfg.preprocessing.include_sub_replies,
            min_likes=cfg.preprocessing.min_likes,
        )
        if comments_df.empty:
            raise RuntimeError("没有可用评论，请检查输入文件、字段或过滤参数。")
        logger.info(f"  收集完成: {len(comments_df)} 条（min_likes≥{cfg.preprocessing.min_likes}）")

    with _timer("步骤 4/9  过滤评论"):
        n0 = len(comments_df)

        if (cfg.preprocessing.sample_size and 0 < cfg.preprocessing.sample_size < len(comments_df)):
            comments_df = comments_df.sample(n=cfg.preprocessing.sample_size, random_state=42).reset_index(drop=True)
            comments_df["doc_id"] = comments_df.index
            logger.info(f"  随机采样: {n0} → {len(comments_df)} 条")
            n0 = len(comments_df)

        comments_df["text"] = comments_df["text"].fillna("").astype(str)
        comments_df = comments_df[comments_df["text"].str.len() >= cfg.preprocessing.min_char_len].copy()
        n1 = len(comments_df)
        logger.info(f"  最短字符过滤 (≥{cfg.preprocessing.min_char_len}字): {n0} → {n1} 条，丢弃 {n0 - n1}")

        comments_df["_tokens"] = comments_df["text"].apply(tokenizer)
        comments_df = comments_df[comments_df["_tokens"].apply(len) >= cfg.preprocessing.min_token_count].copy()
        n2 = len(comments_df)
        logger.info(f"  最少 token 过滤 (≥{cfg.preprocessing.min_token_count}个): {n1} → {n2} 条，丢弃 {n1 - n2}")

        before_chatter = len(comments_df)
        comments_df = comments_df[~comments_df["_tokens"].apply(
            lambda ts: should_drop_social_chatter(
                ts, cfg.preprocessing.drop_social_chatter,
                cfg.social_chatter_tokens, cfg.research_signal_tokens,
            )
        )].copy()
        n3 = len(comments_df)
        logger.info(f"  寒暄评论过滤: {before_chatter} → {n3} 条，丢弃 {before_chatter - n3}")

        comments_df["has_research_signal"] = comments_df["text"].apply(
            lambda t: contains_research_signal(t, cfg.research_signal_tokens)
        )
        if cfg.preprocessing.require_research_signal:
            before_signal = len(comments_df)
            comments_df = comments_df[
                comments_df["text"].apply(
                    lambda text: should_keep_research_doc(text, irrelevant_keywords, cfg.research_signal_tokens)
                )
            ].copy()
            n4 = len(comments_df)
            logger.info(f"  研究信号过滤: {before_signal} → {n4} 条，丢弃 {before_signal - n4}")

        comments_df = comments_df.reset_index(drop=True)
        comments_df["doc_id"] = comments_df.index
        if comments_df.empty:
            raise RuntimeError("过滤后没有可用评论。")
        logger.info(f"  过滤完成，最终保留 {len(comments_df)} 条评论")

    # ---- 4. API embeddings ----
    docs = comments_df["text"].astype(str).tolist()
    cache_path = os.path.join(cfg.io.output_dir, f"embedding_cache_{_model_slug}.npz")
    _api_client = OpenAI(api_key=cfg.api.api_key, base_url=cfg.api.base_url)
    api_embedder = APIEmbedder(
        client=_api_client,
        model=cfg.api.model,
        batch_size=int(cfg.api.batch_size),
        request_timeout=int(cfg.api.request_timeout),
        max_retries=int(cfg.api.max_retries),
    )
    logger.info(f"[api] 模型: {cfg.api.model}  base_url: {cfg.api.base_url}")
    with _timer("步骤 5/9  API Embedding"):
        embeddings = compute_api_embeddings(docs, cfg, cache_path)
        logger.info(f"  embedding 矩阵: {embeddings.shape}")

    run_bertopic = bool(_cfg_get(cfg.topic_modeling, "run_bertopic", True))

    # ---- 5. automatic K selection (optional) ----
    effective_min_df = max(1, int(cfg.topic_modeling.min_df))
    selected_k_text = str(parse_nr_topics(cfg.topic_modeling.nr_topics))
    k_report_path = ""
    topic_model = None
    topic_info = pd.DataFrame(columns=["Topic", "Count", "Name", "Representation"])
    if not run_bertopic:
        selected_k_text = "skipped"
        logger.info("步骤 6/9  跳过 BERTopic 自动选 K（topic_modeling.run_bertopic=false）")
    elif cfg.auto_select_k.enabled:
        if cfg.auto_select_k.k_min < 2 or cfg.auto_select_k.k_max < cfg.auto_select_k.k_min:
            raise ValueError("k 范围非法，请确保 k_min>=2 且 k_max>=k_min")
        candidate_ks = list(range(cfg.auto_select_k.k_min, cfg.auto_select_k.k_max + 1, max(1, cfg.auto_select_k.k_step)))
        logger.info(f"步骤 6/9  自动选 K，候选: {candidate_ks}")
        rows: List[Dict[str, float]] = []
        best_score = -1.0
        best_k = candidate_ks[0]
        for k_idx, k in enumerate(candidate_ks, start=1):
            logger.info(f"  [{k_idx}/{len(candidate_ks)}] 试验 k={k} ...")
            with _timer(f"  k={k} 拟合"):
                cm, ct, _ = fit_topics_with_fallback(
                    cfg=cfg, tokenizer=tokenizer, docs=docs, embeddings=embeddings,
                    nr_topics=k, effective_min_df=effective_min_df,
                    vectorizer_tokenizer=vectorizer_tokenizer,
                    fallback_vectorizer_tokenizer=fallback_vectorizer_tokenizer,
                    api_embedder=api_embedder,
                )
            metrics = evaluate_topic_quality(
                cm, ct, cfg.topic_modeling.top_n_words,
                cfg.auto_select_k.target_topic_count, cfg.auto_select_k.topic_count_penalty,
            )
            logger.info(
                f"  k={k}  主题数={int(metrics['topic_count'])}  噪声率={metrics['noise_ratio']:.2%}"
                f"  diversity={metrics['diversity']:.3f}  score={metrics['score']:.4f}"
            )
            row: Dict[str, float] = {"k": float(k)}
            row.update(metrics)
            rows.append(row)
            if metrics["score"] > best_score:
                best_score = metrics["score"]
                best_k = k
        k_report_path = os.path.join(cfg.io.output_dir, f"主题数自动评估报告_{_run_tag}.xlsx")
        save_dataframe_excel(pd.DataFrame(rows), k_report_path)
        selected_k_text = str(best_k)
        logger.info(f"  → 选定最优 k={best_k}（score={best_score:.4f}）")
    else:
        logger.info(f"步骤 6/9  跳过（auto_select_k 未启用），使用 nr_topics={selected_k_text}")

    # ---- 6. official fit ----
    if run_bertopic:
        with _timer(f"步骤 7/9  BERTopic 正式拟合 (nr_topics={selected_k_text})"):
            topic_model, topics, probs = fit_topics_with_fallback(
                cfg=cfg, tokenizer=tokenizer, docs=docs, embeddings=embeddings,
                nr_topics=int(selected_k_text) if selected_k_text.isdigit() else selected_k_text,
                effective_min_df=effective_min_df,
                vectorizer_tokenizer=vectorizer_tokenizer,
                fallback_vectorizer_tokenizer=fallback_vectorizer_tokenizer,
                api_embedder=api_embedder,
            )
    else:
        logger.info("步骤 7/9  跳过 BERTopic 正式拟合，直接进入 7 类语义编码本分类")
        topics = np.full(len(comments_df), -1, dtype=int)
        probs = None

    # ---- 7. labeling / post-processing ----
    topic_count_valid = len([t for t in set(topics) if t != -1])
    if run_bertopic:
        logger.info(f"  拟合完成: 识别出 {topic_count_valid} 个主题，噪声(-1)占比 {(np.array(topics)==-1).mean():.2%}")
    else:
        logger.info("  BERTopic 已跳过：micro_topic/topic_probability 仅保留占位字段，最终分类由 macro_topic 决定")

    logger.info("▶ 步骤 8/9  后处理 / 噪声标记")
    _t_post = time.time()
    comments_df["micro_topic"] = topics
    comments_df["topic"] = comments_df["micro_topic"]
    if probs is not None:
        top2 = extract_top2_from_probs(probs)
        comments_df["top1_topic"] = top2["top1_topic"]
        comments_df["top1_prob"] = top2["top1_prob"]
        comments_df["top2_topic"] = top2["top2_topic"]
        comments_df["top2_prob"] = top2["top2_prob"]
        comments_df["topic_probability"] = comments_df["top1_prob"]
        comments_df["prob_gap"] = comments_df["top1_prob"].fillna(0) - comments_df["top2_prob"].fillna(0)
    else:
        comments_df["top1_topic"] = comments_df["micro_topic"]
        comments_df["top1_prob"] = np.nan
        comments_df["top2_topic"] = -1
        comments_df["top2_prob"] = np.nan
        comments_df["topic_probability"] = np.nan
        comments_df["prob_gap"] = np.nan

    macro_defs = build_macro_topic_defs(cfg)
    comments_df["model_is_outlier"] = comments_df["micro_topic"].eq(-1)
    comments_df["is_noise"] = False if macro_defs else comments_df["model_is_outlier"]
    comments_df["noise_reason"] = np.where(comments_df["is_noise"], "hdbscan_outlier", "")
    comments_df["confidence_band"] = ""
    comments_df["needs_manual_review"] = False
    comments_df["review_reason"] = ""

    if probs is not None:
        comments_df["confidence_band"] = comments_df["topic_probability"].fillna(0).apply(classify_confidence)
    else:
        comments_df["confidence_band"] = "low"

    if probs is not None and cfg.topic_modeling.min_topic_prob > 0:
        low_prob_mask = comments_df["topic_probability"].fillna(0) < cfg.topic_modeling.min_topic_prob
        low_prob_mask = low_prob_mask & (~comments_df["has_research_signal"])
        comments_df.loc[low_prob_mask, "is_noise"] = True
        comments_df.loc[low_prob_mask, "noise_reason"] = "low_probability"

    if probs is not None:
        manual_review_mask = comments_df["prob_gap"].fillna(1) < cfg.topic_modeling.review_prob_gap
        comments_df.loc[manual_review_mask, "needs_manual_review"] = True
        comments_df.loc[manual_review_mask, "review_reason"] = "small_prob_gap"

    if irrelevant_keywords:
        pattern = "|".join(re.escape(k) for k in sorted(irrelevant_keywords, key=len, reverse=True))
        irrelevant_mask = comments_df["text"].str.contains(pattern, case=False, regex=True, na=False)
        irrelevant_mask = irrelevant_mask & (~comments_df["has_research_signal"])
        newly_irrelevant = irrelevant_mask & (~comments_df["is_noise"])
        comments_df.loc[newly_irrelevant, "noise_reason"] = "irrelevant_keyword"
        comments_df.loc[irrelevant_mask, "is_noise"] = True

    if macro_defs:
        macro_cfg = getattr(cfg.topic_modeling, "macro_topicing", SimpleNamespace())
        macro_df = classify_macro_topics_semantic(
            texts=comments_df["text"].astype(str).tolist(),
            doc_embeddings=embeddings,
            macro_defs=macro_defs,
            api_embedder=api_embedder,
            macro_cfg=macro_cfg,
        )
        macro_df.index = comments_df.index
        comments_df = pd.concat([comments_df, macro_df], axis=1)
        macro_noise_mask = comments_df["macro_topic"].eq(-1)
        newly_macro_noise = macro_noise_mask & (~comments_df["is_noise"])
        comments_df.loc[newly_macro_noise, "noise_reason"] = comments_df.loc[newly_macro_noise, "macro_reason"]
        comments_df.loc[macro_noise_mask, "is_noise"] = True
        macro_review_mask = comments_df["macro_reason"].isin(["codebook_tie", "codebook_low_similarity"])
        comments_df.loc[macro_review_mask, "needs_manual_review"] = True
        comments_df.loc[macro_review_mask, "review_reason"] = comments_df.loc[macro_review_mask, "macro_reason"]
        comments_df["topic"] = comments_df["macro_topic"]
        comments_df["topic_name"] = comments_df["macro_topic_name"]
        macro_counts = comments_df["macro_topic_name"].value_counts(dropna=False).to_dict()
        logger.info(f"  最终研究大类归并: {macro_counts}")
    else:
        comments_df.loc[comments_df["is_noise"], "topic"] = -1

    subtopic_summary = pd.DataFrame()
    if macro_defs:
        sub_cfg = getattr(cfg.topic_modeling, "sub_topicing", SimpleNamespace(enabled=False))
        comments_df, subtopic_summary = build_hierarchical_subtopics(
            comments_df=comments_df,
            doc_embeddings=embeddings,
            tokenizer=tokenizer,
            sub_cfg=sub_cfg,
        )

    logger.info(f"✓ 步骤 8/9  后处理 / 噪声标记 完成，耗时 {time.time() - _t_post:.1f}s")

    # ---- 8. outputs ----
    logger.info("▶ 步骤 9/9  写出结果文件")
    _t_output = time.time()
    comments_path = os.path.join(cfg.io.output_dir, f"评论主题分类结果_{_run_tag}.xlsx")
    comments_out = comments_df.drop(columns=["_tokens"], errors="ignore")
    save_dataframe_excel(comments_out, comments_path)

    analysis_comments_path = os.path.join(cfg.io.output_dir, f"评论主题分类结果_主分析_{_run_tag}.xlsx")
    save_dataframe_excel(comments_out[~comments_out["is_noise"]], analysis_comments_path)

    macro_summary_path = ""
    macro_summary = build_macro_summary(comments_out[~comments_out["is_noise"]])
    if not macro_summary.empty:
        macro_summary_path = os.path.join(cfg.io.output_dir, f"主题大类概览_{_run_tag}.xlsx")
        save_dataframe_excel(macro_summary, macro_summary_path)

    subtopic_summary_path = ""
    if not subtopic_summary.empty:
        subtopic_summary_path = os.path.join(cfg.io.output_dir, f"一级二级主题表_{_run_tag}.xlsx")
        save_dataframe_excel(subtopic_summary, subtopic_summary_path)

    if run_bertopic and topic_model is not None:
        topic_info = topic_model.get_topic_info()
        if not topic_info.empty and "Representation" in topic_info.columns:
            topic_info = topic_info.copy()
            topic_info["Name"] = topic_info["Name"].astype(str).apply(lambda n: clean_topic_keyword(n, stopwords))
    else:
        topic_info = pd.DataFrame(columns=["Topic", "Count", "Name", "Representation"])
        logger.info("  跳过 BERTopic，主题概览输出为空表；请查看主题大类概览和一级二级主题表")
    topic_info_path = os.path.join(cfg.io.output_dir, f"主题概览_{_run_tag}.xlsx")
    save_dataframe_excel(topic_info, topic_info_path)

    topic_rows: List[Dict[str, object]] = []
    valid_topics = [t for t in topic_info["Topic"].tolist() if int(t) != -1]
    logger.info(f"  有效主题数（非噪声）: {len(valid_topics)}")
    if run_bertopic and topic_model is not None:
        for topic_id in valid_topics:
            words = topic_model.get_topic(int(topic_id)) or []
            if not words:
                logger.warning(f"  主题 {topic_id}: get_topic() 返回空，跳过")
                continue
            for rank, (word, score) in enumerate(words, start=1):
                cw = clean_topic_keyword(str(word), stopwords)
                if not cw:
                    continue
                topic_rows.append({"topic": int(topic_id), "rank": rank, "word": cw, "score": float(score)})
    if run_bertopic and not topic_rows:
        logger.warning("  主题关键词明细为空：若有效主题数>0，请检查 stopwords 是否过滤了所有关键词")
    topic_words_path = os.path.join(cfg.io.output_dir, f"主题关键词明细_{_run_tag}.xlsx")
    save_dataframe_excel(pd.DataFrame(topic_rows, columns=["topic", "rank", "word", "score"]), topic_words_path)

    if cfg.output.save_model and run_bertopic and topic_model is not None:
        logger.info("  保存模型中 ...")
        model_save_path = os.path.join(cfg.io.output_dir, f"bertopic_model_{_run_tag}")
        topic_model.save(model_save_path, serialization="safetensors", save_ctfidf=True)
        logger.info(f"  模型已保存至 {model_save_path}")
    elif cfg.output.save_model and not run_bertopic:
        logger.info("  跳过模型保存：topic_modeling.run_bertopic=false")

    logger.info(f"✓ 步骤 9/9  写出结果文件 完成，耗时 {time.time() - _t_output:.1f}s")

    # ---- 9. summary ----
    noise_count = int((comments_df["is_noise"]).sum())
    valid_count = len(comments_df) - noise_count
    logger.info("")
    logger.info("=" * 60)
    logger.info("BERTopic 全流程完成")
    logger.info("=" * 60)
    logger.info(f"  配置文件:       {args.config}")
    logger.info(f"  文档总数:       {len(comments_df)}")
    logger.info(f"  有效文档:       {valid_count}  噪声: {noise_count} ({noise_count/len(comments_df):.1%})")
    logger.info(f"  嵌入模型:       {cfg.api.model}  维度: {embeddings.shape[1]}")
    logger.info(f"  最终 nr_topics: {selected_k_text}  实际主题数(含噪声): {topic_info.shape[0]}")
    logger.info(f"  评论级结果:     {comments_path}")
    logger.info(f"  主分析结果:     {analysis_comments_path}")
    logger.info(f"  主题概览:       {topic_info_path}")
    if macro_summary_path:
        logger.info(f"  主题大类概览:   {macro_summary_path}")
    if subtopic_summary_path:
        logger.info(f"  一级二级主题表: {subtopic_summary_path}")
    logger.info(f"  关键词明细:     {topic_words_path}")
    if k_report_path:
        logger.info(f"  主题数评估报告: {k_report_path}")
    logger.info(f"  日志文件:       {log_file}")


if __name__ == "__main__":
    main()

