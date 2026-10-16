from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# Category labels live in a shared module so every stage uses the same vocabulary.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_categories import (  # noqa: E402
    CATEGORY_COLUMN,
    MISSING_VIDEO_METADATA,
    standardize_category_column,
)


# --- Path configuration ---------------------------------------------------------------
# Priority: CLI args > env vars BFD_DATA_ROOT / BFD_OUTPUT_ROOT > defaults "data" / "output"
# Input files live under <BFD_DATA_ROOT>/derived/ (filenames below),
# and can also be specified directly via --sentiment-file / --topic-file / --video-file.
_DATA_ROOT = Path(os.environ.get("BFD_DATA_ROOT", "data"))
_OUTPUT_ROOT = Path(os.environ.get("BFD_OUTPUT_ROOT", "output"))

RESULT_DIR = _DATA_ROOT / "derived"
OUTPUT_DIR = _OUTPUT_ROOT / "sentiment_topic_video_analysis"
# Note: OUTPUT_DIR is not created at module level, to avoid side effects on import / --help;
# the directory is created by main() after the final output path is determined.

# Sentiment analysis result table (contains sentiment_label / sentiment_score columns)
SENTIMENT_FILE = RESULT_DIR / "comments_sentiment.xlsx"
# Topic classification result table (boundary-noise fine-grained coding sheet, contains macro_topic_name etc.)
TOPIC_FILE = RESULT_DIR / "comments_topic_classification.xlsx"
# Video info table (contains the hand-coded analytic category)
VIDEO_FILE = RESULT_DIR / "video_info.xlsx"

LABEL_ORDER = ["积极", "中立", "消极"]
LABEL_MAP = {
    "positive": "积极",
    "neutral": "中立",
    "negative": "消极",
    "积极": "积极",
    "中立": "中立",
    "中性": "中立",
    "消极": "消极",
    "负面": "消极",
    "正面": "积极",
}


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_label(value: Any) -> str:
    if pd.isna(value):
        return "未判定"
    return LABEL_MAP.get(str(value).strip().lower(), LABEL_MAP.get(str(value).strip(), str(value).strip()))


def safe_num(series: pd.Series, fill: float = 0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill)


def safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def pct(value: float) -> float:
    if value is None or pd.isna(value) or math.isinf(value):
        return 0.0
    return float(value)


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    comments = pd.read_excel(SENTIMENT_FILE)
    topics = pd.read_excel(TOPIC_FILE, sheet_name="评论分类_更新")
    # Accept `analytic_category` or the legacy `学术分类`, and normalise labels to English.
    videos = standardize_category_column(pd.read_excel(VIDEO_FILE))

    comments["Comment_ID"] = clean_id(comments["Comment_ID"])
    comments["Video_ID"] = clean_id(comments["Video_ID"])
    comments["sentiment_label_norm"] = comments["sentiment_label"].map(clean_label)
    comments["sentiment_score_num"] = safe_num(comments["sentiment_score"])
    comments["Comment_Likes_num"] = safe_num(comments["Comment_Likes"])
    comments["Comment_Time_dt"] = pd.to_datetime(comments["Comment_Time"], errors="coerce")
    comments["Comment_Year"] = comments["Comment_Time_dt"].dt.year

    topics["root_comment_id"] = clean_id(topics["root_comment_id"])
    topics["video_id"] = clean_id(topics["video_id"])
    topics["完全不相关标记_num"] = safe_num(topics.get("完全不相关标记", pd.Series(0, index=topics.index)))
    topics["primary_topic_no_display"] = topics["primary_topic_no"].map(lambda x: "" if pd.isna(x) else str(x).rstrip(".0"))
    topics["primary_topic_name_display"] = topics["primary_topic_name"].fillna("未分类").astype(str)
    topics["secondary_topic_no_display"] = topics["secondary_topic_no"].map(lambda x: "" if pd.isna(x) else str(x).rstrip("0").rstrip("."))
    topics["secondary_topic_name_display"] = topics["规范二级主题名称"].fillna(topics.get("secondary_topic_name")).fillna("未分类").astype(str)
    topics["primary_topic_label"] = topics["primary_topic_no_display"] + " " + topics["primary_topic_name_display"]
    topics["secondary_topic_label"] = topics["secondary_topic_no_display"] + " " + topics["secondary_topic_name_display"]

    videos["Video_ID"] = clean_id(videos["Video_ID"])
    for col in ["Views", "Comments", "Likes", "Collections", "Coins", "Shares", "Comment Ratio", "Bullet Ratio", "Approval Ratio", "Post_Year", "Post_Month"]:
        if col in videos.columns:
            videos[col + "_num"] = safe_num(videos[col])

    diagnostics = {
        "comments_rows": int(len(comments)),
        "topics_rows": int(len(topics)),
        "topics_relevant_rows": int((topics["完全不相关标记_num"] != 1).sum()),
        "videos_rows": int(len(videos)),
        "comments_duplicate_comment_id": int(comments["Comment_ID"].duplicated().sum()),
        "topics_duplicate_root_comment_id": int(topics["root_comment_id"].duplicated().sum()),
        "videos_duplicate_video_id": int(videos["Video_ID"].duplicated().sum()),
        "sentiment_status_counts": comments.get("sentiment_status", pd.Series(dtype=str)).fillna("空").value_counts().to_dict(),
    }
    return comments, topics, videos, diagnostics


def prepare_merged(comments: pd.DataFrame, topics: pd.DataFrame, videos: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    comments_unique = comments.drop_duplicates("Comment_ID", keep="first").copy()
    topics_relevant = topics[topics["完全不相关标记_num"] != 1].drop_duplicates("root_comment_id", keep="first").copy()

    merged = comments_unique.merge(
        topics_relevant,
        left_on="Comment_ID",
        right_on="root_comment_id",
        how="inner",
        suffixes=("_sent", "_topic"),
    )

    video_cols = [
        "Video_ID",
        "UP_name",
        "Views_num",
        "Comments_num",
        "Likes_num",
        "Collections_num",
        "Coins_num",
        "Shares_num",
        "Post_Year_num",
        "Post_Month_num",
        "主题名称",
        "四分区",
        CATEGORY_COLUMN,
        "Comment Ratio_num",
        "Bullet Ratio_num",
        "Approval Ratio_num",
    ]
    video_cols = [c for c in video_cols if c in videos.columns]
    videos_unique = videos.drop_duplicates("Video_ID", keep="first")[video_cols].copy()
    merged = merged.merge(videos_unique, on="Video_ID", how="left")

    merged["analysis_text"] = merged["Comment_Content"].where(merged["Comment_Content"].notna(), merged.get("text", ""))
    merged["analysis_text"] = merged["analysis_text"].fillna("").astype(str)
    merged["academic_class"] = merged[CATEGORY_COLUMN].fillna(MISSING_VIDEO_METADATA).astype(str)
    merged["video_theme"] = merged["主题名称"].fillna("视频元数据缺失").astype(str)
    merged["video_partition"] = merged["四分区"].fillna("视频元数据缺失").astype(str)
    merged["topic_source"] = merged.get("分类来源", pd.Series("未知", index=merged.index)).fillna("未知").astype(str)
    merged["needs_manual_review_text"] = merged.get("needs_manual_review", pd.Series(False, index=merged.index)).fillna(False).astype(str)

    coverage = {
        "analysis_rows": int(len(merged)),
        "analysis_unique_videos": int(merged["Video_ID"].nunique()),
        "comments_matched_to_relevant_topics": int(merged["Comment_ID"].nunique()),
        "comments_without_relevant_topic": int(comments_unique["Comment_ID"].nunique() - merged["Comment_ID"].nunique()),
        "topic_relevant_without_sentiment": int(topics_relevant["root_comment_id"].nunique() - merged["Comment_ID"].nunique()),
        "rows_with_video_metadata": int(merged["academic_class"].ne("视频元数据缺失").sum()),
        "rows_without_video_metadata": int(merged["academic_class"].eq("视频元数据缺失").sum()),
        "unique_videos_with_metadata": int(merged.loc[merged["academic_class"].ne("视频元数据缺失"), "Video_ID"].nunique()),
        "unique_videos_without_metadata": int(merged.loc[merged["academic_class"].eq("视频元数据缺失"), "Video_ID"].nunique()),
    }
    return merged, coverage


def sentiment_mix(df: pd.DataFrame, label_col: str = "sentiment_label_norm", weight_col: str = "Comment_Likes_num") -> dict[str, Any]:
    n = len(df)
    likes_total = float(df[weight_col].sum()) if weight_col in df.columns else 0
    score_mean = float(df["sentiment_score_num"].mean()) if n else 0
    score_median = float(df["sentiment_score_num"].median()) if n else 0
    weighted_score = 0.0
    if likes_total > 0:
        weighted_score = float(np.average(df["sentiment_score_num"], weights=df[weight_col]))
    out: dict[str, Any] = {
        "评论数": int(n),
        "视频数": int(df["Video_ID"].nunique()) if "Video_ID" in df.columns else 0,
        "评论获赞数": likes_total,
        "平均情绪分": score_mean,
        "中位情绪分": score_median,
        "获赞加权情绪分": weighted_score,
    }
    for label in LABEL_ORDER:
        mask = df[label_col].eq(label)
        count = int(mask.sum())
        likes = float(df.loc[mask, weight_col].sum()) if weight_col in df.columns else 0
        out[f"{label}数"] = count
        out[f"{label}占比"] = count / n if n else 0
        out[f"{label}获赞数"] = likes
        out[f"{label}获赞占比"] = likes / likes_total if likes_total else 0
    out["净积极占比"] = out["积极占比"] - out["消极占比"]
    out["情绪极化占比"] = out["积极占比"] + out["消极占比"]
    return out


def aggregate(df: pd.DataFrame, group_cols: list[str], min_count: int = 1, sort_by: str = "评论数") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(sentiment_mix(sub))
        row["平均点赞"] = float(sub["Comment_Likes_num"].mean()) if len(sub) else 0
        row["人工复核占比"] = float(sub["needs_manual_review_text"].str.lower().isin(["true", "1", "是"]).mean()) if "needs_manual_review_text" in sub.columns else 0
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result[result["评论数"] >= min_count].copy()
    if sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=False)
    return result.reset_index(drop=True)


def add_lift_columns(table: pd.DataFrame, overall: dict[str, Any]) -> pd.DataFrame:
    if table.empty:
        return table
    result = table.copy()
    base_neg = overall["消极占比"]
    base_pos = overall["积极占比"]
    result["消极占比差_百分点"] = (result["消极占比"] - base_neg) * 100
    result["积极占比差_百分点"] = (result["积极占比"] - base_pos) * 100
    result["消极提升倍数"] = result["消极占比"] / base_neg if base_neg else 0
    return result


def crosstab_long(df: pd.DataFrame, row_col: str, col_col: str = "sentiment_label_norm") -> pd.DataFrame:
    ct = pd.crosstab(df[row_col], df[col_col])
    for label in LABEL_ORDER:
        if label not in ct.columns:
            ct[label] = 0
    ct = ct[LABEL_ORDER]
    share = ct.div(ct.sum(axis=1), axis=0).add_suffix("占比")
    out = pd.concat([ct.add_suffix("数"), share], axis=1).reset_index()
    out["评论数"] = ct.sum(axis=1).values
    return out.sort_values("评论数", ascending=False).reset_index(drop=True)


def chi_square_test(df: pd.DataFrame, var: str) -> dict[str, Any]:
    contingency = pd.crosstab(df[var], df["sentiment_label_norm"])
    contingency = contingency.loc[contingency.sum(axis=1) > 0, contingency.sum(axis=0) > 0]
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"变量": var, "样本数": int(contingency.values.sum()), "卡方": None, "p值": None, "Cramers_V": None}
    observed = contingency.to_numpy(dtype=float)
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sum @ col_sum / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_manual = np.nansum((observed - expected) ** 2 / expected)
    try:
        from scipy.stats import chi2_contingency

        chi2, p, _, _ = chi2_contingency(contingency)
    except Exception:
        chi2, p = chi2_manual, np.nan
    n = contingency.values.sum()
    cramers_v = math.sqrt(float(chi2) / (n * (min(contingency.shape) - 1))) if n and not pd.isna(chi2) else np.nan
    return {
        "变量": var,
        "样本数": int(n),
        "类别数": int(contingency.shape[0]),
        "卡方": float(chi2) if not pd.isna(chi2) else None,
        "p值": float(p) if not pd.isna(p) else None,
        "Cramers_V": float(cramers_v) if not pd.isna(cramers_v) else None,
    }


def keyword_analysis(df: pd.DataFrame, overall: dict[str, Any]) -> pd.DataFrame:
    keyword_groups = {
        "三孩/二孩/多孩政策": r"三孩|三胎|二孩|二胎|多孩|一孩|计划生育|独生子女",
        "育儿补贴/福利托育": r"育儿补贴|补贴|福利|托育|幼儿园|产假|育儿假|医保|社保|生育津贴",
        "低生育率/人口结构": r"低生育|生育率|出生率|出生人口|人口|少子化|老龄化|老龄|养老",
        "经济成本/住房收入": r"彩礼|嫁妆|房贷|房子|买房|工资|收入|钱|成本|养不起|教育费|学区|奶粉",
        "身体风险/医疗经验": r"怀孕|孕期|生产|生孩子|剖腹产|顺产|产后|月子|宫缩|疼|医院|医生|产检",
        "母职照护/家务分工": r"妈妈|母亲|宝妈|带娃|育儿|照顾|家务|保姆|全职妈妈|丧偶式",
        "不婚不育/自由生活": r"不婚|不育|丁克|单身|自由|不结婚|不生|恐婚|恐育|绝育",
        "性别权力/权益": r"女性|女权|男权|性别|平等|权益|压迫|父权|冠姓权|重男轻女",
    }
    rows = []
    text = df["analysis_text"].fillna("").astype(str)
    for name, pattern in keyword_groups.items():
        mask = text.str.contains(pattern, regex=True, na=False)
        sub = df[mask]
        row = {"关键词组": name, "匹配规则": pattern}
        row.update(sentiment_mix(sub))
        rows.append(row)
    table = pd.DataFrame(rows)
    return add_lift_columns(table, overall).sort_values("评论数", ascending=False).reset_index(drop=True)


def video_level(df: pd.DataFrame, overall: dict[str, Any]) -> pd.DataFrame:
    group_cols = ["Video_ID", "Title", "UP_name", "video_theme", "video_partition", "academic_class", "Post_Year_num"]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(sentiment_mix(sub))
        for source_col, out_col in [
            ("Views_num", "视频播放量"),
            ("Comments_num", "视频评论量"),
            ("Likes_num", "视频点赞量"),
            ("Comment Ratio_num", "视频评论率"),
            ("Approval Ratio_num", "视频认可率"),
        ]:
            row[out_col] = float(sub[source_col].iloc[0]) if source_col in sub.columns and pd.notna(sub[source_col].iloc[0]) else None
        rows.append(row)
    table = pd.DataFrame(rows)
    table = add_lift_columns(table, overall)
    return table.sort_values(["评论数", "消极占比"], ascending=[False, False]).reset_index(drop=True)


def top_comment_samples(df: pd.DataFrame, label: str, limit: int = 50) -> pd.DataFrame:
    cols = [
        "Comment_ID",
        "Video_ID",
        "Title",
        "primary_topic_label",
        "secondary_topic_label",
        "academic_class",
        "sentiment_label_norm",
        "sentiment_score_num",
        "Comment_Likes_num",
        "analysis_text",
    ]
    sub = df[df["sentiment_label_norm"].eq(label)].copy()
    sub = sub.sort_values(["Comment_Likes_num", "sentiment_score_num"], ascending=[False, True if label == "消极" else False])
    return sub[cols].head(limit).reset_index(drop=True)


def correlation_table(video_table: pd.DataFrame) -> pd.DataFrame:
    numeric_pairs = [
        ("消极占比", "视频播放量"),
        ("消极占比", "视频评论率"),
        ("消极占比", "视频认可率"),
        ("积极占比", "视频播放量"),
        ("平均情绪分", "视频评论率"),
        ("获赞加权情绪分", "视频认可率"),
    ]
    rows = []
    for a, b in numeric_pairs:
        valid = video_table[[a, b]].dropna()
        if len(valid) >= 3:
            spearman = valid[a].rank(method="average").corr(valid[b].rank(method="average"), method="pearson")
            rows.append(
                {
                    "变量1": a,
                    "变量2": b,
                    "视频数": int(len(valid)),
                    "Pearson相关": float(valid[a].corr(valid[b], method="pearson")),
                    "Spearman相关": float(spearman),
                }
            )
    return pd.DataFrame(rows)


def as_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, Any]]:
    if max_rows is not None:
        df = df.head(max_rows)
    return json_sanitize(df.to_dict(orient="records"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="主题 × 情感 × 视频类别交叉分析（论文 RQ3 / RQ4）"
    )
    parser.add_argument("--sentiment-file", type=Path, default=None, help="情感分析结果表 xlsx")
    parser.add_argument("--topic-file", type=Path, default=None, help="主题分类结果表 xlsx（sheet: 评论分类_更新）")
    parser.add_argument("--video-file", type=Path, default=None, help="视频信息表 xlsx（含「学术分类」列）")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录，默认 output/sentiment_topic_video_analysis")
    return parser.parse_args()


def main() -> None:
    global SENTIMENT_FILE, TOPIC_FILE, VIDEO_FILE, OUTPUT_DIR
    args = parse_args()
    if args.sentiment_file:
        SENTIMENT_FILE = args.sentiment_file
    if args.topic_file:
        TOPIC_FILE = args.topic_file
    if args.video_file:
        VIDEO_FILE = args.video_file
    if args.out_dir:
        OUTPUT_DIR = args.out_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, path in [("情感分析结果表", SENTIMENT_FILE), ("主题分类结果表", TOPIC_FILE), ("视频信息表", VIDEO_FILE)]:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{label}不存在：{path}\n"
                f"请用对应 CLI 参数指定，或设置环境变量 BFD_DATA_ROOT（当前 {_DATA_ROOT}），"
                f"并将文件放到 <BFD_DATA_ROOT>/derived/ 下，命名见 data/README.md"
            )

    comments, topics, videos, diagnostics = load_data()
    merged, coverage = prepare_merged(comments, topics, videos)
    overall = sentiment_mix(merged)

    primary = add_lift_columns(aggregate(merged, ["primary_topic_no_display", "primary_topic_name_display", "primary_topic_label"], min_count=30), overall)
    secondary = add_lift_columns(aggregate(merged, ["primary_topic_label", "secondary_topic_no_display", "secondary_topic_name_display", "secondary_topic_label"], min_count=30), overall)
    academic = add_lift_columns(aggregate(merged, ["academic_class"], min_count=30), overall)
    partition = add_lift_columns(aggregate(merged, ["video_partition"], min_count=30), overall)
    video_theme = add_lift_columns(aggregate(merged, ["video_theme"], min_count=30), overall)
    topic_source = add_lift_columns(aggregate(merged, ["topic_source"], min_count=1), overall)
    comment_year = add_lift_columns(aggregate(merged[merged["Comment_Year"].notna()], ["Comment_Year"], min_count=20), overall)
    post_year = add_lift_columns(aggregate(merged[merged["Post_Year_num"].notna()], ["Post_Year_num"], min_count=20), overall)
    keywords = keyword_analysis(merged, overall)

    interaction = add_lift_columns(
        aggregate(merged, ["primary_topic_label", "academic_class"], min_count=80, sort_by="消极占比"),
        overall,
    ).sort_values(["消极占比", "评论数"], ascending=[False, False]).reset_index(drop=True)

    secondary_academic = add_lift_columns(
        aggregate(merged, ["secondary_topic_label", "academic_class"], min_count=80, sort_by="消极占比"),
        overall,
    ).sort_values(["消极占比", "评论数"], ascending=[False, False]).reset_index(drop=True)

    video_table = video_level(merged, overall)
    correlations = correlation_table(video_table[video_table["评论数"] >= 20].copy())

    tests = pd.DataFrame(
        [
            chi_square_test(merged, "primary_topic_label"),
            chi_square_test(merged, "secondary_topic_label"),
            chi_square_test(merged, "academic_class"),
            chi_square_test(merged, "video_partition"),
            chi_square_test(merged, "topic_source"),
        ]
    )

    high_neg_secondary = secondary[secondary["评论数"] >= 300].sort_values(["消极占比", "评论数"], ascending=[False, False]).head(15)
    high_pos_secondary = secondary[secondary["评论数"] >= 300].sort_values(["积极占比", "评论数"], ascending=[False, False]).head(15)
    high_neg_academic = academic.sort_values(["消极占比", "评论数"], ascending=[False, False]).head(15)
    high_pos_academic = academic.sort_values(["积极占比", "评论数"], ascending=[False, False]).head(15)
    high_neg_videos = video_table[video_table["评论数"] >= 30].sort_values(["消极占比", "评论数"], ascending=[False, False]).head(30)
    high_pos_videos = video_table[video_table["评论数"] >= 30].sort_values(["积极占比", "评论数"], ascending=[False, False]).head(30)

    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "sentiment_file": str(SENTIMENT_FILE),
            "topic_file": str(TOPIC_FILE),
            "video_file": str(VIDEO_FILE),
            "label_order": LABEL_ORDER,
            "notes": [
                "分析样本为评论情绪表与评论分类_更新表的交集，并剔除主题分类本标记为完全不相关的样本。",
                "视频学术分类没有独立情绪标签；本分析统计该学术分类下相关评论的情绪分布。",
                "关键词组为非互斥匹配，同一评论可进入多个关键词组。",
            ],
        },
        "diagnostics": {**diagnostics, **coverage},
        "overall": overall,
        "tables": {
            "primary": as_records(primary),
            "secondary": as_records(secondary),
            "academic": as_records(academic),
            "partition": as_records(partition),
            "video_theme": as_records(video_theme),
            "topic_source": as_records(topic_source),
            "comment_year": as_records(comment_year),
            "post_year": as_records(post_year),
            "keywords": as_records(keywords),
            "interaction_primary_academic": as_records(interaction, 300),
            "interaction_secondary_academic": as_records(secondary_academic, 300),
            "videos": as_records(video_table, 500),
            "high_neg_secondary": as_records(high_neg_secondary),
            "high_pos_secondary": as_records(high_pos_secondary),
            "high_neg_academic": as_records(high_neg_academic),
            "high_pos_academic": as_records(high_pos_academic),
            "high_neg_videos": as_records(high_neg_videos),
            "high_pos_videos": as_records(high_pos_videos),
            "tests": as_records(tests),
            "correlations": as_records(correlations),
            "top_negative_comments": as_records(top_comment_samples(merged, "消极", 50)),
            "top_positive_comments": as_records(top_comment_samples(merged, "积极", 50)),
        },
    }

    json_path = OUTPUT_DIR / "analysis_payload.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, ensure_ascii=False, indent=2)

    def show_table(name: str, records: list[dict[str, Any]], cols: list[str], n: int = 8) -> None:
        print(f"\n{name}")
        if not records:
            print("(empty)")
            return
        frame = pd.DataFrame(records).head(n)
        print(frame[cols].to_string(index=False))

    print("输出:", json_path)
    print("诊断:", json.dumps(json_sanitize(payload["diagnostics"]), ensure_ascii=False, indent=2))
    print("总体:", json.dumps(json_sanitize(overall), ensure_ascii=False, indent=2))
    show_table("一级主题", payload["tables"]["primary"], ["primary_topic_label", "评论数", "积极占比", "中立占比", "消极占比", "平均情绪分", "获赞加权情绪分"])
    show_table("高消极二级主题", payload["tables"]["high_neg_secondary"], ["secondary_topic_label", "评论数", "积极占比", "中立占比", "消极占比", "平均情绪分"])
    show_table("学术分类", payload["tables"]["academic"], ["academic_class", "评论数", "积极占比", "中立占比", "消极占比", "平均情绪分"])


if __name__ == "__main__":
    main()
