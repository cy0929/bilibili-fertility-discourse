from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Category labels live in a shared module so every stage uses the same vocabulary.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_categories import (  # noqa: E402
    CATEGORY_COLUMN,
    CATEGORY_LABELS_ZH_TO_EN,
    standardize_category_column,
)


# Path priority: CLI argument > env var BFD_DATA_ROOT > default "data"
_DATA_ROOT = Path(os.environ.get("BFD_DATA_ROOT", "data"))
BASE = _DATA_ROOT / "derived"

# Topic classification result table (contains root_comment_id / macro_topic_name)
CLS_F = BASE / "comments_topic_classification.xlsx"
# Sentiment analysis result table (contains Comment_ID / sentiment_label)
SENT_F = BASE / "comments_sentiment.xlsx"
# Video info table (contains Video_ID / analytic category)
VIDEO_F = BASE / "video_info.xlsx"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="主题 × 视频学术分类的情感交叉统计（含点赞加权情感分，论文 RQ3/RQ4）"
    )
    parser.add_argument("--topic-file", type=Path, default=None, help="主题分类结果表（sheet: 评论分类_更新）")
    parser.add_argument("--sentiment-file", type=Path, default=None, help="情感分析结果表（sheet: Sheet1）")
    parser.add_argument("--video-file", type=Path, default=None, help="视频信息表（含「学术分类」列）")
    parser.add_argument("--out-csv", type=Path, default=None, help="交叉统计结果 CSV 输出路径")
    args = parser.parse_args()

    global CLS_F, SENT_F, VIDEO_F
    if args.topic_file:
        CLS_F = args.topic_file
    if args.sentiment_file:
        SENT_F = args.sentiment_file
    if args.video_file:
        VIDEO_F = args.video_file

    for label, path in [("主题分类结果表", CLS_F), ("情感分析结果表", SENT_F), ("视频信息表", VIDEO_F)]:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{label}不存在：{path}\n"
                f"请设置环境变量 BFD_DATA_ROOT（当前 {_DATA_ROOT}）并把文件放到 "
                f"<BFD_DATA_ROOT>/derived/ 下，或用 CLI 参数指定。详见 data/README.md"
            )

    cls = pd.read_excel(CLS_F, sheet_name="评论分类_更新", dtype={"root_comment_id": str, "video_id": str})
    sent = pd.read_excel(SENT_F, sheet_name="Sheet1", dtype={"Comment_ID": str, "Video_ID": str})
    # Accept `analytic_category` or the legacy `学术分类`, and normalise labels to English.
    video = standardize_category_column(pd.read_excel(VIDEO_F, dtype={"Video_ID": str}))

    print("rows", len(cls), len(sent), len(video))
    print("cls macro names")
    print(cls["macro_topic_name"].value_counts(dropna=False).head(10).to_string())
    print("sent labels")
    print(sent["sentiment_label"].value_counts(dropna=False).to_string())

    cls_ids = set(cls["root_comment_id"].dropna().astype(str))
    sent_ids = set(sent["Comment_ID"].dropna().astype(str))
    print("id overlap", len(cls_ids & sent_ids), "cls ids", len(cls_ids), "sent ids", len(sent_ids))

    merged = sent.merge(
        cls[["root_comment_id", "macro_topic_name"]],
        left_on="Comment_ID",
        right_on="root_comment_id",
        how="inner",
    )
    print("merged sent-cls", len(merged), merged["Comment_ID"].nunique())
    merged = merged.merge(video[["Video_ID", CATEGORY_COLUMN]], on="Video_ID", how="left")
    print("with video", len(merged), "missing academic", int(merged[CATEGORY_COLUMN].isna().sum()))

    grouped = (
        merged.groupby(["macro_topic_name", CATEGORY_COLUMN])
        .agg(
            count=("Comment_ID", "count"),
            positive=("sentiment_label", lambda x: (x == "积极").sum()),
            neutral=("sentiment_label", lambda x: (x == "中立").sum()),
            negative=("sentiment_label", lambda x: (x == "消极").sum()),
            mean_score=("sentiment_score", "mean"),
            likes=("Comment_Likes", "sum"),
        )
        .reset_index()
    )
    grouped["negative_pct"] = grouped["negative"] / grouped["count"]

    weighted = (
        merged.assign(weighted_score=merged["sentiment_score"] * merged["Comment_Likes"])
        .groupby(["macro_topic_name", CATEGORY_COLUMN])
        .agg(weighted_score=("weighted_score", "sum"), likes=("Comment_Likes", "sum"))
        .reset_index()
    )
    weighted["like_weighted_score"] = weighted["weighted_score"] / weighted["likes"]
    grouped = grouped.merge(
        weighted[["macro_topic_name", CATEGORY_COLUMN, "like_weighted_score"]],
        on=["macro_topic_name", CATEGORY_COLUMN],
        how="left",
    )

    top = grouped[grouped["count"] >= 100].sort_values(
        ["negative_pct", "count"], ascending=[False, False]
    )
    print("\nTOP >=100")
    print(
        top[
            [
                "macro_topic_name",
                CATEGORY_COLUMN,
                "count",
                "positive",
                "neutral",
                "negative",
                "negative_pct",
                "mean_score",
                "like_weighted_score",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            formatters={
                "negative_pct": pct,
                "mean_score": lambda x: f"{x:.3f}",
                "like_weighted_score": lambda x: f"{x:.3f}",
            },
        )
    )

    # Chinese literals below are the catchy source names, mapped through the shared
    # vocabulary rather than retyped, so they cannot drift out of sync.
    combos = [
        ("原生家庭与代际养育伦理", CATEGORY_LABELS_ZH_TO_EN["性别议题与权益探讨类"]),
        ("生育身体风险与医疗经验", CATEGORY_LABELS_ZH_TO_EN["经济基础与阶层生存类"]),
        ("生育身体风险与医疗经验", CATEGORY_LABELS_ZH_TO_EN["宏观政策与社会发展类"]),
        ("婚育经济成本与生活机会", CATEGORY_LABELS_ZH_TO_EN["性别议题与权益探讨类"]),
    ]
    print("\nREQUESTED COMBOS")
    for primary, academic in combos:
        row = grouped[
            (grouped["macro_topic_name"] == primary) & (grouped[CATEGORY_COLUMN] == academic)
        ]
        print(primary, "|", academic)
        print(
            row[
                [
                    "count",
                    "positive",
                    "neutral",
                    "negative",
                    "negative_pct",
                    "mean_score",
                    "like_weighted_score",
                    "likes",
                ]
            ].to_string(
                index=False,
                formatters={
                    "negative_pct": pct,
                    "mean_score": lambda x: f"{x:.3f}",
                    "like_weighted_score": lambda x: f"{x:.3f}",
                },
            )
        )

    out = args.out_csv or Path(os.environ.get("BFD_OUTPUT_ROOT", "output")) / "cross_analysis" / "topic_category_sentiment_cross_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out, index=False, encoding="utf-8-sig")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
