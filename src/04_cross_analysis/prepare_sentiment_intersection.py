import collections
import datetime as dt
import json
import os
from pathlib import Path

import openpyxl


# Path priority: env vars BFD_DATA_ROOT / BFD_OUTPUT_ROOT > default "data" / "output"
_DATA_ROOT = Path(os.environ.get("BFD_DATA_ROOT", "data"))
BASE = _DATA_ROOT
RESULT = _DATA_ROOT / "derived"
WORK = Path(os.environ.get("BFD_OUTPUT_ROOT", "output")) / "sentiment_intersection"

# Sentiment analysis result table
SENTIMENT_FILE = RESULT / "comments_sentiment.xlsx"
# Topic classification result table (boundary-noise fine-grained classification version)
TOPIC_FILE = RESULT / "comments_topic_classification.xlsx"
# Cross-analysis summary table (output of analyze_sentiment_topic_video.py)
ANALYSIS_FILE = RESULT / "sentiment_topic_video_cross_analysis.xlsx"
OUT_JSON = WORK / "sentiment_intersection_data.json"


def json_value(value):
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    return value


def load_sheet_rows(path, sheet_name):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    iterator = ws.iter_rows(values_only=True)
    header = [json_value(v) for v in next(iterator)]
    rows = []
    for index, row in enumerate(iterator, start=2):
        values = [json_value(v) for v in row[: len(header)]]
        rows.append((index, dict(zip(header, values)), values))
    wb.close()
    return header, rows


def read_analysis_overall(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["01_总体情绪"]
    observed = {}
    for row in ws.iter_rows(min_row=1, max_row=20, max_col=4, values_only=True):
        label, count, _share, likes = row
        if label in {"积极", "中立", "消极"}:
            observed[label] = {"count": count, "likes": likes}
    wb.close()
    return observed


def main():
    WORK.mkdir(parents=True, exist_ok=True)

    sentiment_header, sentiment_rows = load_sheet_rows(SENTIMENT_FILE, "Sheet1")
    topic_header, topic_rows = load_sheet_rows(TOPIC_FILE, "评论分类_更新")

    comment_id_col = sentiment_header.index("Comment_ID")
    topic_id_col = topic_header.index("root_comment_id")

    first_sentiment_by_id = {}
    all_sentiment_by_id = collections.defaultdict(list)
    for row_index, row_dict, values in sentiment_rows:
        comment_id = values[comment_id_col]
        all_sentiment_by_id[comment_id].append((row_index, row_dict, values))
        first_sentiment_by_id.setdefault(comment_id, (row_index, row_dict, values))

    topic_ids = [values[topic_id_col] for _row_index, _row_dict, values in topic_rows]
    topic_id_set = set(topic_ids)
    sentiment_id_set = set(first_sentiment_by_id)
    duplicate_ids = {
        comment_id: row_entries
        for comment_id, row_entries in all_sentiment_by_id.items()
        if len(row_entries) > 1
    }

    missing_topic_ids = [comment_id for comment_id in topic_ids if comment_id not in first_sentiment_by_id]
    if missing_topic_ids:
        raise RuntimeError(f"Topic IDs not found in sentiment table: {len(missing_topic_ids)}")

    output_rows = []
    for comment_id in topic_ids:
        output_rows.append(first_sentiment_by_id[comment_id][2])

    label_idx = sentiment_header.index("sentiment_label")
    score_idx = sentiment_header.index("sentiment_score")
    likes_idx = sentiment_header.index("Comment_Likes")
    label_counts = collections.Counter(row[label_idx] for row in output_rows)
    likes_by_label = collections.Counter()
    scores = []
    for row in output_rows:
        likes_by_label[row[label_idx]] += row[likes_idx] or 0
        if isinstance(row[score_idx], (int, float)):
            scores.append(row[score_idx])

    duplicate_detail_header = [
        "Comment_ID",
        "是否在主题样本",
        "保留原表行号",
        "保留情绪",
        "保留分数",
        "保留点赞",
        "丢弃原表行号",
        "丢弃情绪",
        "丢弃分数",
        "丢弃点赞",
    ]
    duplicate_detail_rows = []
    for comment_id, entries in sorted(duplicate_ids.items()):
        kept = entries[0]
        for dropped in entries[1:]:
            duplicate_detail_rows.append(
                [
                    comment_id,
                    "是" if comment_id in topic_id_set else "否",
                    kept[0],
                    kept[1].get("sentiment_label"),
                    kept[1].get("sentiment_score"),
                    kept[1].get("Comment_Likes"),
                    dropped[0],
                    dropped[1].get("sentiment_label"),
                    dropped[1].get("sentiment_score"),
                    dropped[1].get("Comment_Likes"),
                ]
            )

    analysis_overall = read_analysis_overall(ANALYSIS_FILE)
    diagnostics = [
        ["指标", "数值", "说明"],
        ["情绪表数据行数", len(sentiment_rows), "原始情绪分析数据，不含表头"],
        ["情绪表唯一 Comment_ID", len(sentiment_id_set), "按 Comment_ID 去重后"],
        ["情绪表重复 Comment_ID 数", len(duplicate_ids), "存在 30 个重复 ID"],
        ["主题样本数据行数", len(topic_rows), "评论分类_更新，不含表头"],
        ["主题样本唯一 root_comment_id", len(topic_id_set), "该表无重复 root_comment_id"],
        ["交集唯一评论数", len(topic_id_set & sentiment_id_set), "root_comment_id ∩ Comment_ID"],
        ["情绪表未进入主题样本的唯一评论数", len(sentiment_id_set - topic_id_set), "保留在原始情绪表，但不进入 v2.0"],
        ["主题样本缺失情绪的评论数", len(topic_id_set - sentiment_id_set), "应为 0"],
        ["直接筛选会得到的行数", sum(1 for _idx, row, _values in sentiment_rows if row.get("Comment_ID") in topic_id_set), "因 28 个重复 ID 落在主题样本内，会多出 28 行"],
        ["v2.0 输出评论数", len(output_rows), "按主题样本顺序，每个 root_comment_id 保留情绪表第一次出现记录"],
        ["积极评论数", label_counts.get("积极", 0), "用于对齐现有交叉分析总体情绪"],
        ["中立评论数", label_counts.get("中立", 0), "用于对齐现有交叉分析总体情绪"],
        ["消极评论数", label_counts.get("消极", 0), "用于对齐现有交叉分析总体情绪"],
        ["平均情绪分", round(sum(scores) / len(scores), 6) if scores else None, "sentiment_score 平均值"],
        ["既有分析-积极评论数", analysis_overall.get("积极", {}).get("count"), "来自 01_总体情绪，用于校验"],
        ["既有分析-中立评论数", analysis_overall.get("中立", {}).get("count"), "来自 01_总体情绪，用于校验"],
        ["既有分析-消极评论数", analysis_overall.get("消极", {}).get("count"), "来自 01_总体情绪，用于校验"],
        ["生成口径", "Comment_ID = root_comment_id", "以主题样本为基准取交集，保留情绪表原字段"],
    ]

    payload = {
        "main_header": sentiment_header,
        "main_rows": output_rows,
        "diagnostics": diagnostics,
        "duplicate_detail_header": duplicate_detail_header,
        "duplicate_detail_rows": duplicate_detail_rows,
        "source_files": {
            "sentiment": str(SENTIMENT_FILE),
            "topic": str(TOPIC_FILE),
            "analysis": str(ANALYSIS_FILE),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "json": str(OUT_JSON),
        "output_rows": len(output_rows),
        "label_counts": dict(label_counts),
        "duplicate_ids": len(duplicate_ids),
        "duplicate_ids_in_topic": sum(1 for comment_id in duplicate_ids if comment_id in topic_id_set),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
