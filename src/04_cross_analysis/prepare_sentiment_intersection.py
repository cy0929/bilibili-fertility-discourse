import collections
import datetime as dt
import json
import os
from pathlib import Path

import openpyxl


# Presentation names for this script only; source values and research rules are unchanged.
FIELD_NAMES = {
    "是否在主题样本": "in_topic_sample",
    "保留原表行号": "retained_source_row",
    "保留情绪": "retained_sentiment",
    "保留分数": "retained_score",
    "保留点赞": "retained_likes",
    "丢弃原表行号": "discarded_source_row",
    "丢弃情绪": "discarded_sentiment",
    "丢弃分数": "discarded_score",
    "丢弃点赞": "discarded_likes",
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

SHEET_NAMES = {
    "评论分类_更新": "Comments_updated",
    "01_总体情绪": "01_Overall_sentiment"
}

def english_name(name):
    return FIELD_NAMES.get(name, name)



def english_headers(headers):
    """Translate copied source headers without changing any table data or order."""
    reserved = {h for h in headers if isinstance(h, str) and not any("\u3400" <= c <= "\u9fff" for c in h)}
    output = []
    for index, header in enumerate(headers, 1):
        name = header
        if isinstance(header, str) and any("\u3400" <= c <= "\u9fff" for c in header):
            name = FIELD_NAMES.get(header, f"source_field_{index}")
            while name in reserved:
                name += f"_source_{index}"
            reserved.add(name)
        output.append(name)
    return output


def english_sheet(name):
    return SHEET_NAMES.get(name, name)


def resolve_sheet(names, requested):
    """Prefer the requested historical name; also accept the English export name."""
    return requested if requested in names else english_sheet(requested)




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
    ws = wb[resolve_sheet(wb.sheetnames, sheet_name)]
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
    ws = wb[resolve_sheet(wb.sheetnames, "01_总体情绪")]
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
        ["Metric", "Value", "Description"],
        ["Sentiment table data rows", len(sentiment_rows), "Original sentiment data, excluding the header"],
        ["Unique sentiment Comment_ID values", len(sentiment_id_set), "After deduplication by Comment_ID"],
        ["Duplicated sentiment Comment_ID values", len(duplicate_ids), "The historical dataset contains 30 duplicated IDs"],
        ["Topic sample data rows", len(topic_rows), "Comments_updated, excluding the header"],
        ["Unique topic root_comment_id values", len(topic_id_set), "The historical table has no duplicated root_comment_id values"],
        ["Unique comments in the intersection", len(topic_id_set & sentiment_id_set), "root_comment_id ∩ Comment_ID"],
        ["Unique sentiment comments outside the topic sample", len(sentiment_id_set - topic_id_set), "Retained in the original sentiment table; excluded from v2.0"],
        ["Topic comments missing sentiment", len(topic_id_set - sentiment_id_set), "Expected: 0"],
        ["Rows obtained by direct filtering", sum(1 for _idx, row, _values in sentiment_rows if row.get("Comment_ID") in topic_id_set), "The historical topic sample contains 28 duplicated IDs, producing 28 extra rows"],
        ["v2.0 output comment count", len(output_rows), "Follow topic sample order and retain the first sentiment record for each root_comment_id"],
        ["Positive comment count", label_counts.get("积极", 0), "For comparison with overall sentiment in the existing cross analysis"],
        ["Neutral comment count", label_counts.get("中立", 0), "For comparison with overall sentiment in the existing cross analysis"],
        ["Negative comment count", label_counts.get("消极", 0), "For comparison with overall sentiment in the existing cross analysis"],
        ["Mean sentiment score", round(sum(scores) / len(scores), 6) if scores else None, "Mean of sentiment_score"],
        ["Existing analysis: positive comment count", analysis_overall.get("积极", {}).get("count"), "From the overall sentiment sheet, for validation"],
        ["Existing analysis: neutral comment count", analysis_overall.get("中立", {}).get("count"), "From the overall sentiment sheet, for validation"],
        ["Existing analysis: negative comment count", analysis_overall.get("消极", {}).get("count"), "From the overall sentiment sheet, for validation"],
        ["Construction rule", "Comment_ID = root_comment_id", "Intersect with the topic sample as the reference; retain the original sentiment fields"],
    ]

    payload = {
        "main_header": english_headers(sentiment_header),
        "main_rows": output_rows,
        "diagnostics": diagnostics,
        "duplicate_detail_header": [english_name(h) for h in duplicate_detail_header],
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
