# -*- coding: utf-8 -*-
"""Merge human review decisions for boundary comments into the final topic table.

Stage 3 of topic classification:

    bertopic_api.py  ->  build_boundary_review_workbook.py  ->  [human review]  ->  THIS SCRIPT

Input
-----
--review-file   The workbook produced by build_boundary_review_workbook.py after the
                reviewer has filled in the Boundary_manual_review sheet:
                  manual_decision           include | exclude           (required for every row)
                  manual_primary_topic_no    1..7                 (required when include)
                  manual_secondary_topic_no    e.g. 1.2             (optional)
                  reviewer / review_notes                           (optional)
                Optional second coder: manual_primary_topic_no_2 (and manual_decision_2).
                When present, percent agreement and Cohen's kappa are reported.

Output
------
comments_topic_final_<timestamp>.xlsx with
    Comments_updated      high-confidence comments + manually accepted boundary comments
                       (this is the sheet src/04_cross_analysis reads)
    Manual_review_summary       counts by decision / topic, agreement with the top-1 candidate
    Coder_agreement       percent agreement + Cohen's kappa (only with a second coder)
    Unreviewed_or_invalid       rows with missing / invalid decisions (excluded from the analysis)

Rows without a valid decision are NEVER assigned automatically; they are listed in
Unreviewed_or_invalid and left out of Comments_updated.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# Presentation names for this script only; source values and research rules are unchanged.
FIELD_NAMES = {
    "说明": "description",
    "项目": "item",
    "指标": "metric",
    "值": "value",
    "数值": "numeric_value",
    "样本数": "sample_count",
    "评论数": "comment_count",
    "一级主题编号": "primary_topic_number",
    "一级主题名称": "primary_topic_title",
    "完全不相关标记": "unrelated_flag",
    "人工复核_处理": "manual_decision",
    "人工复核_一级主题编号": "manual_primary_topic_no",
    "人工复核_二级主题编号": "manual_secondary_topic_no",
    "人工复核_处理_2": "manual_decision_2",
    "人工复核_一级主题编号_2": "manual_primary_topic_no_2",
    "复核人": "reviewer",
    "复核备注": "review_notes",
    "分类来源": "classification_source",
    "候选一级主题编号_1": "candidate_primary_no_1",
    "复核状态": "review_status",
    "人工纳入数": "manually_included_count"
}

SHEET_NAMES = {
    "说明": "Notes",
    "评论分类_更新": "Comments_updated",
    "边界样本_待人工复核": "Boundary_manual_review",
    "人工复核汇总": "Manual_review_summary",
    "编码者一致性": "Coder_agreement",
    "未复核或无效": "Unreviewed_or_invalid"
}

def english_name(name):
    return FIELD_NAMES.get(name, name)


def english_sheet(name):
    return SHEET_NAMES.get(name, name)


def resolve_sheet(names, requested):
    """Prefer the requested historical name; also accept the English export name."""
    return requested if requested in names else english_sheet(requested)


def _rename_checked(frame, mapping):
    columns = [mapping.get(c, c) for c in frame.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("Ambiguous table schema: column aliases would produce duplicate names")
    result = frame.copy()
    result.columns = columns
    result.index.names = [mapping.get(n, n) for n in result.index.names]
    return result


def export_frame(frame):
    """Return an English-header copy without changing any values or row order."""
    if isinstance(frame, pd.Series):
        result = frame.copy()
        result.name = english_name(result.name)
        result.index.names = [english_name(n) for n in result.index.names]
        return result
    return _rename_checked(frame, FIELD_NAMES)


def read_excel_compatible(path, *args, **kwargs):
    """Read either export schema into the unchanged historical internal schema.

    All pipeline calls use a single sheet. Keyword dtype keys are adapted to the
    actual sheet headers before parsing so string identifiers remain strings.
    """
    with pd.ExcelFile(path) as book:
        requested = kwargs.pop("sheet_name", args[0] if args else 0)
        rest = args[1:] if args else ()
        actual = resolve_sheet(book.sheet_names, requested)
        if isinstance(kwargs.get("dtype"), dict):
            headers = pd.read_excel(book, sheet_name=actual, nrows=0).columns
            kwargs["dtype"] = {
                k if k in headers else english_name(k): v
                for k, v in kwargs["dtype"].items()
            }
        frame = pd.read_excel(book, *rest, sheet_name=actual, **kwargs)
    return _rename_checked(frame, {v: k for k, v in FIELD_NAMES.items()})




HIGH_CONF_SHEET = "评论分类_更新"
REVIEW_SHEET = "边界样本_待人工复核"

PRIMARY_NAMES = {
    1: "婚恋关系与婚姻进入",
    2: "生育意愿与人生选择",
    3: "生育身体风险与医疗经验",
    4: "母职照护与产后处境",
    5: "婚育经济成本与生活机会",
    6: "性别权力与身体自主",
    7: "原生家庭与代际养育伦理",
}
ACCEPT = {"纳入", "include", "keep", "accept"}
REJECT = {"排除", "exclude", "drop", "reject"}


def _norm_decision(value) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text in ACCEPT:
        return "纳入"
    if text in REJECT:
        return "排除"
    return ""


def _norm_primary(value) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        no = int(float(str(value).strip()))
    except ValueError:
        return None
    return no if no in PRIMARY_NAMES else None


def _norm_secondary(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.1f}"
    return str(value).strip()


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    labels = sorted(set(a) | set(b))
    idx = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(axis=1) * m.sum(axis=0)).sum()) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description='Merge human boundary-review decisions into the final Comments_updated sheet')
    parser.add_argument("--review-file", type=Path, required=True, help='Review workbook with completed human decision columns')
    parser.add_argument("--out-file", type=Path, default=None, help='Output path; defaults to the input directory')
    args = parser.parse_args()

    high = read_excel_compatible(args.review_file, sheet_name=HIGH_CONF_SHEET)
    review = read_excel_compatible(args.review_file, sheet_name=REVIEW_SHEET)

    review["_decision"] = review["人工复核_处理"].map(_norm_decision)
    review["_primary"] = review["人工复核_一级主题编号"].map(_norm_primary)
    review["_secondary"] = review.get("人工复核_二级主题编号", pd.Series("", index=review.index)).map(_norm_secondary)

    valid_accept = review["_decision"].eq("纳入") & review["_primary"].notna()
    valid_reject = review["_decision"].eq("排除")
    invalid = ~(valid_accept | valid_reject)
    review["_status"] = np.select(
        [valid_accept, valid_reject, review["_decision"].eq("纳入") & review["_primary"].isna()],
        ["纳入", "排除", "无效：纳入但未填一级主题编号"],
        default="未复核",
    )

    accepted = review[valid_accept].copy()
    # Build rows in the same schema as the high-confidence sheet.
    merged_rows = high.copy()
    if len(accepted):
        add = pd.DataFrame(index=accepted.index)
        for col in high.columns:
            add[col] = accepted[col] if col in accepted.columns else np.nan
        pno = accepted["_primary"].astype(int)
        add["is_noise"] = False
        add["noise_reason"] = ""
        add["needs_manual_review"] = False
        add["review_reason"] = "manual_review"
        add["topic"] = pno - 1
        add["topic_name"] = pno.map(PRIMARY_NAMES)
        add["macro_topic"] = pno - 1
        add["macro_topic_name"] = pno.map(PRIMARY_NAMES)
        add["primary_topic"] = pno - 1
        add["primary_topic_no"] = pno
        add["primary_topic_name"] = pno.map(PRIMARY_NAMES)
        add["secondary_topic_no"] = accepted["_secondary"]
        add["macro_reason"] = "manual_review"
        add["分类来源"] = "边界样本_人工复核纳入"
        add["完全不相关标记"] = 0
        add["复核人"] = accepted.get("复核人", "")
        add["复核备注"] = accepted.get("复核备注", "")
        merged_rows = pd.concat([merged_rows, add[high.columns.tolist() + [c for c in ["复核人", "复核备注"] if c not in high.columns]]], ignore_index=True)

    # Summaries
    by_status = review["_status"].value_counts().rename_axis("复核状态").reset_index(name="评论数")
    by_topic = (
        accepted.groupby("_primary").size().rename("人工纳入数").reset_index()
        .rename(columns={"_primary": "一级主题编号"})
    )
    by_topic["一级主题名称"] = by_topic["一级主题编号"].map(PRIMARY_NAMES)
    cand_col = "候选一级主题编号_1"
    if cand_col in accepted.columns and len(accepted):
        agree = (accepted[cand_col].astype(float) == accepted["_primary"].astype(float)).mean()
        agree_rows = pd.DataFrame([["Agreement between human decisions and candidate 1", round(float(agree), 4), int(len(accepted))]],
                                  columns=["指标", "值", "样本数"])
    else:
        agree_rows = pd.DataFrame(columns=["指标", "值", "样本数"])

    coder_rows = []
    if "人工复核_一级主题编号_2" in review.columns:
        c1 = review["_primary"]
        c2 = review["人工复核_一级主题编号_2"].map(_norm_primary)
        both = c1.notna() & c2.notna()
        if both.sum() >= 2:
            a = c1[both].astype(int).astype(str)
            b = c2[both].astype(int).astype(str)
            coder_rows.append(["Primary topic percent agreement", round(float((a == b).mean()), 4), int(both.sum())])
            coder_rows.append(["Primary topic Cohen's kappa", round(cohen_kappa(a, b), 4), int(both.sum())])
        if "人工复核_处理_2" in review.columns:
            d1 = review["_decision"]
            d2 = review["人工复核_处理_2"].map(_norm_decision)
            bothd = d1.ne("") & d2.ne("")
            if bothd.sum() >= 2:
                coder_rows.append(["Include/exclude percent agreement", round(float((d1[bothd] == d2[bothd]).mean()), 4), int(bothd.sum())])
                coder_rows.append(["Include/exclude Cohen's kappa", round(cohen_kappa(d1[bothd], d2[bothd]), 4), int(bothd.sum())])
    coder = pd.DataFrame(coder_rows, columns=["指标", "值", "样本数"])

    notes = pd.DataFrame([
        ["High-confidence sample", int(len(high))],
        ["Total boundary comments", int(len(review))],
        ["Manually included", int(valid_accept.sum())],
        ["Manually excluded", int(valid_reject.sum())],
        ["Unreviewed/invalid (excluded from analysis)", int(invalid.sum())],
        ["Final analysis sample", int(len(merged_rows))],
    ], columns=["项目", "数值"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_file or args.review_file.parent / f"comments_topic_final_{timestamp}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        export_frame(notes).to_excel(writer, sheet_name=english_sheet("说明"), index=False)
        export_frame(merged_rows).to_excel(writer, sheet_name=english_sheet(HIGH_CONF_SHEET), index=False)
        export_frame(by_status).to_excel(writer, sheet_name=english_sheet("人工复核汇总"), index=False, startrow=0)
        export_frame(by_topic).to_excel(writer, sheet_name=english_sheet("人工复核汇总"), index=False, startrow=len(by_status) + 3)
        export_frame(agree_rows).to_excel(writer, sheet_name=english_sheet("人工复核汇总"), index=False, startrow=len(by_status) + len(by_topic) + 6)
        export_frame(coder).to_excel(writer, sheet_name=english_sheet("编码者一致性"), index=False)
        export_frame(review.loc[invalid].drop(columns=[c for c in review.columns if c.startswith("_")])).to_excel(writer, sheet_name=english_sheet("未复核或无效"), index=False)

    print(export_frame(notes).to_string(index=False))
    if len(coder):
        print(export_frame(coder).to_string(index=False))
    if invalid.sum():
        print(f"[warn] {int(invalid.sum())} boundary comments have missing or invalid decisions and are excluded from Comments_updated (see Unreviewed_or_invalid).")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
