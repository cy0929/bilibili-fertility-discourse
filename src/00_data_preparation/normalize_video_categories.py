# -*- coding: utf-8 -*-
"""Normalise the hand-coded video category column to English labels.

Early exports of the video table use the Chinese column name `学术分类` and Chinese
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

_ENGLISH_LABELS = set(CATEGORY_LABELS)


def find_category_column(worksheet) -> int:
    """Return the 1-based index of the category column in the header row."""
    headers = [cell.value for cell in worksheet[1]]
    for name in CATEGORY_COLUMNS:
        if name in headers:
            return headers.index(name) + 1
    raise SystemExit(
        f"[error] 未找到视频分类列，期望之一 {list(CATEGORY_COLUMNS)}；表头：{headers}"
    )


def convert(src: Path, dst: Path, rename_column: bool = True) -> dict:
    """Rewrite the category values (and optionally the header) into ``dst``."""
    workbook = load_workbook(src)
    sheet = workbook[workbook.sheetnames[0]]
    if len(workbook.sheetnames) > 1:
        print(f"[warn] 工作簿含多个 sheet，只处理第一个：{workbook.sheetnames[0]}")

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

    workbook.save(dst)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把视频表的「学术分类」取值统一为英文标签（默认输出到新文件）"
    )
    parser.add_argument("input", type=Path, help="来源视频信息表 xlsx")
    parser.add_argument("--output", type=Path, default=None, help="输出路径，默认 <stem>_en.xlsx")
    parser.add_argument("--in-place", action="store_true", help="原地改写源文件（先生成 .bak.xlsx 备份）")
    parser.add_argument(
        "--keep-column-name",
        action="store_true",
        help="不重命名列，保留原来的列名（取值仍转为英文）",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"[error] 文件不存在：{args.input}", file=sys.stderr)
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
        print(f"[warn] 未识别的取值（保持原样）：{sorted(stats['unknown'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
