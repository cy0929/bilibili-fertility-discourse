# -*- coding: utf-8 -*-
"""Video-level analytic categories: canonical English labels and legacy Chinese aliases.

Every video in the study was hand-coded into exactly one of five mutually exclusive
categories. The English names below are the ones used in the manuscript:

    "Through iterative discussion and consensus, we defined five mutually exclusive
     categories: (1) Micro Individual Experience and Lifestyle, (2) Gender Issues and
     Rights Discussion, (3) Macro Policy and Social Development, (4) Economic Foundation
     and Class Survival, and (5) Pan-Entertainment and Cultural Projection."

Early exports of the video table carry the Chinese labels and the Chinese column name
`学术分类`. Both spellings are accepted on input and normalised to the English form, so
the analysis scripts run unchanged against either generation of the source file.

See ``data/README.md`` for the full label table.
"""
from __future__ import annotations

import pandas as pd

# --- Column name -------------------------------------------------------------
# `analytic_category` is the canonical name; `学术分类` is accepted for legacy files.
CATEGORY_COLUMN = "analytic_category"
LEGACY_CATEGORY_COLUMN = "学术分类"
CATEGORY_COLUMNS: tuple[str, ...] = (CATEGORY_COLUMN, LEGACY_CATEGORY_COLUMN)

# --- Category labels ---------------------------------------------------------
CATEGORY_LABELS: tuple[str, ...] = (
    "Micro Individual Experience and Lifestyle",
    "Gender Issues and Rights Discussion",
    "Macro Policy and Social Development",
    "Economic Foundation and Class Survival",
    "Pan-Entertainment and Cultural Projection",
)

CATEGORY_LABELS_ZH_TO_EN: dict[str, str] = {
    "微观个体经验与生活方式类": "Micro Individual Experience and Lifestyle",
    "性别议题与权益探讨类": "Gender Issues and Rights Discussion",
    "宏观政策与社会发展类": "Macro Policy and Social Development",
    "经济基础与阶层生存类": "Economic Foundation and Class Survival",
    "泛娱乐与文化投射类": "Pan-Entertainment and Cultural Projection",
}

# Fallback label for comments whose parent video is missing from the video table.
MISSING_VIDEO_METADATA = "Video metadata missing"


def resolve_category_column(df: pd.DataFrame) -> str:
    """Return the category column present in ``df`` (English name preferred)."""
    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            return col
    raise KeyError(
        f"缺少视频学术分类列，期望之一 {list(CATEGORY_COLUMNS)}；实际列：{list(df.columns)}"
    )


def normalize_category_values(values: pd.Series) -> pd.Series:
    """Map Chinese category labels to their English equivalents, leaving others intact."""
    stripped = values.astype("object").where(values.notna(), None)
    stripped = stripped.map(lambda v: v.strip() if isinstance(v, str) else v)
    mapped = stripped.map(lambda v: CATEGORY_LABELS_ZH_TO_EN.get(v, v))
    unexpected = {
        v for v in mapped.dropna().unique() if isinstance(v, str) and v not in CATEGORY_LABELS
    }
    if unexpected:
        print(
            f"[warn] 未识别的视频分类取值（保持原样）：{sorted(unexpected)}",
        )
    return mapped


def standardize_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with an `analytic_category` column holding English labels."""
    col = resolve_category_column(df)
    out = df.copy()
    out[CATEGORY_COLUMN] = normalize_category_values(out[col])
    if col != CATEGORY_COLUMN:
        out = out.drop(columns=[col])
    return out
