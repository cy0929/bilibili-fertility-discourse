# -*- coding: utf-8 -*-
"""Six-dimensional normalized interaction metrics (paper RQ1 / Figure 1 / Table S1).

For each video, compute 6 metrics of "interaction counts per 1,000 views":

    rate_ik = count_ik / views_i * 1000

where k ∈ {comment, danmaku, like, coin, favorite, share}.
Then group by "video academic category" (5 categories) and compute means,
followed by one-way ANOVA + effect size eta-squared (η²).

Sample criteria (consistent with the paper, n = 269):
    1. `Views` > 0 and non-missing;
    2. hand-coded analytic category present;
    3. exclude the videos listed in EXCLUDED_VIDEO_IDS (invalid videos +
       samples excluded from the main analysis).

Usage:
    python compute_interaction_metrics.py \
        --video-file data/raw/video_info.xlsx \
        --out-dir output/interaction_metrics

Environment variables:
    BFD_DATA_ROOT   data root directory, default "data" (searched under this
                    directory when --video-file is omitted)

Outputs:
    interaction_rates_by_video.csv    the 6 normalized metrics per video
    interaction_means_by_category.csv means per analytic category (paper Figure 1)
    interaction_anova.csv             ANOVA results for the 6 metrics (F / p / η²)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Category labels live in a shared module so every stage uses the same vocabulary.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_categories import CATEGORY_COLUMN, CATEGORY_LABELS, standardize_category_column  # noqa: E402


# Presentation names for this script only; source values and research rules are unchanged.
FIELD_NAMES = {
    "评论率": "comment_rate",
    "弹幕率": "danmaku_rate",
    "点赞率": "like_rate",
    "投币率": "coin_rate",
    "收藏率": "favorite_rate",
    "分享率": "share_rate"
}

def english_name(name):
    return FIELD_NAMES.get(name, name)


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



# --- Metric definitions: output column name -> raw count column in the video table ---
INTERACTION_METRICS: dict[str, str] = {
    "comment_rate": "Comments",        # comment rate: in-depth textual expression
    "danmaku_rate": "Bullets_count",   # danmaku rate: immediate emotional response
    "like_rate": "Likes",              # like rate: basic approval (unlimited single click)
    "coin_rate": "Coins",              # coin rate: costly approval (limited daily coin quota)
    "favorite_rate": "Collections",    # favorite rate: retention intent
    "share_rate": "Shares",            # share rate: cross-platform diffusion
}

# Chinese header aliases, used for output and cross-reference with paper Figure 1
METRIC_LABELS_ZH: dict[str, str] = {
    "comment_rate": "评论率",
    "danmaku_rate": "弹幕率",
    "like_rate": "点赞率",
    "coin_rate": "投币率",
    "favorite_rate": "收藏率",
    "share_rate": "分享率",
}

# Category column: `analytic_category` (legacy Chinese exports use the legacy Chinese category header).
# Both the column name and the Chinese label values are normalised to English on load;
# see src/video_categories.py for the five canonical labels.
VIEWS_COLUMN = "Views"
VIDEO_ID_COLUMN = "Video_ID"
INVALID_FLAG_COLUMN = "已失效视频"

# --- Exclusion list for the main analysis ------------------------------------
# BV1AgcseTEDU: invalid video (flagged as 1 in the source table's the source invalid-video flag column)
# BV1jm421g7pS: sample excluded from the main analysis
# After exclusion n = 269 (micro 191 / gender 32 / macro-policy 23 / economic 13 /
# general entertainment 10), exactly consistent with paper Table S1 and Figure 1.
EXCLUDED_VIDEO_IDS: tuple[str, ...] = (
    "BV1AgcseTEDU",
    "BV1jm421g7pS",
)

PER_MILLE = 1000.0


def resolve_default_video_file() -> Path:
    """Locate the video info table by common naming under BFD_DATA_ROOT."""
    root = Path(os.environ.get("BFD_DATA_ROOT", "data"))
    candidates = [
        root / "raw" / "video_info.xlsx",
        root / "video_info.xlsx",
    ]
    candidates.extend(sorted((root / "raw").glob("*视频*信息*.xlsx")) if (root / "raw").is_dir() else [])
    candidates.extend(sorted(root.glob("*视频*信息*.xlsx")))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Video information table not found. Specify --video-file or place it in "
        f"{root / 'raw' / 'video_info.xlsx'}"
    )


def load_and_filter(video_file: Path) -> pd.DataFrame:
    """Read the video table and apply the paper's sample criteria."""
    df = pd.read_excel(video_file, dtype={VIDEO_ID_COLUMN: str})

    # Accept either `analytic_category` or the legacy the legacy Chinese category header, and normalise both the
    # column name and the Chinese label values to the canonical English form.
    df = standardize_category_column(df)

    missing = [
        col
        for col in [VIEWS_COLUMN, CATEGORY_COLUMN, VIDEO_ID_COLUMN, *INTERACTION_METRICS.values()]
        if col not in df.columns
    ]
    if missing:
        raise KeyError(f"Video information table is missing required columns: {missing}; actual columns: {list(df.columns)}")

    total = len(df)

    views = pd.to_numeric(df[VIEWS_COLUMN], errors="coerce")
    step_views = df[views.notna() & (views > 0)]

    step_coded = step_views[step_views[CATEGORY_COLUMN].notna()]

    if INVALID_FLAG_COLUMN in step_coded.columns:
        step_valid = step_coded[step_coded[INVALID_FLAG_COLUMN].isna()]
    else:
        step_valid = step_coded

    step_final = step_valid[~step_valid[VIDEO_ID_COLUMN].isin(EXCLUDED_VIDEO_IDS)].copy()

    print(f"[filter] Input rows: {total}")
    print(f"[filter] Views > 0           : {len(step_views)}")
    print(f"[filter] Coded analytic category: {len(step_coded)}")
    print(f"[filter] After excluding invalid videos: {len(step_valid)}")
    print(f"[filter] After main analysis exclusions: {len(step_final)}  <- Paper analysis sample size")
    return step_final


def compute_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 6 normalized interaction metrics per video (per 1,000 views)."""
    out = df.copy()
    views = pd.to_numeric(out[VIEWS_COLUMN], errors="coerce")
    for metric, source in INTERACTION_METRICS.items():
        count = pd.to_numeric(out[source], errors="coerce")
        out[metric] = count / views * PER_MILLE
    return out


def summarise_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate means per analytic category, output in the paper Figure 1 form (with video counts)."""
    metrics = list(INTERACTION_METRICS)
    grouped = df.groupby(CATEGORY_COLUMN, dropna=False)[metrics].mean()
    counts = df.groupby(CATEGORY_COLUMN, dropna=False).size().rename("n_videos")
    summary = pd.concat([counts, grouped], axis=1).reset_index()

    summary = summary.rename(
        columns={
            CATEGORY_COLUMN: "video_category",
            **{m: METRIC_LABELS_ZH[m] for m in metrics},
        }
    )
    numeric_cols = ["n_videos", *[METRIC_LABELS_ZH[m] for m in metrics]]
    for col in numeric_cols[1:]:
        summary[col] = summary[col].round(3)
    return summary.sort_values("n_videos", ascending=False).reset_index(drop=True)


def one_way_anova(df: pd.DataFrame) -> pd.DataFrame:
    """Run one-way ANOVA for each of the 6 metrics (independent variable = analytic category), reporting F / p / eta-squared (η²)."""
    try:
        from scipy import stats
    except ImportError:  # pragma: no cover
        print('[warn] scipy is unavailable; skipping ANOVA. Install with: pip install scipy', file=sys.stderr)
        return pd.DataFrame()

    groups = [sub for _, sub in df.groupby(CATEGORY_COLUMN, dropna=True)]
    rows = []
    for metric in INTERACTION_METRICS:
        samples = [pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy() for g in groups]
        samples = [s for s in samples if len(s) > 0]
        if len(samples) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*samples)

        # η² = SS_between / SS_total
        all_values = np.concatenate(samples)
        grand_mean = all_values.mean()
        ss_total = float(np.sum((all_values - grand_mean) ** 2))
        ss_between = float(
            sum(len(s) * (s.mean() - grand_mean) ** 2 for s in samples)
        )
        eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")

        df_between = len(samples) - 1
        df_within = len(all_values) - len(samples)
        rows.append(
            {
                "metric": metric,
                "metric_zh": METRIC_LABELS_ZH[metric],
                "F": round(float(f_stat), 3),
                "df1": df_between,
                "df2": df_within,
                "p": float(p_val),
                "eta_squared": round(float(eta_sq), 3),
                "significant": "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description='Compute six normalized interaction metrics (paper RQ1)')
    parser.add_argument("--video-file", type=Path, default=None, help='Video information workbook path')
    parser.add_argument("--out-dir", type=Path, default=Path("output/interaction_metrics"), help='Output directory')
    args = parser.parse_args()

    video_file = args.video_file or resolve_default_video_file()
    if not video_file.is_file():
        print(f"[error] Video information table not found: {video_file}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    filtered = load_and_filter(video_file)
    with_rates = compute_rates(filtered)

    per_video_path = args.out_dir / "interaction_rates_by_video.csv"
    keep_cols = [
        VIDEO_ID_COLUMN,
        "Title",
        "UP_name",
        CATEGORY_COLUMN,
        VIEWS_COLUMN,
        *INTERACTION_METRICS.values(),
        *INTERACTION_METRICS,
    ]
    keep_cols = [c for c in keep_cols if c in with_rates.columns]
    export_frame(with_rates[keep_cols]).to_csv(per_video_path, index=False, encoding="utf-8-sig")

    summary = summarise_by_category(with_rates)
    summary_path = args.out_dir / "interaction_means_by_category.csv"
    export_frame(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")

    anova = one_way_anova(with_rates)
    anova_path = args.out_dir / "interaction_anova.csv"
    if not anova.empty:
        export_frame(anova).to_csv(anova_path, index=False, encoding="utf-8-sig")

    print('\n=== Mean normalized interaction metrics by analytic category (paper Figure 1) ===')
    print(export_frame(summary).to_string(index=False))
    if not anova.empty:
        print('\n=== One-way ANOVA (factor: analytic category) ===')
        print(export_frame(anova).to_string(index=False))
    print(f"\n[out] {per_video_path}")
    print(f"[out] {summary_path}")
    if not anova.empty:
        print(f"[out] {anova_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
