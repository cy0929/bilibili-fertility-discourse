# Data directory (`data/`)

**Everything under this directory is ignored by `.gitignore` and is not distributed
with the repository.** Collect the data yourself and place it here.

Bilibili comment and video data are subject to the platform's terms of service and to
user privacy, so they are not published as repository content.

## Expected layout

```
data/
├── raw/                              # Tables as collected
│   ├── video_info.xlsx               # Video metadata + hand-coded analytic category
│   └── comments_first_level.xlsx     # First-level comments with ≥ 5 likes
└── derived/                          # Pipeline intermediates produced by src/
    ├── comments_topic_classification.xlsx   # Topic classification output
    └── comments_sentiment.xlsx              # Sentiment analysis output
```

## Video analytic categories

The five categories below are the ones defined in the manuscript: *"Through iterative
discussion and consensus, we defined five mutually exclusive categories…"* The names here
are canonical. Earlier exports of the video table carry Chinese labels instead; the accepted
aliases and their mapping live in `src/video_categories.py`.

| # | Category | n |
|---|---|---|
| 1 | Micro Individual Experience and Lifestyle | 191 |
| 2 | Gender Issues and Rights Discussion | 32 |
| 3 | Macro Policy and Social Development | 23 |
| 4 | Economic Foundation and Class Survival | 13 |
| 5 | Pan-Entertainment and Cultural Projection | 10 |

Counts are the analysis sample (n = 269) after applying the exclusion rules described in
`src/03_interaction_metrics/compute_interaction_metrics.py`.

The scripts read either the canonical column `analytic_category` or a legacy column with
Chinese labels (see `src/video_categories.py`), and normalise both automatically, so either
generation of the source file works unchanged. To rewrite a workbook permanently:

```bash
python src/00_data_preparation/normalize_video_categories.py data/raw/video_info.xlsx
```

## Required columns

`video_info.xlsx`:

| Column | Meaning |
|---|---|
| `Video_ID` | Video identifier (e.g. `BV...`) |
| `Title` | Video title |
| `Views` | Play count (denominator of every rate) |
| `Comments` / `Bullets_count` / `Likes` / `Coins` / `Collections` / `Shares` | Six raw interaction counts |
| `analytic_category` | Hand-coded analytic category, one of the five above (aliases in `src/video_categories.py`) |

`comments_first_level.xlsx`:

| Column | Meaning |
|---|---|
| `Comment_ID` | Comment identifier; joins to `root_comment_id` in the topic output |
| `Video_ID` | Parent video; joins to `video_info.xlsx` |
| `Comment_Content` | Comment text (the field sent to the sentiment model) |
| `Comment_Likes` | Like count (weights the like-weighted score) |

Topic outputs must contain a sheet with `root_comment_id`, `primary_topic` and
`primary_topic_name`; sentiment outputs must contain `Comment_ID`,
`sentiment_label` and `sentiment_score`.

## Collection protocol

1. Search Bilibili with the 22 Chinese queries in [`../docs/search_keywords.csv`](../docs/search_keywords.csv);
2. Keep videos with ≥ 1,000,000 views and capture metadata plus the six interaction counts;
3. Capture comments and keep only **first-level** comments with ≥ 5 likes;
4. Hand-code one of the five analytic categories listed above for each video (researcher-coded, reached by iterative discussion to consensus);
5. Arrange the results into the two tables using the columns above.

> Comply with the Bilibili terms of service, robots directives, and applicable law on
> data collection. Throttle requests and do not collect personal information beyond
> what the analysis requires.

## Want to run only the interaction metrics?

`src/03_interaction_metrics/compute_interaction_metrics.py` depends solely on
`data/raw/video_info.xlsx` — no comment data needed — and reproduces Figure 1 on its own.
