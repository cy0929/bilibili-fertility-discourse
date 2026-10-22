# Digital Discourse on Marriage and Childbearing among Chinese Youth

**Analysis code for the Bilibili marriage-and-childbearing discourse study.**

This repository contains the computational analysis pipeline behind a study of
marriage-and-childbearing discourse on Bilibili: standardized interaction metrics,
embedding-based topic classification driven by a semantic codebook, LLM-based
sentiment analysis, and the cross-dimensional / like-weighted visibility analysis.

**This repository ships code and method materials only — no data.**
See [`data/README.md`](data/README.md) for how to prepare inputs.

---

## Study at a glance

| | |
|---|---|
| **Platform** | Bilibili |
| **Videos** | 269 videos with ≥ 1,000,000 views (272 retrieved → 269 after screening) |
| **Comments** | 53,534 first-level comments with ≥ 5 likes (replies removed during preprocessing) |
| **Retrieval** | 16 English keyword groups → 22 Chinese queries ([`docs/search_keywords.csv`](docs/search_keywords.csv)) |
| **Embedding model** | `Qwen/Qwen3-Embedding-8B`, served from a locally deployed OpenAI-compatible endpoint |
| **Sentiment model** | `deepseek-ai/DeepSeek-V4-Pro`, same local endpoint (JSON mode, temperature 0.1, batches of 50 comments) |
| **Privacy** | Only comment text + video title are sent to the model; commenter names/IDs are never read or transmitted |
| **Topic thresholds** | codebook similarity ≥ 0.28, top-1/top-2 margin ≥ 0.025, keyword boost 0.030 per weight unit (weights 2.5/2.0/1.5/1.0 by phrase length, capped at 0.090), rebalance from topic 1 up to 0.180 |
| **Video categories** | 5 hand-coded analytic categories ([`data/README.md`](data/README.md)) |
| **Comment topics** | 7 mutually exclusive first-level topics (semantic codebook + embedding similarity) |

Method materials kept in the repository:

- [`docs/search_keywords.csv`](docs/search_keywords.csv) — the 22 Chinese retrieval queries and their English source groups.
- [`docs/topic_codebook_zh.md`](docs/topic_codebook_zh.md) — the semantic codebook: definitions, inclusion and exclusion rules for the 7 first-level topics.

---

## Repository layout

```
.
├── README.md                      This file
├── LICENSE
├── requirements.txt              Python dependencies
├── .gitignore
├── config/
│   ├── bertopic_config.example.yaml      Topic-classification config template
│   ├── sentiment_config.example.toml     Sentiment-analysis config template
│   └── sentiment_prompt.md               Sentiment prompt and labelling rules
├── lexicons/
│   ├── stopwords_bilibili.txt            Stop-word list for the Bilibili context
│   ├── fertility_dict.txt                Domain dictionary (jieba user dictionary)
│   └── irrelevant_keywords_bilibili.txt  Filter list for irrelevant content
├── src/
│   ├── video_categories.py                                Shared vocabulary: 5 category labels, ZH→EN aliases
│   ├── 00_data_preparation/
│   │   └── normalize_video_categories.py                  Rewrite a workbook's category column into English
│   ├── 01_topic_classification/
│   │   ├── bertopic_api.py                                    Stage 1: codebook + embedding classification
│   │   ├── build_boundary_review_workbook.py                  Stage 2: review workbook for boundary comments
│   │   └── apply_manual_review.py                             Stage 3: merge human decisions into the final table
│   ├── 02_sentiment_analysis/
│   │   └── sentiment_analysis_llm.py                          LLM sentiment analysis (concurrency / retry / resume)
│   ├── 03_interaction_metrics/
│   │   └── compute_interaction_metrics.py                     Six-dimensional normalized rates + ANOVA
│   └── 04_cross_analysis/
│       ├── analyze_sentiment_topic_video.py                   Cross analysis (chi-square / like-weighted / correlation)
│       ├── compute_cross_analysis.py                          Topic × video-category sentiment cross-tab
│       └── prepare_sentiment_intersection.py                  Sentiment × topic intersection export
├── docs/
│   ├── search_keywords.csv        22 Chinese queries ↔ 16 English keyword groups
│   ├── topic_codebook_zh.md       Semantic codebook for the 7 first-level topics
│   └── local_llm_endpoint.md      How to serve the two models locally (OpenAI-compatible API)
└── legacy/
    └── build_forced_boundary_classification_workbook.py   Deprecated: automatic assignment of boundary cases (not used)
```

> **Note on `bertopic_api.py`:** the script supports two paths — full unsupervised
> BERTopic clustering (UMAP + HDBSCAN + c-TF-IDF) and, under
> `topic_modeling.run_bertopic: false`, a path that **skips clustering entirely** and
> assigns first-level topics directly from the semantic codebook. The study used the
> latter (see `config/bertopic_config.example.yaml`); in that mode `micro_topic` and
> `topic_probability` are placeholder fields and `macro_topic` carries the final label.

---

## Setup

Requires **Python 3.11+** (`sentiment_analysis_llm.py` uses the standard-library `tomllib`).

```bash
git clone <this-repo> && cd bilibili-fertility-discourse
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Both models are called through a **locally deployed OpenAI-compatible endpoint**
(see [`docs/local_llm_endpoint.md`](docs/local_llm_endpoint.md)). No cloud API is used and no
comment text leaves the machine. Point the scripts at your server:

```bash
# Windows
setx LLM_API_BASE_URL "http://localhost:8000/v1"
# Linux / macOS
export LLM_API_BASE_URL=http://localhost:8000/v1
# Optional: only if your local server enforces a bearer token
export LLM_API_KEY=your-token
```

If the variables are unset, the scripts default to `http://localhost:8000/v1` and send the
placeholder token `EMPTY`.

Copy the config templates and adjust them as needed:

```bash
cp config/bertopic_config.example.yaml  config/bertopic_config.yaml
cp config/sentiment_config.example.toml config/sentiment_config.toml
```

> `config/*.yaml` and `config/*.toml` (non-`.example`) are listed in `.gitignore`
> and will never be committed.

---

## Usage

All scripts accept `--help`. See [`data/README.md`](data/README.md) for the input
tables and the columns each stage requires.

> **Always run scripts from the repository root** (`cd bilibili-fertility-discourse`).
> Relative paths inside the config files (e.g. `data/raw/...`, `lexicons/...`) resolve
> against the **current working directory**, not against the directory of the config file.
> Note in particular that the `user_dict` domain dictionary is **silently skipped** when
> the path does not exist — no error is raised, but segmentation and classification
> results change. After a run, check the log for
> `User dictionary: lexicons/fertility_dict.txt` to confirm it loaded.

**0. Optional — normalise the video category column to English**:

```bash
python src/00_data_preparation/normalize_video_categories.py data/raw/video_info.xlsx
```

All downstream scripts already accept either the Chinese or the English labels, so this
step is only needed if you want the stored data itself in English. See the five canonical
category names in [`data/README.md`](data/README.md).

**1. Six-dimensional interaction metrics** — runs standalone, needs only the video table:

```bash
python src/03_interaction_metrics/compute_interaction_metrics.py \
    --video-file data/raw/video_info.xlsx \
    --out-dir output/interaction_metrics
```

**2. Topic classification** (three stages; boundary comments are never assigned automatically):

```bash
# Stage 1 — semantic-codebook classification. Comments below the similarity threshold
# (0.28) or with an insufficient top-1/top-2 margin (0.025) are flagged as boundary cases.
python src/01_topic_classification/bertopic_api.py --config config/bertopic_config.yaml

# Stage 2 — review workbook: sheet Comments_updated holds ONLY the high-confidence comments;
# sheet Boundary_manual_review lists every boundary comment with its two nearest candidate
# topics, a rule-based suggestion (e.g. platform chatter → suggest exclude), and empty
# columns manual_decision / manual_primary_topic_no / manual_secondary_topic_no for the reviewer.
python src/01_topic_classification/build_boundary_review_workbook.py \
    --input-dir output/topic_classification

# Stage 3 — after the reviewer(s) fill in the decisions, merge accepted comments into the
# final table. Rows without a valid decision stay out of the analysis. If a second coder
# fills manual_primary_topic_no_2 / manual_decision_2, percent agreement and Cohen's kappa are reported.
python src/01_topic_classification/apply_manual_review.py \
    --review-file output/topic_classification/comments_boundary_review_<run_tag>_<timestamp>.xlsx
```

**3. Sentiment analysis**:

```bash
# Dry-run 20 rows first to confirm the API and JSON parsing work
python src/02_sentiment_analysis/sentiment_analysis_llm.py \
    --config config/sentiment_config.toml --dry-run --limit 20

# Full run (resumable: re-run the same command after an interruption)
python src/02_sentiment_analysis/sentiment_analysis_llm.py \
    --config config/sentiment_config.toml
```

**4. Cross analysis**:

> **Wiring note:** stages 2 and 3 write timestamped files under `output/`, while stage 4
> defaults to fixed filenames under `data/derived/`. Pick either approach:
> - **A (recommended):** point the CLI flags at the actual stage 2 / 3 outputs, no file moving;
> - **B:** copy the outputs to `data/derived/comments_topic_classification.xlsx` and
>   `data/derived/comments_sentiment.xlsx` (the default names).

```bash
# Approach A
python src/04_cross_analysis/analyze_sentiment_topic_video.py \
    --sentiment-file output/sentiment/20260531162934_..._sentiment.xlsx \
    --topic-file     output/topic_classification/comments_topic_final_<timestamp>.xlsx \
    --video-file     data/raw/video_info.xlsx \
    --out-dir        output/sentiment_topic_video_analysis

# Approach B
python src/04_cross_analysis/analyze_sentiment_topic_video.py \
    --sentiment-file data/derived/comments_sentiment.xlsx \
    --topic-file     data/derived/comments_topic_classification.xlsx \
    --video-file     data/raw/video_info.xlsx \
    --out-dir        output/sentiment_topic_video_analysis

python src/04_cross_analysis/compute_cross_analysis.py
```

> `--topic-file` must point to the workbook written by `apply_manual_review.py`
> (sheet `Comments_updated` = high-confidence comments + manually accepted boundary comments).

Environment variables can replace the individual flags:

```bash
export BFD_DATA_ROOT=data
export BFD_OUTPUT_ROOT=output
export BFD_TOPIC_OUTPUT_DIR=output/topic_classification
```

---

## Verified reproduction

`compute_interaction_metrics.py` was re-run against the original video table and
**reproduces Figure 1 of the paper (and the Table S1 sample counts) exactly**:

- Sample size 269 (Micro Individual Experience and Lifestyle 191 / Gender Issues and Rights Discussion 32 / Macro Policy and Social Development 23 / Economic Foundation and Class Survival 13 / Pan-Entertainment and Cultural Projection 10) ✔
- 5 categories × 6 metrics = 30 group means, all matching the published values with **zero deviation** ✔
- Also emits ANOVA results not reported in the paper: comment rate F = 10.757, p < 0.001, η² = 0.140; danmaku rate F = 5.522, p < 0.001, η² = 0.077; the remaining four dimensions are not significant.

The run was repeated against both the legacy Chinese-labelled export and the English-labelled
one produced by `src/00_data_preparation/normalize_video_categories.py`. All 13 numeric output
columns are identical (maximum absolute difference 0.0) across the two inputs, so switching the
labels changes presentation only, never results.


## English presentation and historical input compatibility

Generated topic/review workbooks, statistical CSV headers and cross-analysis JSON
field names use English. Readers accept both these English export names and the
historical Chinese worksheet/header names through local presentation mappings
inside the original scripts. Duplicate aliases are rejected rather than silently
losing a field. Column order and cell values are preserved. The repository keeps
its original 27 files; audit records and synthetic tests are kept outside it.

In `Boundary_manual_review`, enter `include` or `exclude` in `manual_decision`,
and a primary topic number (1–7) for included comments. Secondary topic numbers,
reviewer names and notes remain optional. Historical Chinese decisions continue
to be accepted. Blank or invalid decisions remain excluded. Optional second-coder
columns are `manual_primary_topic_no_2` and `manual_decision_2`.

Chinese topic names, semantic codebooks, keywords, regular expressions, sentiment
labels, review status values and prompts are preserved as research material.
The sentiment script translates headers and sheet names only when saving its
output. Before resuming, it recovers source labels from the unchanged original
input workbook, so model context and worksheet selection remain the same. Comment
text, topic names and all result values remain unchanged. Unrecognized source
headers/sheet names receive deterministic positional English identifiers rather
than guessed translations. The category-normalization utility still accepts
`--keep-column-name` for CLI compatibility, but its final export headers are
always English. No input workbooks are edited as part of this update.

The boundary review sheet historically omits `root_comment_id` from its selected
columns. A manually accepted row can therefore lack that identifier in the final
table. This existing limitation is documented without changing the sample or
inventing identifiers. The deprecated legacy script remains deprecated; this
presentation update does not reconstruct or revise historical boundary assignments.
