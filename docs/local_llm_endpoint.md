# Serving the two models locally

All scripts talk to **one OpenAI-compatible HTTP endpoint** (`LLM_API_BASE_URL`, default
`http://localhost:8000/v1`). Any server that implements `/v1/embeddings` (for
`Qwen/Qwen3-Embedding-8B`) and `/v1/chat/completions` (for `deepseek-ai/DeepSeek-V4-Pro`)
works: vLLM, SGLang, LM Studio, Ollama, llama.cpp server, or a gateway in front of them.
Nothing in the pipeline depends on a specific vendor; comment text is only ever posted to
this endpoint.

## What the scripts expect

| Script | Endpoint | Model name sent | Notes |
|---|---|---|---|
| `src/01_topic_classification/bertopic_api.py` | `POST /v1/embeddings` | `api.model` in `bertopic_config.yaml` (default `Qwen/Qwen3-Embedding-8B`) | batches of 32 texts, ≤ 1,500 chars each; results cached on disk by text hash |
| `src/02_sentiment_analysis/sentiment_analysis_llm.py` | `POST /v1/chat/completions` | `model.name` in `sentiment_config.toml` (default `deepseek-ai/DeepSeek-V4-Pro`) | JSON mode (`response_format`), temperature 0.1, 50 comments per request |

The model name must match what your server registers (`GET /v1/models`). If you serve the
models under different names, change `api.model` / `model.name` or set `LLM_MODEL`.

## Typical setups

- **Two vLLM processes** (one embedding model, one chat model) behind a small reverse proxy
  that routes `/v1/embeddings` and `/v1/chat/completions` to the right port — or simply run
  the two pipeline stages at different times against different `LLM_API_BASE_URL` values.
- **A single gateway** (e.g. LiteLLM) that exposes both models on one base URL.

Consult the server's own documentation for the exact launch flags of each model
(embedding task mode, context length, JSON-mode support). The repository does not pin a
particular server.

## Determinism

Sentiment labels are generated with `temperature = 0.1` and no fixed seed; repeated runs are
not guaranteed to be identical comment-by-comment. Report the server, model build, and run
date together with any re-analysis.

## Authentication

Local servers normally ignore the bearer token. The scripts send `LLM_API_KEY` if set,
otherwise the placeholder `EMPTY`. Never commit a real token.
