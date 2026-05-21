# q3_reranker — LLM listwise reranking + citation-graph reranking

Q3 submission for the ASAT task: pick one key step from **Asta's Find Papers** workflow, build a SOTA solution, evaluate it, and propose a novel improvement.

**Step chosen:** reranking. Given a pool of papers retrieved from an academic search source, rerank them so the most relevant ones appear first. This project implements seven rankers on top of three retrieval backends, runs a gold-labeled evaluation across 8 benchmark stages, and exposes everything through a side-by-side CLI demo and a browser-based web demo.

---

## Table of contents

1. [Architecture](#architecture)
2. [Quick start](#quick-start)
3. [Environment variables](#environment-variables)
4. [Module reference](#module-reference)
5. [CLI commands](#cli-commands)
6. [Web demo](#web-demo)
7. [Evaluation results](#evaluation-results)
8. [Gold benchmark stages](#gold-benchmark-stages)
9. [Status](#status)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Query (keyword form for retrieval)             │
└───────────────────────┬─────────────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │   sources.py       │  RETRIEVAL_SOURCE env var
              │   dispatcher       │
              └──┬─────────────┬───┘
                 │             │
    ┌────────────▼──┐     ┌────▼────────────────┐
    │ retriever.py  │     │ serpapi_retriever.py │
    │ Semantic      │     │ SerpAPI / Google     │
    │ Scholar API   │     │ Scholar              │
    └──────┬─────── ┘     └────┬────────────────┘
           └──────────┬────────┘
                      │  list[Paper]  (SHA256-cached on disk)
                      │
        ┌─────────────▼──────────────────────────┐
        │              Rankers                   │
        │                                        │
        │  ss_default   trivial baseline         │
        │  bm25          BM25 over title+abstract│
        │  dense         MiniLM-L6 cosine sim    │
        │  rrf           Reciprocal Rank Fusion  │
        │                (default+bm25+dense)    │
        │  cross_encoder ms-marco MiniLM-L12     │
        │                full attention reranker │
        │  llm_rerank    RankGPT sliding window  │
        │                (DeepSeek / Claude)     │
        │  llm+citation  LLM score blended with  │
        │                in-set citation graph   │
        └─────────────┬──────────────────────────┘
                      │  list[RankedResult]
                      │
        ┌─────────────▼──────────────────────────┐
        │           Evaluation / Demo            │
        │  NDCG@10 / MRR / Recall@10             │
        │  vs analyst-labeled gold (8 stages)    │
        └────────────────────────────────────────┘
```

---

## Quick start

### 1. Install

```bash
cd /home/kaiyul3/ASAT/q3_reranker
pip install -e .                    # core deps (BM25, sentence-transformers, etc.)
pip install -e ".[openai]"          # add if using the DeepSeek backend
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — fill in at least one of:
#   DEEPSEEK_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   SERPAPI_KEY=...                 (only needed for SerpAPI retrieval)
#   SEMANTIC_SCHOLAR_API_KEY=...    (optional; bumps SS rate limit ~100x)
```

### 3. Run the evaluation (cached — free after first run)

```bash
# Recommended: arXiv source + arXiv gold + full LLM (Run 5 — cached):
RETRIEVAL_SOURCE=arxiv python -m q3_reranker.eval \
    --gold data/gold/queries_arxiv.json \
    --out reports/q3_results_arxiv_llm.md

# Smoke run (no LLM calls, arXiv source):
RETRIEVAL_SOURCE=arxiv python -m q3_reranker.eval \
    --gold data/gold/queries_arxiv.json \
    --no-llm \
    --out reports/q3_results_arxiv.md

# SerpAPI source, all 8 stages (requires SERPAPI_KEY or cached data):
python -m q3_reranker.eval

# Single stage only:
python -m q3_reranker.eval --stage 3
```

### 4. Browser demo

```bash
python -m q3_reranker.web_demo --port 8080
# open http://127.0.0.1:8080
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `RERANKER_BACKEND` | no | `claude` | LLM backend: `claude` or `deepseek` |
| `RERANKER_MODEL` | no | backend default | Override the specific model name |
| `RETRIEVAL_SOURCE` | no | `arxiv` | `serpapi`, `semantic_scholar`, or `arxiv` |
| `ANTHROPIC_API_KEY` | if using Claude | — | Anthropic API key |
| `DEEPSEEK_API_KEY` | if using DeepSeek | — | DeepSeek API key |
| `SERPAPI_KEY` | if using SerpAPI | — | SerpAPI key |
| `SEMANTIC_SCHOLAR_API_KEY` | no | — | Boosts SS rate limit; required for citation-graph ranker |

All variables are loaded from `.env` automatically when using any module's `main()` entry point.

---

## Module reference

| Module | Phase | Purpose |
|---|---|---|
| `retriever.py` | 1 | Semantic Scholar `/graph/v1/paper/search` client. SHA256 disk cache under `data/cache/retrieve/`. Defines the `Paper` dataclass used throughout. |
| `serpapi_retriever.py` | 1 | SerpAPI Google Scholar client. Returns the same `Paper` shape. Cache under `data/cache/serpapi/`. |
| `arxiv_retriever.py` | 1 | arXiv Atom feed client (free, no API key). Returns full abstracts. `citation_count=0` always. Cache under `data/cache/arxiv/`. |
| `sources.py` | 1 | Dispatcher: reads `RETRIEVAL_SOURCE` and routes every `search()` call to the right backend. |
| `baselines.py` | 2 | Three fast baselines: `ss_default` (preserve retrieval order), `bm25` (Okapi BM25), `dense` (MiniLM-L6 cosine). Defines `RankedResult`. |
| `rrf_rerank.py` | 2 | Reciprocal Rank Fusion ensemble: fuses ranked lists from multiple rankers using `score(d) = Σ 1/(60 + rank)`. No model or API required. |
| `cross_encoder_rerank.py` | 2 | Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-12-v2`, ~33 MB). Full attention over `(query, title+abstract)` pairs; faster than LLM, no API key. |
| `llm_rerank.py` | 3 | RankGPT-style sliding-window listwise reranker. Pluggable backends (`AnthropicBackend`, `DeepSeekBackend`). LLM response cache under `data/cache/llm/`. |
| `gold.py` | 4 | Loads gold-label JSON files (`data/gold/queries.json` by default; `queries_arxiv.json` for arXiv). Parses relevance grades 0–3. Fuzzy title normalisation for matching retrieved papers against hand-labeled gold. |
| `eval.py` | 4 | Orchestrates all rankers over all gold stages. Computes NDCG@10, MRR, Recall@10. Writes `reports/q3_results.md` by default. |
| `citation_rerank.py` | 6 | Novel angle. `citation_rerank`: in-set graph via SS batch API (requires `semantic_scholar` source). `citation_rerank_global`: log1p-normalised global citation count fallback for SerpAPI. |
| `query_expand.py` | 7 | Smart query expander. Classifies a user string as natural-language vs. keyword query and generates the counterpart via the LLM backend. Graceful fallback to raw text on any failure. |
| `demo.py` | 5 | CLI: column-aligned side-by-side top-K comparison table. `--text` runs the smart expander with an interactive review prompt. |
| `web_demo.py` | 6 | Browser-based demo. Pure Python stdlib HTTP server (`ThreadingHTTPServer`). Two-phase response: fast rankers sync, LLM rankers via background thread + client polling. Custom search box with LLM expansion + review panel. |

### Data flow

```
search(retrieval_query, limit=30)
  └─ reads from disk if cached, else calls API
  └─ list[Paper]  →  data/cache/retrieve/  or  data/cache/serpapi/

ss_default(papers)              # preserve retrieval order, O(n)
bm25(nl_query, papers)          # tokenise, index, BM25 score, O(n log n)
dense(nl_query, papers)         # MiniLM encode, cosine sim, O(n)

rrf({"ss_default": ..., "bm25": ..., "dense": ...})
  └─ score(d) = Σ_r 1 / (60 + rank_r(d))   [Cormack et al. 2009]
  └─ documents absent from a ranker get rank n+1 (last place)
  └─ no model or API call required

cross_encoder_rerank(nl_query, papers)
  └─ loads cross-encoder/ms-marco-MiniLM-L-12-v2 once per process
  └─ builds (query, title + abstract[:1000]) pairs
  └─ model.predict(pairs) → relevance logits
  └─ sorted descending; ~0.5 s for 30 papers on CPU

llm_rerank(nl_query, papers)
  └─ sliding window bottom→top: window=20, stride=10
  └─ each window → LLM call (cached in data/cache/llm/)
  └─ LLM returns JSON array of ranked integer IDs
  └─ re-stitches windows into global order

citation_rerank(llm_results, papers)
  └─ fetch one-hop references via SS /paper/batch (cached in data/cache/refs/)
  └─ build in-set directed graph: edge j→i if paper j cites paper i
  └─ authority(i) = in-degree within top-K subgraph
  └─ final_score = 0.7 * norm(llm_score) + 0.3 * norm(authority)
```

---

## CLI commands

### Retrieve papers

```bash
python -m q3_reranker.retriever "monocular depth estimation SOTA"
```

### Fast baselines (no LLM)

```bash
python -m q3_reranker.baselines "monocular depth estimation SOTA" --top-k 10
```

### LLM reranker only

```bash
python -m q3_reranker.llm_rerank "agentic retrieval augmented generation" --limit 20
```

### Side-by-side CLI demo

```bash
python -m q3_reranker.demo \
    --query "agentic retrieval augmented generation scientific QA" \
    --nl-query "What are the key papers on agentic RAG for science?" \
    --top-k 5
```

`--query` is the keyword form sent to the retrieval backend. `--nl-query` is the natural-language form passed to the LLM reranker (defaults to `--query` when omitted).

Or pass a **single** string in either form with `--text` and let the LLM expander
classify it and generate the counterpart. Both forms are printed and you get a
`[y]es / [e]dit / [n]o` prompt before retrieval runs; `--yes` skips the prompt.

```bash
python -m q3_reranker.demo \
    --text "what papers established self-supervised monocular depth?" \
    --top-k 5
```

### Full evaluation

```bash
python -m q3_reranker.eval [--gold PATH] [--no-llm] [--no-citation] [--no-cross] [--stage N] [--limit 30]
```

`--gold` selects the label file. Use `data/gold/queries_arxiv.json` when `RETRIEVAL_SOURCE=arxiv`.

`--no-cross` skips downloading and running the cross-encoder model (saves ~33 MB on first run and ~0.5 s per stage).

### Makefile shortcuts

```bash
make setup                                      # pip install -e .
make retrieve Q="protein structure prediction"  # fetch + cache
make baselines Q="protein structure prediction" # run fast rankers
make demo Q="protein structure prediction"      # side-by-side table
make eval                                       # full eval (all stages, all rankers)
make eval EVAL_ARGS="--no-llm"                  # fast smoke run, no LLM
make eval-stage1                                # run stage 1 only
make clean                                      # wipe data/cache
```

---

## Web demo

```bash
python -m q3_reranker.web_demo [--port 8080] [--no-llm] [--no-citation]
# open http://127.0.0.1:8080
```

The UI shows **8 preset chips** (one per gold benchmark stage). Clicking a chip triggers a two-phase response:

1. **Phase 1 — instant:** BM25, dense, and `ss_default` return from disk cache in under 1 second. The comparison table renders immediately.
2. **Phase 2 — background (~20 s):** LLM reranker and citation blend run in a daemon thread. The browser polls `GET /api/llm/{job_id}` every 2.5 s and fills in the LLM columns in place — no page reload.

The table uses a **papers-as-rows, rankers-as-columns** layout. Papers are the union of all rankers' top-10, sorted by `ss_default` rank. Rank badges: green = 1, amber = 2–3, blue = 4–10, grey dash = not in top-10.

### Custom search

Above the preset chips, a free-text search box accepts any query (≤ 300 chars) in **either** form — a natural-language question or a keyword query. Submitting it:

1. Calls `POST /api/expand_query`, which uses the LLM expander to classify the input and generate the missing counterpart.
2. Shows a **review panel** with both forms in editable fields — correct either, then click **Run search**.
3. On approval, the keyword form drives retrieval and the natural-language form is the reranker prompt (the same two-query design the gold stages use).

Preset chips spend no credits (all cached). A novel custom query calls the live source — free on arXiv, one SerpAPI credit per new query — and is cached on disk by SHA256 so repeats are instant.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Single-page app HTML |
| `GET` | `/api/presets` | JSON array of gold stages |
| `POST` | `/api/expand_query` | `{raw}` → `{kind, retrieval_query, nl_query}` |
| `POST` | `/api/search` | `{preset_idx}`, `{query}`, or `{retrieval_query, query}` → fast rankings + `job_id` |
| `GET` | `/api/llm/{job_id}` | `{status: pending\|done\|error, rankings}` |

---

## Evaluation results

### Run 5 — arXiv source, full LLM, arXiv gold (2026-05-18) — **current best**

Cross-stage mean over all 8 stages (`RETRIEVAL_SOURCE=arxiv`, `--limit 30`,
gold: `data/gold/queries_arxiv.json`):

| Ranker | NDCG@10 | MRR | Recall@10 |
|---|---|---|---|
| **llm_rerank** | **0.850** | **1.000** | **0.555** |
| dense | 0.737 | 1.000 | 0.544 |
| cross_encoder | 0.737 | 1.000 | 0.477 |
| rrf(default+bm25+dense) | 0.667 | 0.938 | 0.495 |
| bm25 | 0.580 | 0.938 | 0.419 |
| ss_default | 0.451 | 0.792 | 0.343 |

**Key finding:** `llm_rerank` leads by 11 NDCG points. The improvement over prior LLM runs stems from fixing `DEFAULT_MAX_TOKENS` from 1024 → 8192 (DeepSeek reasoning models were emitting empty content after an internal chain-of-thought that consumed the budget), and raising `LLMRerankError` on empty parse instead of silently returning source order. Stage 3 (agentic RAG) = **0.987**; Stage 9 (RAG foundations) = **0.816** vs dense 0.509 — the largest single-stage gap.

### Run 4 — arXiv source, no-LLM, arXiv gold (2026-05-18)

| Ranker | NDCG@10 | MRR | Recall@10 |
|---|---|---|---|
| **dense** | **0.737** | 1.000 | 0.544 |
| cross_encoder | 0.737 | 1.000 | 0.477 |
| rrf(default+bm25+dense) | 0.667 | 0.938 | 0.495 |
| bm25 | 0.580 | 0.938 | 0.419 |
| ss_default | 0.451 | 0.792 | 0.343 |

The arXiv gold fix (source-matched labels, 30 per stage) corrects the all-zero
Run 3 scores which were a gold-mismatch artifact, not poor retrieval.

### Run 2 — SerpAPI source, 5 rankers (2026-05-15)

Cross-stage mean (`--limit 30`, SerpAPI, `data/gold/queries.json`):

| Ranker | NDCG@10 | MRR | Recall@10 |
|---|---|---|---|
| **ss_default** | **0.601** | **0.818** | **0.690** |
| llm+citation(global) | 0.582 | 0.621 | 0.661 |
| llm_rerank | 0.508 | 0.509 | 0.624 |
| dense | 0.358 | 0.576 | 0.395 |
| bm25 | 0.267 | 0.333 | 0.455 |

`ss_default` leads here because Stages 6–9 had only 2–4 gold papers in Scholar's
top-30 (retrieval coverage bottleneck, not a reranking result). The `llm_rerank`
score of 0.508 also reflects the now-fixed silent-failure bug. Full per-stage
breakdown: `reports/q3_results.md`. Experimental record: `reports/experiments_log.md`.

---

## Gold benchmark stages

| Stage | Topic | Gold papers |
|---|---|---|
| 1 | LLM-based reranking | 7 |
| 2 | Scientific document embeddings and retrieval | 7 |
| 3 | Agentic RAG / autonomous research agents | 3 |
| 5 | Literature review generation with citation grounding | 5 |
| 6 | Monocular depth estimation | 12 |
| 7 | Diffusion models for image generation | 16 |
| 8 | Protein structure prediction | 10 |
| 9 | Retrieval-augmented generation foundations | 13 |

Stages 1–5 are derived from Q1 analyst labels. Stages 6–9 were hand-encoded for this project to test the rankers on vision and biology topics where Asta's retriever is regularly exercised.

---

## Status

- [x] Phase 0 — scaffold (pyproject.toml, .env.example, Makefile)
- [x] Phase 1 — cached retrieval via Semantic Scholar and SerpAPI
- [x] Phase 2 — BM25 / dense / ss_default baselines
- [x] Phase 3 — LLM listwise reranker (RankGPT sliding-window; Claude + DeepSeek)
- [x] Phase 4 — eval harness (NDCG@10 / MRR / Recall@10 against analyst gold)
- [x] Phase 5 — CLI side-by-side demo
- [x] Phase 6 — citation-aware reranker (in-set graph for SS; global-citation fallback for SerpAPI)
- [x] Phase 6 — browser-based web demo (two-phase: fast sync + LLM background poll)
- [x] Phase 7 — 8-stage benchmark expansion + full evaluation report
- [x] Phase 8 — RRF ensemble (`rrf_rerank.py`): fuses ss_default + bm25 + dense via Reciprocal Rank Fusion
- [x] Phase 8 — cross-encoder reranker (`cross_encoder_rerank.py`): full-attention scoring, no API key required
- [x] Phase 8 — arXiv retrieval source (`arxiv_retriever.py`): free, full abstracts, no API key
- [x] Phase 9 — arXiv gold set (`data/gold/queries_arxiv.json`): 30 source-matched labels per stage; `--gold PATH` eval flag; `--path` gold validator flag
- [x] Phase 9 — LLM silent-failure fix: `max_tokens=8192`, `LLMRerankError` on empty parse, deterministic integer-scan fallback; `llm_rerank` now leads at NDCG@10 = 0.850
- [x] Phase 9 — web demo: arXiv 429 resilience, single `render()` SPA rewrite (fixes empty LLM column bug), per-column sort + compare mode, paper hyperlinks, dynamic source label

## Context files

| File | Description |
|---|---|
| `/home/kaiyul3/ASAT/comparison_report.md` | Q1 Asta evaluation; source of gold labels for Stages 1–5 |
| `/home/kaiyul3/ASAT/observation.tex` | Q2 deep-dive into how Asta's paper finder works internally |
| `/home/kaiyul3/ASAT/q3.tex` | Full Q3 written report (LaTeX) |
| `reports/experiments_log.md` | Experimental record: gold construction, run parameters, findings (all 5 runs) |
| `reports/q3_results.md` | SerpAPI eval table (Run 2) |
| `reports/q3_results_arxiv.md` | arXiv no-LLM eval table (Run 4) |
| `reports/q3_results_arxiv_llm.md` | arXiv full LLM eval table (Run 5) — current best |
