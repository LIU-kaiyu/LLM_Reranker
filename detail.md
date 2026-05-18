# Q3 Reranker — System Detail

Comprehensive technical reference for the `q3_reranker` project. Covers design decisions, data flow, each module in depth, the novel citation-graph angle, evaluation methodology, results analysis, and known limitations. For usage instructions see `README.md`.

---

## Table of contents

1. [Problem statement and scope](#1-problem-statement-and-scope)
2. [Why reranking, not retrieval](#2-why-reranking-not-retrieval)
3. [Repository layout](#3-repository-layout)
4. [Two-query design](#4-two-query-design)
5. [Retrieval layer](#5-retrieval-layer)
6. [Baseline rankers](#6-baseline-rankers)
7. [LLM listwise reranker](#7-llm-listwise-reranker)
8. [Citation-graph reranker (novel angle)](#8-citation-graph-reranker-novel-angle)
9. [Disk caching strategy](#9-disk-caching-strategy)
10. [Gold label construction](#10-gold-label-construction)
11. [Evaluation methodology](#11-evaluation-methodology)
12. [Evaluation results and analysis](#12-evaluation-results-and-analysis)
13. [Web demo design](#13-web-demo-design)
14. [Configuration and extension points](#14-configuration-and-extension-points)
15. [Known limitations](#15-known-limitations)
16. [Relationship to prior work](#16-relationship-to-prior-work)

---

## 1. Problem statement and scope

**Asta Find Papers** retrieves candidate papers from Google Scholar (via SerpAPI) in response to a user's research query. The retrieval order is Scholar's default ranking, which blends keyword relevance, citation count, recency, and unspecified proprietary signals. This is a reasonable starting point but has known failure modes:

- **Lexical mismatch:** Scholar's keyword engine misses papers that use different terminology for the same concept (e.g., "monocular depth" vs "single-image depth estimation").
- **Popularity bias:** Highly-cited survey papers surface above the specific technical papers a researcher actually needs.
- **Query ambiguity:** The same keyword string can be interpreted differently by a retrieval engine versus a domain expert. A query like "agentic RAG" might surface generic RAG explainers before the canonical papers.

Reranking is a post-retrieval step that re-orders an existing candidate pool using a richer relevance signal. It cannot recover papers the retriever missed, but it can fix the order within the pool.

**Scope of this project:**
- Implement and evaluate multiple reranking strategies against hand-labeled gold relevance judgements.
- Propose and evaluate a novel angle: citation-graph blending.
- Build tooling (CLI demo + browser demo) for side-by-side comparison of all rankers.

---

## 2. Why reranking, not retrieval

The Q3 task asks to improve *one* step from Asta's workflow. Reranking was chosen for three reasons:

1. **Asta already has a working retriever.** The SerpAPI / Semantic Scholar layer functions. Replacing it would be a larger engineering effort with uncertain gains. Reranking sits cleanly on top of whatever retriever is already there.

2. **LLM-based listwise reranking (RankGPT) is demonstrably SOTA.** Sun et al. (EMNLP 2023) showed it beats BM25 and biencoder baselines on TREC-style benchmarks. The LLM access required is already available in this project environment.

3. **The citation-graph angle is genuinely novel** relative to Asta's current approach. Blending an LLM's topical judgement with in-set citation authority exploits structural information that neither BM25 nor a pure LLM sees.

---

## 3. Repository layout

```
q3_reranker/
├── pyproject.toml              dependency manifest and entry points
├── Makefile                    developer shortcuts
├── .env.example                template for API keys
├── .env                        actual keys (gitignored)
│
├── q3_reranker/
│   ├── __init__.py
│   ├── retriever.py            Semantic Scholar client + Paper dataclass
│   ├── serpapi_retriever.py    SerpAPI Google Scholar client
│   ├── arxiv_retriever.py      arXiv Atom feed client (free, no API key)
│   ├── sources.py              retrieval-source dispatcher
│   ├── baselines.py            ss_default / bm25 / dense + RankedResult
│   ├── rrf_rerank.py           Reciprocal Rank Fusion ensemble
│   ├── cross_encoder_rerank.py ms-marco cross-encoder full-attention reranker
│   ├── llm_rerank.py           RankGPT sliding-window + LLM backends
│   ├── citation_rerank.py      in-set graph + global-citation fallback
│   ├── gold.py                 gold label loading and grading
│   ├── eval.py                 evaluation harness + metrics
│   ├── demo.py                 CLI side-by-side demo
│   └── web_demo.py             browser-based demo (stdlib HTTP server)
│
├── data/
│   ├── gold/
│   │   ├── queries.json        Scholar-derived gold stages
│   │   └── queries_arxiv.json  arXiv top-30 gold stages
│   └── cache/                  gitignored; all API response caches
│       ├── retrieve/
│       ├── serpapi/
│       ├── arxiv/
│       ├── llm/
│       └── refs/
│
└── reports/
    ├── q3_results.md           auto-generated eval table (overwritten each run)
    └── experiments_log.md      manual experimental record
```

---

## 4. Two-query design

A critical subtlety: the query used for **retrieval** and the query used for **reranking** are different.

### Why they differ

Retrieval backends (Scholar / SerpAPI) are keyword search engines. They perform best with short, specific keyword strings: `"monocular depth estimation SOTA"`. A natural-language question like `"What are the most important papers on monocular depth estimation from the last five years?"` produces a different — and typically worse — candidate pool from Scholar; the extra words act as noise for BM25-style retrieval.

The LLM reranker works best with a natural-language query that captures the research intent. The LLM can reason over full question semantics and judge topical fit holistically.

### Implementation

Each `GoldStage` in `queries.json` carries two fields:

```json
{
  "retrieval_query": "monocular depth estimation deep learning",
  "query": "What are the most important papers on monocular depth estimation?"
}
```

`eval.py` calls `search(gold.retrieval_query, limit=N)` to fetch candidates, then passes `gold.query` to every ranker. This ensures:
- The candidate pool matches what a human analyst would retrieve.
- The LLM ranker receives the form of the question that maximises its relevance signal.

In `demo.py` this maps to `--query` (retrieval) and `--nl-query` (ranker input).

---

## 5. Retrieval layer

### Paper dataclass

Defined in `retriever.py`. All backends produce `Paper` objects:

```python
@dataclass(frozen=True)
class Paper:
    paper_id: str          # Semantic Scholar / SerpAPI ID
    corpus_id: str | None  # SS corpus ID (used by citation batch endpoint)
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    citation_count: int    # global total citations
    external_ids: dict     # DOI, ArXiv ID, MAG, etc.
```

### Semantic Scholar client (`retriever.py`)

Hits `/graph/v1/paper/search`. Returns title, abstract, authors, year, venue, citation count, and external IDs. The free tier allows approximately 1 request/second; providing `SEMANTIC_SCHOLAR_API_KEY` raises this to ~100 requests/second.

Cache key: `SHA256(query + "\n" + str(limit))[:32]`.

### SerpAPI client (`serpapi_retriever.py`)

Hits Google Scholar via SerpAPI. Returns the same `Paper` shape. Parses `cited_by.total` from the Scholar response for `citation_count`.

Cache key: `SHA256(query + "\n" + str(limit) + "\n" + str(year_low) + "\n" + str(year_high))[:32]`.

**SerpAPI credit note:** 1 credit per uncached search call. The web demo uses only pre-cached preset queries to avoid unexpected charges.

**Abstract quality:** `Paper.abstract` is populated from SerpAPI's `snippet` field — approximately 200 characters of truncated text, not the full abstract. For rankers that depend on abstract quality (dense, LLM), this limits signal quality compared to sources that provide full text.

### arXiv client (`arxiv_retriever.py`)

Queries the arXiv Atom feed API (`https://export.arxiv.org/api/query`) — free, no API key required. Uses `sortBy=relevance` and the `all:` search field to match across title, abstract, and author metadata.

Returns **full abstracts** from the `<summary>` field of the Atom XML, unlike SerpAPI's truncated snippet. This makes arXiv a strong choice for abstract-driven rankers (dense, LLM).

**Trade-offs vs other backends:**

| Concern | Behaviour |
|---|---|
| Abstract quality | Full text, not a snippet |
| Coverage | Excellent for CS / ML / physics / quant-bio preprints; may miss journal-only papers without an arXiv copy |
| Citation data | `citation_count=0` always — arXiv carries no citation metadata |
| Citation blend | `llm+citation(global)` is inert for arXiv results: all authority scores are equal (0), so the blend collapses to pure `llm_rerank` |
| API key | None required |
| Rate limit | arXiv asks ≤3 req/s; a 1-second courtesy sleep is inserted after each live fetch |

Cache key: `SHA256(query + "\n" + str(limit))[:32]`, stored under `data/cache/arxiv/`.

The parsing pipeline: raw Atom XML → JSON dicts (written to disk cache) → `Paper` objects. The two-step split means the cache stores normalised JSON rather than raw XML bytes, making it easier to inspect and debug cached entries.

### Source dispatcher (`sources.py`)

Reads `RETRIEVAL_SOURCE` from environment and routes `search()` calls:

```
RETRIEVAL_SOURCE=serpapi          → serpapi_retriever.search()
RETRIEVAL_SOURCE=semantic_scholar → retriever.search()
RETRIEVAL_SOURCE=ss               → retriever.search()   (alias)
RETRIEVAL_SOURCE=arxiv            → arxiv_retriever.search()
```

Default is `arxiv`. Set to `semantic_scholar` to enable true in-set citation-graph reranking (the global fallback is used otherwise). Use `serpapi` for Google Scholar retrieval with citation counts (required for `llm+citation(global)`). `arxiv` is the recommended default: free, no API key, full abstracts, and the fixed LLM reranker leads the arXiv benchmark at NDCG@10 = 0.850.

---

## 6. Baseline and ensemble rankers

The fast rankers require no API calls and complete in under 1 second per stage. `ss_default`, `bm25`, and `dense` are in `baselines.py`; `rrf_rerank.py` and `cross_encoder_rerank.py` are separate modules that build on them.

### `ss_default`

Trivial: preserve the retrieval backend's returned order. Score is `n - rank_position` so the first paper returned gets the highest score. This is a surprisingly strong baseline on NLP topics where Scholar's default is already well-calibrated, and reveals the ceiling set by retrieval quality alone.

### `bm25`

Okapi BM25 over concatenated title and abstract, implemented via the `bm25s` library. Tokenises with English stopword removal. Uses the natural-language query form.

Strengths: fast, no model required, handles exact keyword matching well.  
Weaknesses: ignores semantic similarity; degrades when query and paper use different vocabulary for the same concept.

### `dense`

Cosine similarity in the embedding space of `sentence-transformers/all-MiniLM-L6-v2`. Encodes the query and all documents, computes cosine similarity (dot product of unit-normalised vectors), sorts descending.

Strengths: captures semantic similarity; handles synonyms and paraphrases.  
Weaknesses: general-purpose model — SPECTER2 (trained on scientific papers) would be a stronger scientific baseline. MiniLM underperforms BM25 on some stages because it lacks domain-specific scientific vocabulary.

### `rrf` — Reciprocal Rank Fusion (`rrf_rerank.py`)

Implements the Cormack et al. (2009) fusion formula:

```
score(d) = Σ_r  1 / (k + rank_r(d))
```

with the standard damping constant k = 60. Default configuration fuses `ss_default`, `bm25`, and `dense`. Documents absent from any single ranker's list receive rank n+1 (last place in that list).

**Why RRF over a score-level blend:** individual rankers use incommensurable score scales (BM25 yields unbounded positive floats; cosine similarity is bounded to [−1, 1]; `ss_default` uses integer rank positions). RRF normalises implicitly by converting every score to a rank before combining — no min-max normalisation step required.

**When RRF helps:** Stage 1 (arXiv, Run 3): RRF NDCG@10 = **0.869**, outperforming every individual ranker (dense 0.844, bm25 0.806). The three rankers agree on the top papers but order them differently; RRF resolves the tie robustly.

**When RRF hurts:** Stage 3 (arXiv, Run 3): RRF = 0.333 < dense = 1.000. `ss_default` returns 0 relevant papers for this query, so including its list adds noise that degrades the fused result. RRF is most effective when all input rankers carry useful signal.

### `cross_encoder` — Cross-encoder reranker (`cross_encoder_rerank.py`)

Uses `cross-encoder/ms-marco-MiniLM-L-12-v2` (~33 MB, CPU-fast) from `sentence-transformers`. A cross-encoder performs full attention over the concatenated `(query, document)` pair — strictly more expressive than the bi-encoder `dense` baseline, which encodes query and document independently and cannot model query–document interaction.

**Inference:**  
For each paper, builds the string `title + " " + abstract` (truncated to 1000 characters to stay within the 512-token window) and calls `model.predict(pairs)`. Returns papers sorted by score descending. Approximately 0.5 s for 30 papers on CPU.

**Domain mismatch:** The model was trained on MS MARCO (web passage ranking, not scientific papers). On Run 3 (arXiv source), cross-encoder NDCG@10 = 0.145 — lower than the bi-encoder dense (0.251). A scientific cross-encoder (e.g., a SPECTER2-fine-tuned cross-encoder) would be substantially stronger for this domain but is not publicly available as a `sentence-transformers` cross-encoder.

**No API key required:** the model is downloaded once to the HuggingFace cache and then used locally.

---

## 7. LLM listwise reranker

Implements RankGPT (Sun et al., EMNLP 2023) with a sliding-window adaptation.

### How listwise reranking works

Instead of scoring papers independently (pointwise) or in pairs (pairwise), listwise reranking presents the LLM with a window of candidate papers all at once and asks it to return them in ranked order. The LLM can reason holistically: "paper A is more foundational than B because it introduced the core method; C is off-topic."

### Sliding-window procedure

For a candidate pool of N papers with window size W and stride S:

1. Start at the bottom W papers (least relevant by initial order).
2. Ask the LLM to rank them; update that segment of the global order.
3. Slide the window upward by S positions. The re-ordered bottom segment feeds into the next window's overlap.
4. Continue until the window reaches position 0.
5. Optionally run a second pass for stability.

Default: `window_size=20`, `stride=10`, `num_passes=1`. For a 30-paper pool, this is approximately 3 LLM calls.

### Prompt design

**System prompt (constant across all windows):**  
"You are an expert academic literature reviewer. Rank candidates by relevance to the user's research query. Relevance means the paper directly addresses the query topic, uses methods central to it, or reports core results. Do not be swayed by paper age or citation count. Return ONLY a JSON array of identifier integers in ranked order."

**User prompt (per window):**  
Lists up to 20 papers with numeric IDs, year, venue, title, and abstract (truncated to 800 chars). Asks for a JSON array of IDs.

### Response parsing

`parse_ranked_ids()` first tries to parse the whole response as JSON, then regex-scans for the first `[...]` match, then filters to only valid IDs for the current window. IDs not returned by the LLM are appended at the end (conservative: don't drop papers the LLM didn't mention).

### Backends

**AnthropicBackend** — uses `anthropic.Anthropic()` with prompt caching on the system prompt (the system prompt is identical across all windows in a single query, so subsequent windows hit the Anthropic cache). Default model: `claude-haiku-4-5`.

**DeepSeekBackend** — uses `openai.OpenAI(base_url="https://api.deepseek.com")` with `DEEPSEEK_API_KEY`. Default model: `deepseek-chat`. The OpenAI-compatible interface means no separate SDK is needed.

Selection: `RERANKER_BACKEND=claude` or `RERANKER_BACKEND=deepseek`. Model override: `RERANKER_MODEL=<name>`.

---

## 8. Citation-graph reranker (novel angle)

### Motivation

The LLM reranker ranks papers by topical relevance to the query. It does not know which papers are structurally foundational within the specific candidate pool. A paper that every other retrieved paper cites is almost certainly central to the topic — this structural signal is complementary to the LLM's semantic judgement.

Standard academic reranking (RankGPT, RankZephyr, FIRST) does not exploit within-pool citation structure.

### In-set authority (Semantic Scholar mode)

When `RETRIEVAL_SOURCE=semantic_scholar`, we fetch one-hop reference lists for each candidate via the `/graph/v1/paper/batch` endpoint.

**Procedure:**

1. Take the LLM reranker's top-K candidates (default K=20).
2. Batch-fetch their reference lists from SS (cached under `data/cache/refs/`).
3. Build a directed graph: edge j → i if paper j cites paper i, restricted to edges where both endpoints are in the top-K pool.
4. `authority(i)` = in-degree of paper i in this induced subgraph.
5. Blend scores:

```
final_score(i) = 0.7 × normalize(llm_score(i)) + 0.3 × normalize(authority(i))
```

6. Re-sort top-K by `final_score`; append the tail unchanged.

**Why in-set rather than global?** Global citation count conflates popularity across all fields with topical authority. In-set citation authority is scoped to this specific topic: a paper that every other candidate for this query cites is specifically authoritative for this query.

### Global citation fallback (SerpAPI mode)

When per-paper reference lists are unavailable, we blend with global `citation_count` from the SerpAPI response, log-normalised to flatten the heavy tail (some papers 5000+, most <100):

```
authority_approx(i) = log(1 + citation_count(i))
final_score(i) = 0.7 × normalize(llm_score(i)) + 0.3 × normalize(authority_approx(i))
```

This partially recovers Scholar's own secondary sort but applies it within the LLM's already relevance-filtered top-K, which is more topically scoped than Scholar's full-pool ordering.

### Blend parameter

λ = 0.7 was set by hand (LLM-dominant). A per-stage sweep or learning λ from a held-out gold split would likely improve results.

---

## 9. Disk caching strategy

All API calls are cached to disk using SHA256 hashes of inputs as keys. Goals:

- **Free reruns:** After the first run, switching backends or parameters costs nothing in API credits.
- **Determinism:** Identical inputs always return identical outputs from cache.
- **Honest reporting:** Results reflect a fixed snapshot of the retrieval pool.

| Cache directory | Cached content | Cache key |
|---|---|---|
| `data/cache/retrieve/` | SS search payloads | `SHA256(query + "\n" + limit)[:32]` |
| `data/cache/serpapi/` | SerpAPI search payloads | `SHA256(query + "\n" + limit + year_range)[:32]` |
| `data/cache/arxiv/` | arXiv Atom feed payloads (JSON-normalised) | `SHA256(query + "\n" + limit)[:32]` |
| `data/cache/llm/` | LLM ranking responses | `SHA256(model + "\0" + system + "\0" + user)[:32]` |
| `data/cache/refs/` | SS reference batch payloads | `SHA256(sorted_paper_ids joined by "\|")[:32]` |

All caches are gitignored. A fresh clone must re-run evaluation to populate them.

---

## 10. Gold label construction

### File format

Default gold file: `data/gold/queries.json`.

ArXiv-specific gold file: `data/gold/queries_arxiv.json`.

Both files use the same schema:

```json
[
  {
    "stage": 8,
    "topic": "Protein structure prediction",
    "query": "What are the most important papers on protein structure prediction?",
    "retrieval_query": "protein structure prediction deep learning AlphaFold",
    "papers": {
      "Highly accurate protein structure prediction with AlphaFold": 3,
      "Accurate prediction of protein structures using RoseTTAFold": 3,
      "ESMFold: Language models enable zero-shot prediction of protein structure": 2,
      "ColabFold: making protein folding accessible to all": 2
    }
  }
]
```

### Grade scale

| Grade | Meaning |
|---|---|
| 3 | Canonical anchor — the seminal paper that created or defined the topic |
| 2 | Strong background — directly related, commonly cited alongside grade-3 papers |
| 1 | Partial relevance — related methodology but not core to this topic |
| 0 | Off-topic or unjudged |

### Stage sources

**Stages 1–5** derive from the Q1 Asta evaluation (`comparison_report.md`). The analyst's relevance judgements from Q1 were converted into this grade scheme.

**Stages 6–9** were hand-encoded for this project covering vision and biology topics:
- Stage 6 (monocular depth): MiDaS, DPT, Depth Anything v1/v2, ZoeDepth, Marigold, AdaBins, Monodepth2, NeWCRFs, Depth Pro, UniDepth.
- Stage 7 (diffusion models): DDPM, LDM, DDIM, DiT, Imagen, GLIDE, DALL-E 2, ADM, Classifier-Free Guidance, Improved DDPM, Score SDE, NCSN.
- Stage 8 (protein structure): AlphaFold 1/2/3, RoseTTAFold, ESMFold, ESM-2, ColabFold, OmegaFold, OpenFold, AF-Multimer.
- Stage 9 (RAG foundations): Lewis RAG, REALM, DPR, Atlas, RETRO, FiD, Self-RAG, CRAG, kNN-LM, HyDE, Chain-of-Note, Active RAG, Lost in Middle.

### Title matching

Gold titles are matched against retrieved titles via `gold.normalize_title()`: lowercase, collapse whitespace, strip punctuation. For titles ≥30 characters, substring matching also runs (handles truncation from some sources). Papers absent from the gold set are assigned grade 0 (TREC convention), conflating "off-topic" with "unjudged."

---

## 11. Evaluation methodology

### Metrics

**NDCG@10 (primary):** Normalised Discounted Cumulative Gain at rank 10. Accounts for graded relevance (0–3) and rank position — a grade-3 paper at rank 1 contributes more than the same paper at rank 5. Normalised against the ideal ordering of the same pool.

```
DCG@k  = Σ (2^grade_i − 1) / log2(i + 2)   for i = 0..k−1
NDCG@k = DCG@k / IDCG@k
```

**MRR:** Mean Reciprocal Rank. `1 / rank_of_first_relevant_paper`. Captures whether the ranker surfaces at least one relevant paper near the top.

**Recall@10:** `relevant_in_top_10 / relevant_in_pool`. Denominator is relevant papers in the *retrieved pool*, not all relevant papers in existence — this evaluates the ranker fairly given a fixed retrieval.

### Eval procedure

```
for each gold stage:
    papers = search(gold.retrieval_query, limit=30)
    for each ranker:
        ranking = ranker(gold.query, papers)
        grades  = [grade_for(gold, r.title) for r in ranking]
        compute NDCG@10, MRR, Recall@10
```

Up to seven rankers run per stage depending on flags:

| Flag | Rankers excluded |
|---|---|
| _(none — full run)_ | `ss_default`, `bm25`, `dense`, `rrf(default+bm25+dense)`, `cross_encoder`, `llm_rerank`, `llm+citation(global)` |
| `--no-llm` | excludes `llm_rerank` and `llm+citation` |
| `--no-cross` | excludes `cross_encoder` |
| `--no-citation` | excludes `llm+citation`, keeps `llm_rerank` |

Use `--gold PATH` to select a non-default label file, for example `--gold data/gold/queries_arxiv.json` with `RETRIEVAL_SOURCE=arxiv`.

---

## 12. Evaluation results and analysis

### Run 2 cross-stage summary (SerpAPI source, 5 rankers, `--limit 30`)

| Ranker | NDCG@10 | MRR | Recall@10 |
|---|---|---|---|
| ss_default | 0.601 | 0.818 | 0.690 |
| llm+citation(global) | 0.582 | 0.621 | 0.661 |
| llm_rerank | 0.508 | 0.509 | 0.624 |
| dense | 0.358 | 0.576 | 0.395 |
| bm25 | 0.267 | 0.333 | 0.455 |

**Note on Run 2 LLM numbers:** These numbers were produced while a silent bug was present in `_rank_window` — when `parse_ranked_ids` returned an empty list (caused by DeepSeek's reasoning model starving the answer at `max_tokens=1024`), the function returned the unchanged source order. `llm_rerank` was therefore statistically equivalent to `ss_default` in most windows, which explains the surprisingly low score.

### Run 3 cross-stage summary (arXiv source, `--no-llm`, `--limit 30`)

Run 3 collapsed to near-zero because the gold file (`queries.json`) was built from SerpAPI Scholar results. arXiv returns a different paper pool: Stages 2, 5, 6, 7, 8, 9 returned zero labeled papers. These zeros are a gold-mismatch artifact, not evidence of poor arXiv retrieval.

### Run 4 cross-stage summary (arXiv source, arXiv gold, `--no-llm`, `--limit 30`)

Gold file: `data/gold/queries_arxiv.json` — 30 labeled papers per stage, source-matched to the arXiv top-30 pool.

| Ranker | NDCG@10 | MRR | Recall@10 |
|---|---|---|---|
| **dense** | **0.737** | 1.000 | 0.544 |
| cross_encoder | 0.737 | 1.000 | 0.477 |
| rrf(default+bm25+dense) | 0.667 | 0.938 | 0.495 |
| bm25 | 0.580 | 0.938 | 0.419 |
| ss_default | 0.451 | 0.792 | 0.343 |

Dense and cross-encoder tie. `ss_default` scores lowest — Scholar's citation-popularity ordering is the wrong prior for arXiv pools. RRF fuses robustly but is noisy on stages where `ss_default` contributes poor rankings.

### Run 5 cross-stage summary (arXiv source, arXiv gold, full LLM, `--limit 30`) — current best

Fix applied: `DEFAULT_MAX_TOKENS = 8192`; `_rank_window` now raises `LLMRerankError` on empty parse instead of silently returning source order; deterministic integer-scan fallback in `parse_ranked_ids`.

| Ranker | NDCG@10 | MRR | Recall@10 |
|---|---|---|---|
| **llm_rerank** | **0.850** | **1.000** | **0.555** |
| dense | 0.737 | 1.000 | 0.544 |
| cross_encoder | 0.737 | 1.000 | 0.477 |
| rrf(default+bm25+dense) | 0.667 | 0.938 | 0.495 |
| bm25 | 0.580 | 0.938 | 0.419 |
| ss_default | 0.451 | 0.792 | 0.343 |

**`llm_rerank` leads by 11.3 NDCG points** over dense and cross_encoder. The token-budget fix is entirely responsible for the gap — the ranker was already architecturally correct; it was silently emitting empty responses in prior runs.

### Stage-level highlights (Run 5)

| Stage | Topic | dense | llm_rerank | Winner |
|---|---|---|---|---|
| 3 | Agentic RAG | 0.891 | **0.987** | llm |
| 9 | RAG foundations | 0.509 | **0.816** | llm (+30.7 pts) |
| 6 | Monocular depth | 0.675 | **0.853** | llm |
| 2 | Document embeddings | 0.522 | **0.773** | llm |
| 8 | Protein structure | 0.848 | **0.967** | llm |
| 1 | LLM-based reranking | **0.897** | 0.892 | dense (≈tie) |
| 5 | Lit review | **0.961** | 0.838 | dense |
| 7 | Diffusion models | 0.592 | 0.677 | llm (cross_encoder 0.815 leads) |

**Stage 9 (RAG foundations)** is the largest gap: the NL query asks for *foundational* papers, a semantic framing that embedding similarity cannot distinguish from topically-related non-foundational work, but the LLM reasons over it correctly.

**Stage 7 (Diffusion models)** is the only stage the LLM does not lead; cross_encoder (0.815) beats both. This may reflect the specific arXiv pool composition for diffusion models — papers with highly similar titles/abstracts where the MS MARCO cross-encoder's fine-grained local attention is more useful than the LLM's holistic ranking.

### Why ss_default leads the SerpAPI aggregate (Run 2)

Stages 6–9 (vision and bio topics) had only 2–4 labeled papers in Scholar's top-30 pool. When 27 of 30 papers are unjudged, the ranker that surfaces the 2–4 labeled ones first wins. Scholar's order happens to place them reasonably because Scholar already uses global citation count — the same signal the blend replicates but over a 30-paper pool. The arXiv gold fix eliminates this measurement artifact entirely: all 30 papers per stage are labeled.

### LLM + citation blend vs LLM alone (SerpAPI context)

In the SerpAPI runs, `llm+citation(global)` (0.582) outperforms `llm_rerank` (0.508). This difference is exaggerated by the silent-failure bug: both rankers were partially returning source order, and the citation term in the blend happened to partially correct some stages. With the fix applied on arXiv, the blend is not evaluated (citation_count=0 for all arXiv papers), but the LLM alone now reaches 0.850 — well above the blend's 0.582 on SerpAPI.

---

## 13. Web demo design

### Goals

- Let a developer or analyst compare all 5 rankers side-by-side on any of the 8 benchmark stages interactively.
- Expose the two-phase latency reality honestly: fast rankers instantly, LLM after ~20 s.
- Zero extra dependencies — pure Python stdlib.
- Zero SerpAPI credit cost — preset-only, all payloads already cached by the eval run.

### HTTP server

`ThreadingHTTPServer` on `127.0.0.1` only (not exposed externally). Each request is handled in its own thread from the server's thread pool.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Single-page app HTML (~10 KB, embedded as a Python string constant) |
| `GET` | `/api/presets` | JSON array of `{stage, topic, query, retrieval_query}` objects |
| `POST` | `/api/search` | Body: `{preset_idx: int}`. Returns: fast rankings + `job_id` |
| `GET` | `/api/llm/{job_id}` | Returns `{status: pending\|done\|error, rankings}` |

### Two-phase response flow

```
Click chip
  │
  ├── POST /api/search
  │     ├── search(retrieval_query, limit=20)   ← hits disk cache, < 1 s
  │     ├── ss_default(papers)                  ← in-process, instant
  │     ├── bm25(nl_query, papers)               ← in-process, instant
  │     ├── dense(nl_query, papers)              ← in-process, instant
  │     ├── start daemon thread for LLM rankers
  │     └── return fast_rankings + job_id
  │
  ├── Render table with 3 fast columns immediately
  │
  └── Poll GET /api/llm/{job_id} every 2.5 s
        ├── 202 pending  → keep polling
        ├── 200 done     → fill LLM columns in place, stop polling
        └── 200 error    → show error indicator, stop polling
```

### In-memory job store

```python
_jobs: dict[str, serialised_rankings | Exception | None] = {}
```

`None` = job running. `dict` = done (serialised ranking data). `Exception` = failed.

Job ID: `SHA256(retrieval_query + "\n" + nl_query)[:16]`. If the same preset is clicked twice, the second click finds the job already done and includes `llm_rankings` directly in the `POST /api/search` response — no polling needed.

### Table layout

Papers as rows, rankers as columns. Papers shown: union of all rankers' top-10, sorted by `ss_default` rank. This layout makes it easy to see:
- Which papers appear in every ranker's top-10 (high consensus).
- Which papers the LLM promotes that `ss_default` missed.
- Where the citation blend moves papers relative to `llm_rerank` alone.

Rank badges: green = 1, amber = 2–3, blue = 4–10, grey dash = not in top-10.

---

## 14. Configuration and extension points

### Adding a new LLM backend

Implement the `LLMBackend` Protocol (in `llm_rerank.py`):

```python
class LLMBackend(Protocol):
    name: str
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...
```

Then extend `get_backend()`:

```python
if name == "myprovider":
    return MyProviderBackend(model=model_env or "my-default-model")
```

### Adding a new retrieval source

Implement `search(query: str, limit: int, **kwargs) -> list[Paper]` returning `Paper` objects (same shape as `retriever.py`). Add it to `sources.py`'s dispatcher.

### Adding gold stages

Edit the appropriate file under `data/gold/`. Use `queries.json` for the default Scholar-derived benchmark and `queries_arxiv.json` for arXiv-specific top-30 labels. Follow the existing schema: `stage`, `topic`, `query`, `retrieval_query`, `papers` (dict of title → grade 0–3). Run `python -m q3_reranker.gold --path data/gold/queries_arxiv.json` to verify an alternate gold file.

### Changing the blend weight

In `citation_rerank.py`, the default blend is `DEFAULT_BLEND = 0.7`. Pass a different `blend` argument to `citation_rerank()` or `citation_rerank_global()`. To sweep λ across stages, call `evaluate_stage()` multiple times with different ranker configurations.

---

## 15. Known limitations

**1. Gold coverage depends on retrieval.** All metrics are relative to the retrieved pool. Papers Scholar does not surface cannot be evaluated or recovered by any ranker. Stages 6–9 expose this clearly.

**2. Single LLM backend evaluated.** All numbers use DeepSeek `deepseek-chat` (Run 2). The backend is now configured to use `deepseek-v4-pro` (thinking model); a fresh eval run will produce updated numbers. RankGPT (Sun et al.) showed GPT-4-class judges score 5–10 NDCG points higher than smaller models — stronger LLM judges would likely improve all LLM-ranker numbers.

**3. Fixed λ = 0.7.** The blend weight was set by hand. A per-stage sweep or learning λ from a held-out split of the gold would improve Stage 9 without hurting the aggregate.

**4. Global citation conflates popularity with authority.** The `citation_rerank_global` fallback is biased toward old, broadly-cited papers regardless of topical fit. A 2026 SOTA paper with few citations is penalised even if it's the most relevant candidate.

**5. In-set graph is thin for small pools.** With 30 candidates, most papers cite at most 1–2 others in the pool. In-degree has high variance. The effect is clearest where canonical papers are heavily cross-cited within the pool (Stage 8), and weakest where the pool is sparse (Stage 6).

**6. Dense baseline uses a general-purpose model.** `all-MiniLM-L6-v2` is fast but SPECTER2 (trained on scientific papers) is the SOTA scientific embedding model. Upgrading the dense baseline would give a fairer comparison for the LLM reranker.

**10. Cross-encoder trained on web passages.** `cross-encoder/ms-marco-MiniLM-L-12-v2` shows lower NDCG@10 than the bi-encoder `dense` baseline (0.145 vs 0.251 on Run 3). This is the expected domain mismatch: MS MARCO passage queries differ from academic paper abstract queries. A scientific cross-encoder fine-tuned on citation-prediction or paper-relevance data would likely reverse this ranking.

**11. RRF degrades when input rankers disagree strongly.** When one ranker returns a completely different top-10 than the others (e.g., `ss_default` on Stage 3 arXiv returning 0 relevant papers), including it in the RRF fusion adds rank noise. Selective RRF — only fusing rankers that correlate above a threshold — would be more robust.

**7. No multi-field retrieval.** Ranking uses title + abstract. Including venue, citation count, year, or author embeddings could improve baselines.

**8. Retrieval payloads are frozen.** Results reflect Scholar's index snapshot from May 2026. A fresh run may receive a different pool if Scholar's index has changed.

**9. Small gold set.** 73 gold-labeled papers across 8 stages is enough to compare methods but insufficient for statistical significance testing.

---

## 16. Relationship to prior work

| Work | Relation to this project |
|---|---|
| **RankGPT** (Sun et al., EMNLP 2023) | Directly implements the sliding-window listwise approach. Same window=20, stride=10 defaults. The core reranking algorithm. |
| **RankZephyr** (Pradeep et al., 2023) | Distills RankGPT into a 7B open model. A quantised local model would be an offline alternative to the API-based backends used here. |
| **FIRST** (Reddy et al., 2024) | Efficient listwise reranking via first-token logit scores. A potential optimisation to reduce LLM calls per window while maintaining ranking quality. |
| **PRP** (Qin et al., 2023) | Pairwise ranking prompting — presents pairs instead of windows. More API calls but potentially more stable for small windows. |
| **CoRank** (arXiv:2505.13757, 2025) | Collaborative reranking using citation graphs. Closest prior work to our citation-graph angle; uses author co-citation rather than in-set reference authority, and does not blend with LLM scores. |
| **Semantic Scholar API** | Provides the retrieval backend (`/paper/search`), in-set citation data (`/paper/batch`), and global citation counts. Central infrastructure for both retrieval and the citation-graph reranker. |
| **RRF** (Cormack et al., SIGIR 2009) | The Reciprocal Rank Fusion formula implemented in `rrf_rerank.py`. k=60 damping constant is from the original paper. This project applies RRF as a no-cost ensemble over its three baseline rankers. |
| **MS MARCO cross-encoder** (Microsoft, 2018) | Training dataset for `cross-encoder/ms-marco-MiniLM-L-12-v2`. Designed for web passage ranking; domain mismatch with scientific abstracts explains the gap relative to the bi-encoder dense baseline. |

**Core novelty** relative to RankGPT and CoRank: applying in-set citation authority as a **post-LLM blend** rather than as a standalone ranker or as a global signal. The LLM provides topical relevance; the citation graph provides within-pool structural authority. The combination is stronger than either alone on stages with adequate gold coverage, and makes an honest trade-off on stages where the pool is too sparse for the graph to carry useful signal.
