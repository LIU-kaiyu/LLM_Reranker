# q3_reranker — LLM listwise reranking + citation-aware reranking

Demo for Q3 of the ASAT first task: pick one key step from Asta Find Papers'
workflow, build a SOTA solution, evaluate, and propose a novel improvement.

Step chosen: **LLM-based listwise reranking** of papers retrieved from an
academic-search source, plus a **citation-aware reranker** that can exploit
one-hop citation edges among the candidate set when Semantic Scholar reference
lists are available, or fall back to global citation-count authority when using
SerpAPI / Google Scholar.

## Layout

```
q3_reranker/
├── pyproject.toml
├── Makefile                # make setup / retrieve / baselines / demo / eval
├── .env.example            # API keys for the LLM backend
├── q3_reranker/
│   ├── __init__.py
│   ├── retriever.py        # Semantic Scholar /paper/search client + cache (Phase 1)
│   ├── serpapi_retriever.py# SerpAPI / Google Scholar client + cache       (Phase 1)
│   ├── sources.py          # retrieval-source dispatcher                    (Phase 1)
│   ├── baselines.py        # BM25, dense (MiniLM/SPECTER2), source-default  (Phase 2)
│   ├── llm_rerank.py       # RankGPT sliding-window via Claude/DeepSeek     (Phase 3)
│   ├── citation_rerank.py  # citation graph / global-citation re-scoring    (Phase 6)
│   ├── eval.py             # NDCG@10 / MRR / Recall@10                       (Phase 4)
│   ├── gold.py             # parses ../comparison_report.md → gold labels   (Phase 4)
│   └── demo.py             # side-by-side top-10 CLI                        (Phase 5)
├── data/cache/             # gitignored: SS payloads + LLM responses
└── reports/                # generated eval tables
```

## Quick start

```bash
make setup                      # pip install -e .
make retrieve Q="monocular depth estimation SOTA"
make baselines Q="monocular depth estimation SOTA"
make eval EVAL_ARGS="--no-llm"   # cheap smoke eval; omit flag for full run
```

The first two phases work without any LLM API key. Phase 3+ adds an LLM
reranker — pick a backend in `.env` (`claude` / `deepseek`). The default
retrieval source is `serpapi`; set `RETRIEVAL_SOURCE=semantic_scholar` to use
the free Semantic Scholar search endpoint and enable true in-set citation-graph
reranking.

## Status

- [x] Phase 0 — scaffold
- [x] Phase 1 — cached retrieval via Semantic Scholar or SerpAPI / Google Scholar
- [x] Phase 2 — BM25 / dense / source-default baselines
- [x] Phase 3 — LLM listwise reranker (sliding-window, RankGPT-style; Claude + DeepSeek)
- [x] Phase 4 — eval harness (NDCG / MRR / Recall on labels from `comparison_report.md`)
- [x] Phase 5 — demo CLI (side-by-side top-10)
- [x] Phase 6 — citation-aware reranker (in-set graph for Semantic Scholar, global-citation fallback for SerpAPI)
- [x] Phase 7 — generated Q3 results report in `reports/q3_results.md`

## Context

See `/home/kaiyul3/ASAT/comparison_report.md` for the Q1 evaluation that
seeded this work, and `/home/kaiyul3/ASAT/observation.tex` for the Q2
deep-dive into how Asta's paper finder works under the hood.
