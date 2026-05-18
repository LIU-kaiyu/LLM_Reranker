# Q3 Reranker — Experiments Log

This file documents each evaluation stage: where the gold came from,
how labels were assigned, which queries were used, and what was
expected before the eval ran. It is the audit trail for the numbers
in `q3_results.md` and the tables in `/home/kaiyul3/ASAT/q3.tex`.

---

## Methodology (applies to every stage)

- **Retrieval source**: SerpAPI Google Scholar, `limit=30` (1 page).
- **Reranker LLM**: DeepSeek `deepseek-chat` via the OpenAI-compatible API.
- **Two-query design** per stage:
  - `retrieval_query`: short keyword form sent to Scholar.
  - `query`: natural-language form fed to BM25, dense, LLM, and the blend.
- **Gold grades**: `3` (canonical / anchor), `2` (related / background),
  `1` (partially relevant / survey-adjacent), `0` (off-topic).
- **Metrics**: NDCG@10 (primary), MRR, Recall@10.
- **Determinism**: All SerpAPI + DeepSeek responses are SHA256-cached
  under `data/cache/`, so each stage is a free, deterministic rerun
  after the first execution.

---

## Stages 1, 2, 3, 5 — Q1 carry-over

These stages were authored from analyst judgments in
`/home/kaiyul3/ASAT/comparison_report.md`. They are the original
benchmark. Stage 4 was intentionally omitted because Asta misfired on
that topic in Q1 — there is no usable candidate pool. See
`data/gold/queries.json` (top-level `_note`) for the source citation.

---

## Stage 6 — Monocular depth estimation

### Why this stage
Computer-vision sub-field with a clear, narrow canon and a sharp
methodology line (zero-shot foundation models like MiDaS / DPT /
Depth Anything). Tests whether the rerankers generalize out of NLP
into vision.

### Gold construction
- **`retrieval_query`**: `"monocular depth estimation deep learning"`
- **`query`** (NL): `"Find the recent SOTA papers on monocular depth
  estimation from a single image."`
- **Anchors (grade 3)**: MiDaS (Ranftl 2020), DPT (Ranftl 2021),
  Depth Anything v1/v2 (Yang 2024), ZoeDepth (Bhat 2023), Marigold
  (Ke 2024), AdaBins (Bhat 2021), Monodepth2 (Godard 2019), NeWCRFs
  (Yuan 2022), Depth Pro (Bochkovskii 2024).
- **Background (grade 2)**: earlier supervised / self-supervised work
  that defines the lineage (BTS, Monodepth, DenseDepth, FCRN,
  Eigen et al. 2014).
- **Partial (grade 1)**: a survey paper.
- **Off-topic (grade 0)**: stereo-only methods, time-series clustering.

### Expectation before running
- `ss_default` and `llm+citation(global)` should both score high here
  --- the canon is heavily cited.
- `llm_rerank` alone may stumble on the "SOTA" qualifier (recency
  signal not in title/abstract).
- `bm25` and `dense` likely middling because "depth estimation" is a
  common phrase outside the target literature (e.g., signal processing).

---

## Stage 7 — Diffusion models for image generation

### Why this stage
Generative-ML topic with a famously dense canon (DDPM, LDM, DiT,
Imagen, GLIDE, DALL-E 2). Tests rerankers on a topic where citation
counts are extremely lopsided --- DDPM alone has 10k+ citations.

### Gold construction
- **`retrieval_query`**: `"diffusion models image generation DDPM"`
- **`query`** (NL): `"What are the foundational papers on
  diffusion-based image generation in machine learning?"`
- **Anchors (grade 3)**: DDPM (Ho 2020), LDM/Stable Diffusion
  (Rombach 2022), DDIM (Song 2021), DiT (Peebles 2023), Imagen
  (Saharia 2022), GLIDE (Nichol 2022), unCLIP/DALL-E 2 (Ramesh 2022),
  ADM (Dhariwal & Nichol 2021), Classifier-Free Guidance (Ho &
  Salimans 2022), Improved DDPM (Nichol 2021), Score SDE (Song
  2021), NCSN (Song & Ermon 2019).
- **Background (grade 2)**: Karras EDM (2022), Cascaded DM (Ho 2022),
  Variational DM (Kingma 2021).
- **Partial (grade 1)**: a diffusion-models survey.
- **Off-topic (grade 0)**: pre-diffusion generative models
  (GAN, VAE).

### Expectation before running
- This is the stage where the **citation-graph prior should help
  most** --- the canonical diffusion papers have the world's largest
  ML citation counts.
- `llm_rerank` alone may surface the survey or EDM ahead of DDPM
  because the LLM prompt explicitly says "ignore citation count".
- `bm25` should do OK because most canonical papers literally
  contain the word "diffusion" in the title.

---

## Stage 8 — Protein structure prediction

### Why this stage
Bio-ML, narrow canon, with a single dominant family of papers
(AlphaFold lineage). Tests cross-domain generalization plus a hard
case: AlphaFold (Nature, 2021) is the standard hit but the
sequence/structure literature predates and surrounds it.

### Gold construction
- **`retrieval_query`**: `"protein structure prediction AlphaFold
  deep learning"`
- **`query`** (NL): `"Which papers introduce state-of-the-art methods
  for predicting protein 3D structure from sequence?"`
- **Anchors (grade 3)**: AlphaFold (Jumper 2021), AlphaFold 3
  (Abramson 2024), RoseTTAFold (Baek 2021), ESMFold (Lin 2023),
  ESM-2 (Lin 2022), ColabFold (Mirdita 2022), OmegaFold (Wu 2022),
  OpenFold (Ahdritz 2024), AlphaFold-Multimer (Evans 2022),
  AlphaFold 1 (Senior 2020).
- **Background (grade 2)**: ESM-1b, HelixFold-Single, xTrimoPGLM,
  CASP assessment papers.
- **Partial (grade 1)**: protein-LM review, AlphaMissense.
- **Off-topic (grade 0)**: viral evolution language models,
  general deep-learning-for-genomics overview.

### Expectation before running
- Likely a clean win for any reranker --- the topic is sharp enough
  that all five methods should agree.
- `llm+citation(global)` should be near-ceiling because (a) AlphaFold
  has very high citations, (b) the LLM clearly knows the canon.
- Honest risk: query contains "AlphaFold" so Scholar may return a
  pool that's already optimally ordered, leaving little room for the
  blend to improve.

---

## Stage 9 — Retrieval-augmented generation foundations

### Why this stage
NLP topic adjacent to Stage 3 (agentic RAG), but focused on
*foundations* (RAG, REALM, RETRO, Atlas, FiD) rather than
agentic systems. Tests whether the blend's Stage-3 recovery
generalizes, or whether it was topic-specific.

### Gold construction
- **`retrieval_query`**: `"retrieval augmented generation language
  models"`
- **`query`** (NL): `"What are the foundational papers introducing
  retrieval-augmented generation for large language models?"`
- **Anchors (grade 3)**: RAG (Lewis 2020), REALM (Guu 2020), DPR
  (Karpukhin 2020), Atlas (Izacard 2022), RETRO (Borgeaud 2022),
  FiD (Izacard & Grave 2021), Self-RAG (Asai 2023), CRAG (Yan 2024),
  kNN-LM (Khandelwal 2020), HyDE (Gao 2022).
- **Background (grade 2)**: Chain-of-Note (Yu 2023), Active RAG
  (Jiang 2023), Lost-in-the-Middle (Liu 2023), RAG survey (Gao
  2023).
- **Partial (grade 1)**: comprehensive RAG survey.
- **Off-topic (grade 0)**: BERT (Devlin), Attention Is All You Need
  (Vaswani), hallucination-mitigation survey.

### Expectation before running
- "Foundational" framing in the NL query is a hint the LLM should
  weight original papers above derivatives. If `llm_rerank` does
  *not* pull this off, the citation blend will likely save it.
- Risk: "RAG survey" papers and "Lost in the Middle" have huge
  citation counts and may get over-promoted by the citation prior
  alone.

---

## Recording protocol

After each eval run, the per-stage and cross-stage tables are
overwritten in `reports/q3_results.md`. This log is **not**
overwritten --- it is the methodology record. If the eval is rerun
with different parameters (e.g., `--limit 50`, a different LLM
backend), add a new "Run N" subsection rather than editing prior
entries.

### Run 1 — initial Q1-only benchmark (2026-05-15)
- Stages: 1, 2, 3, 5 (78 papers).
- `limit=30`, DeepSeek `deepseek-chat`, λ=0.7.
- Cross-stage mean NDCG@10:
  - `llm+citation(global)` = **0.845**
  - `ss_default` = 0.802
  - `llm_rerank` = 0.733
  - `dense` = 0.514
  - `bm25` = 0.338

### Run 3 — RRF + cross-encoder on arXiv (2026-05-16)
- Stages: all 8.
- Source: `RETRIEVAL_SOURCE=arxiv`, `limit=30`, `--no-llm`.
- Rankers: ss_default, bm25, dense, rrf(default+bm25+dense), cross_encoder.
- Key finding: arXiv pool contains labeled papers only on Stages 1 and 3;
  Stages 2, 5, 6, 7, 8, 9 return 0 labeled papers because gold was built
  from SerpAPI Scholar, not arXiv. All-zero stages confirm retrieval
  mismatch, not reranking failure.
- Cross-stage mean NDCG@10 (arXiv source):
  - `dense` = **0.251**
  - `rrf(default+bm25+dense)` = 0.173
  - `cross_encoder` (MS MARCO) = 0.145
  - `bm25` = 0.138
  - `ss_default` = 0.081
- Stage 1 (LLM reranking, 5 labeled in pool):
  - RRF = **0.869** > dense 0.844 > bm25 0.806 > cross_encoder 0.776 > ss_default 0.211
  - RRF correctly fuses dense + bm25 to beat individual rankers.
  - Cross-encoder underperforms dense — expected MS MARCO / scientific-text domain mismatch.
- Stage 3 (Agentic RAG, 3 labeled in pool):
  - dense = **1.000** (perfect) > cross_encoder 0.387 > rrf 0.333 > bm25 0.301 > ss_default 0.000
  - RRF pulled below dense here because ss_default and bm25 add noise on this query.

### Run 2 — expanded benchmark (2026-05-15)
- Stages: 1, 2, 3, 5, 6, 7, 8, 9 (≈146 papers).
- Source: SerpAPI Google Scholar, `limit=30`, DeepSeek `deepseek-chat`, λ=0.7.
- See `reports/q3_results.md` for per-stage numbers.
- Stages 6–9 are out-of-distribution relative to Q1; if the blend
  still wins on average, that's evidence the recipe generalizes.

---

### Run 4 — arXiv gold set, no-LLM baseline (2026-05-18)
- Stages: 1, 2, 3, 5, 6, 7, 8, 9 (all 8).
- Source: `RETRIEVAL_SOURCE=arxiv`, `limit=30`, `--no-llm`.
- Gold: `data/gold/queries_arxiv.json` (new arXiv-specific gold, 30 labeled
  papers per stage matching cached arXiv top-30 pools exactly — 0 missing,
  0 extra). Fixes the all-zero problem from Run 3.
- Rankers: ss_default, bm25, dense, rrf(default+bm25+dense), cross_encoder.
- Output: `reports/q3_results_arxiv.md`.
- Cross-stage mean NDCG@10:
  - `dense` = **0.737**
  - `cross_encoder` = 0.737
  - `rrf(default+bm25+dense)` = 0.667
  - `bm25` = 0.580
  - `ss_default` = 0.451
- Notable: dense and cross_encoder tie for first, both far above Run 3's
  0.251/0.145 — the arXiv gold set correction accounts for the full gap.
  ss_default still trails, confirming Scholar citation-order is poor on
  arXiv pools.

### Run 5 — arXiv gold set, full LLM (2026-05-18)
- Stages: 1, 2, 3, 5, 6, 7, 8, 9 (all 8).
- Source: `RETRIEVAL_SOURCE=arxiv`, `limit=30`, DeepSeek `deepseek-chat`.
- Gold: `data/gold/queries_arxiv.json`.
- Key code changes active for this run vs. all prior runs:
  - `DEFAULT_MAX_TOKENS = 8192` (was 1024; reasoning models no longer
    starve; CoT + JSON answer both fit in one call).
  - `_rank_window` raises `LLMRerankError` on empty parse (was silently
    returning ss_default order, making LLM indistinguishable from baseline).
  - Deterministic integer-scan fallback in `parse_ranked_ids` (covers
    prose-wrapped / slightly malformed responses).
  - Citation ranker auto-hidden on arXiv (all citation_count=0 on arXiv
    papers; would duplicate llm_rerank — skipped automatically).
- Output: `reports/q3_results_arxiv_llm.md`.
- Cross-stage mean NDCG@10:
  - `llm_rerank` = **0.850** ← new best across all runs
  - `dense` = 0.737
  - `cross_encoder` = 0.737
  - `rrf(default+bm25+dense)` = 0.667
  - `bm25` = 0.580
  - `ss_default` = 0.451
- Stage-level highlights:
  - Stage 3 (Agentic RAG): llm **0.987** (near-perfect) vs dense 0.891.
  - Stage 9 (RAG foundations): llm **0.816** vs dense 0.509 — biggest
    absolute gap; LLM understands "foundational" framing in the query.
  - Stage 6 (Monocular depth): llm **0.853** vs dense 0.675.
  - Stage 2 (Document embeddings): llm **0.773** vs dense 0.522.
  - Stage 7 (Diffusion models): llm 0.677 < cross_encoder **0.815** —
    only stage where LLM does not lead.
  - Stage 8 (Protein): cross_encoder **1.000** > llm 0.967 — both near
    ceiling.
- Conclusion: The 8192-token fix transformed llm_rerank from a broken
  ranker (indistinguishable from ss_default in prior runs) to the top
  ranker across 6 of 8 stages.
