# Q3 Reranker — Claim Validation Report

**Date:** 2026-05-22
**Scope:** Verify that every number the Q3 report and video intend to claim traces
to code that actually ran on the stated backend.
**Method:** Re-run the evaluation harness, probe the LLM backend directly, and
inspect source — no code edits to reconcile mismatches; discrepancies are reported
as-found.

---

## 1. Environment and configuration

All numbers below were produced with this configuration unless stated otherwise:

| Setting | Value |
|---|---|
| `RETRIEVAL_SOURCE` | `arxiv` |
| Gold set | `data/gold/queries_arxiv.json` (8 stages: 1,2,3,5,6,7,8,9) |
| `--limit` | 30 candidates per stage |
| LLM backend | DeepSeek `deepseek-chat` (non-reasoning) |
| `DEFAULT_MAX_TOKENS` | 8192 (temporarily 1024 for the Claim 2 probe, then restored) |
| Regenerated report | `reports/q3_verify_full.md` |

The headline numbers in the report correspond to **Run 5** in `experiments_log.md`
(2026-05-18), whose output is `reports/q3_results_arxiv_llm.md`.

---

## 2. What was done

1. **Recon.** Surveyed the project, identified that the headline table comes from
   Run 5, confirmed all 46 LLM cache entries are `deepseek-chat`, and read the
   relevant source (`eval.py`, `llm_rerank.py`, `citation_rerank.py`, `sources.py`).
2. **Full eval re-run.** Ran `python -m q3_reranker.eval --gold data/gold/queries_arxiv.json
   --limit 30 --out reports/q3_verify_full.md` and compared every cell to the report.
3. **Per-paper inspection.** Ran a read-only script to extract Stage-3 per-paper
   rank columns for every ranker, locate AR-RAG, and confirm the citation blend is
   inert on arXiv.
4. **`max_tokens` probe.** Re-ran Stage 3 at `DEFAULT_MAX_TOKENS=1024` and `8192`,
   then called the DeepSeek backend directly to inspect the raw response length.
5. **Restore.** Reverted `DEFAULT_MAX_TOKENS` to 8192 and restored the LLM cache to
   its original 46 files.

---

## 3. Results summary

| # | Claim | Verdict |
|---|---|---|
| 1 | Headline NDCG@10 table is an arXiv-backend run | **PASS (table) / FAIL (win count)** |
| 2 | The `max_tokens` 1024→8192 bug fix is real and changes the result | **FAIL — does not reproduce as described** |
| 3 | The citation blend is inert on arXiv | **PASS** |
| 4 | The two-query design is actually used | **PASS** |
| 5 | The Stage 3 reorder matches the screenshots | **FAIL — specifics wrong** |
| 6 | Negative results are real, not omitted | **PASS** |

---

## 4. Detailed findings

### CLAIM 1 — Headline NDCG@10 table (blocking)

**Mean NDCG@10 reproduces exactly (±0.000)** against `reports/q3_results_arxiv_llm.md`:

| Ranker | Report | Verified |
|---|---|---|
| llm_rerank | 0.850 | 0.850 |
| dense | 0.737 | 0.737 |
| cross_encoder | 0.737 | 0.737 |
| rrf(default+bm25+dense) | 0.667 | 0.667 |
| bm25 | 0.580 | 0.580 |
| ss_default | 0.451 | 0.451 |

- **arXiv fingerprint confirmed:** `citation_count = 0` for every retrieved paper
  across all 8 stages — the signature of an arXiv run.
- **8 stages scored** (1, 2, 3, 5, 6, 7, 8, 9).
- **"llm_rerank wins 6 of 8 stages" is FALSE — actual is 4 of 8.**

Per-stage NDCG@10 winners:

| Stage | Winner | llm_rerank | llm wins? |
|---|---|---|---|
| 1 | dense 0.897 | 0.892 | no |
| 2 | **llm 0.773** | 0.773 | yes |
| 3 | **llm 0.987** | 0.987 | yes |
| 5 | dense 0.961 | 0.838 | no |
| 6 | **llm 0.853** | 0.853 | yes |
| 7 | cross_encoder 0.815 | 0.677 | no |
| 8 | cross_encoder 1.000 | 0.967 | no |
| 9 | **llm 0.816** | 0.816 | yes |

`llm_rerank` wins **stages 2, 3, 6, 9 (4 of 8)**. dense wins 1 and 5; cross_encoder
wins 7 and 8. The `experiments_log.md` Run 5 conclusion makes the same "6 of 8"
overcount — its own per-stage highlights only support 4.

### CLAIM 2 — The `max_tokens` bug fix (blocking)

**The truncation story does not hold for the `deepseek-chat` backend** that produced
the headline numbers.

- Current default confirmed: `DEFAULT_MAX_TOKENS = 8192` (`llm_rerank.py:51`).
- Stage 3 `llm_rerank` NDCG@10: **1024 → 0.824**, **8192 → 0.987** (8192 reproduced
  twice, cache cleared each time).
- **Direct backend probe** (Stage-3 window-1 prompt, cache bypassed):

  | max_tokens | response length | parsed ids | truncated? |
  |---|---|---|---|
  | 1024 | 80 chars | 20 of 20 | no |
  | 8192 | 80 chars | 20 of 20 | no |

  The DeepSeek response is a complete 20-integer JSON array (~30 tokens) at both
  settings. It physically cannot be truncated by a 1024-token cap.

**Conclusion:** the 0.824 vs 0.987 gap is sampling nondeterminism, not truncation.
At 1024, `llm_rerank` remains the #2 ranker on Stage 3 (0.824, far above
`ss_default`'s 0.189) — it was never a "no-op". The truncation-handling code
(`llm_rerank.py:47-51, 177-186`) is real and correct, but only affects *reasoning*
models (`deepseek-reasoner` / `deepseek-v4-pro`), which spend the token budget on
chain-of-thought. The headline run used `deepseek-chat`, where 1024 vs 8192 is
structurally inert. The report's "flat result → best ranker" framing is unsupported
by the actual backend.

### CLAIM 3 — Citation blend is inert on arXiv (blocking)

**PASS.**

- On `RETRIEVAL_SOURCE=arxiv`, `eval.py` registers **no `llm+citation` column** — the
  blend is deliberately skipped (`eval.py:142-151`, logs that it would duplicate
  `llm_rerank`).
- `citation_rerank_global` applied to Stage-3 candidates produces an ordering
  **identical to the base** for `llm_rerank`, `bm25`, and `dense` — because every
  `citation_count` is 0, the authority term normalizes to all-zero and the blend
  collapses to a monotonic transform of the base score.
- Both code paths exist and substantiate "implemented":
  - In-set graph: `citation_rerank.py:37-38` (`/graph/v1/paper/batch`,
    `data/cache/refs/`).
  - Global fallback: `citation_rerank.py:178` (`np.log1p` of `citation_count`).

### CLAIM 4 — Two-query design is used (confirmatory)

**PASS.** All 8 gold stages have a `retrieval_query` distinct from `query`.
`eval.py:180` retrieves with `gold.retrieval_query`; `eval.py:198` passes
`gold.query` to the rankers.

### CLAIM 5 — Stage 3 reorder (confirmatory)

**FAIL on specifics; direction holds.**

- AR-RAG ("AR-RAG: Autoregressive Retrieval Augmentation for Image Generation",
  off-topic, grade 0) is `ss_default` **rank #1** — confirmed.
- `llm_rerank` demotes AR-RAG to **rank #20** — the report says "~rank 8".
- `llm_rerank` Stage-3 top-5:
  1. PaperQA: Retrieval-Augmented Generative Agent for Scientific Research
  2. SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation
  3. Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation
  4. Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics
  5. HiPerRAG: High-Performance Retrieval Augmented Generation for Scientific Insights
- The report states top-3 as "Automated Literature Review #1, then PaperQA, SQuAI" —
  the actual order is **PaperQA #1, SQuAI #2, Automated Literature Review #3**.
- Promoted papers are topically on-query (all top-10 are RAG / agent / scientific-QA
  papers); PaperQA and SQuAI were promoted from outside `ss_default`'s top-10 —
  confirmed.

### CLAIM 6 — Negative results are real (confirmatory)

**PASS.**

- cross_encoder underperforms dense on stages 1, 3, 5, 6, 9 (5 of 8); the two tie at
  0.737 mean. The claimed negative is present and has not flipped.
- RRF is dragged below dense on Stage 3 (0.547 vs 0.891) because `ss_default`
  contributes no signal there — present, and documented in `detail.md:237`.
- λ=0.7: the citation blend is inert on arXiv, so this is not exercised by this run;
  the claim pertains to non-arXiv runs and can be neither confirmed nor refuted here.

---

## 5. Minor observations

- Stage 2 returned **29** arXiv papers rather than 30; the other 7 stages returned
  30. This does not materially affect any metric.

---

## 6. State left behind

- `q3_reranker/llm_rerank.py` — restored to `DEFAULT_MAX_TOKENS = 8192`
  (byte-identical to its original state).
- `data/cache/llm/` — restored to the original 46 cache files.
- **New file:** `reports/q3_verify_full.md` — the regenerated arXiv+LLM eval,
  byte-identical to the existing `reports/q3_results_arxiv_llm.md`. Retained as
  evidence.
- No existing report or source file was edited to reconcile any mismatch.

---

## 7. Bottom line

The headline NDCG@10 table is genuine and arXiv-sourced — it reproduces exactly.
Three sub-claims must be corrected before the report and video are finalized:

1. `llm_rerank` wins **4 of 8** stages, not 6.
2. The `max_tokens` bug story does **not** apply to the `deepseek-chat` backend used
   for the headline numbers; raising the cap to 8192 did not turn a broken ranker
   into the best one. The fix is a legitimate safeguard for reasoning models only.
3. On Stage 3, `llm_rerank` demotes AR-RAG to **rank 20** and ranks **PaperQA #1**
   — not the ranks the report currently states.
