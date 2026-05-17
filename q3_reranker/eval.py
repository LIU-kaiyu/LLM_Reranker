"""Phase 4: Evaluation harness.

Runs all rankers against the gold-labeled queries from
``data/gold/queries.json`` and reports NDCG@10, MRR, and Recall@10.

Recall denominator is "number of relevant (grade > 0) papers found in the
candidate pool", not "all relevant papers in the gold set" — this scores
the *reranker* fairly given a fixed retrieval, rather than penalising it
for retrieval misses.

Cache:
- Semantic Scholar search payloads     -> data/cache/retrieve/
- Citation-graph reference lists       -> data/cache/refs/
- LLM (DeepSeek) listwise responses    -> data/cache/llm/

After the first run the eval is fully deterministic and free to re-run.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .baselines import RankedResult, bm25, dense, ss_default
from .citation_rerank import citation_rerank, citation_rerank_global
from .cross_encoder_rerank import cross_encoder_rerank
from .gold import GoldStage, grade_for, load_gold, normalize_title
from .llm_rerank import get_backend, rerank as llm_rerank_fn
from .retriever import Paper
from .rrf_rerank import rrf
from .sources import get_source, search

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def dcg(grades: Sequence[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(grades: Sequence[int], k: int = 10) -> float:
    actual = dcg(list(grades[:k]))
    ideal = dcg(sorted(grades, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def mrr(grades: Sequence[int]) -> float:
    for i, g in enumerate(grades):
        if g > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(grades: Sequence[int], n_relevant: int, k: int = 10) -> float:
    if n_relevant == 0:
        return 0.0
    hits = sum(1 for g in grades[:k] if g > 0)
    return hits / n_relevant


# ---------------------------------------------------------------------------
# Eval orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRow:
    stage: int
    topic: str
    ranker: str
    ndcg10: float
    mrr: float
    recall10: float
    n_retrieved: int
    n_labeled_in_pool: int
    n_relevant_in_pool: int


def _is_labeled(gold: GoldStage, title: str) -> bool:
    """True if title matches any gold entry (regardless of grade)."""
    norm = normalize_title(title)
    if not norm:
        return False
    if norm in gold.papers:
        return True
    if len(norm) >= 30:
        for gt in gold.papers:
            if len(gt) >= 30 and (gt in norm or norm in gt):
                return True
    return False


def _build_rankers(
    enable_llm: bool,
    enable_citation: bool,
    enable_cross: bool = True,
) -> dict[str, Callable[[str, list[Paper]], list[RankedResult]]]:
    rankers: dict[str, Callable[[str, list[Paper]], list[RankedResult]]] = {
        "ss_default": lambda q, ps: ss_default(ps),
        "bm25": lambda q, ps: bm25(q, ps),
        "dense": lambda q, ps: dense(q, ps),
        "rrf(default+bm25+dense)": lambda q, ps: rrf({
            "ss_default": ss_default(ps),
            "bm25": bm25(q, ps),
            "dense": dense(q, ps),
        }),
    }
    if enable_cross:
        rankers["cross_encoder"] = lambda q, ps: cross_encoder_rerank(q, ps)
    if enable_llm:
        backend = get_backend()

        def _llm(q: str, ps: list[Paper]) -> list[RankedResult]:
            return llm_rerank_fn(q, ps, backend=backend)

        rankers["llm_rerank"] = _llm

        if enable_citation:
            source = get_source()
            if source in ("semantic_scholar", "ss"):

                def _llm_cite(q: str, ps: list[Paper]) -> list[RankedResult]:
                    base = llm_rerank_fn(q, ps, backend=backend)
                    return citation_rerank(
                        base,
                        ps,
                        api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
                    )

                rankers["llm+citation(refs)"] = _llm_cite
            else:
                # SerpAPI / arXiv / sources without per-paper reference lists:
                # fall back to global citation_count as the authority signal.
                # Note: for arXiv, citation_count=0 for all papers, so the
                # blend is a no-op and the result equals pure llm_rerank.
                if source == "arxiv":
                    logger.warning(
                        "arXiv source has citation_count=0 for all papers; "
                        "llm+citation(global) will be identical to llm_rerank."
                    )

                def _llm_cite_global(q: str, ps: list[Paper]) -> list[RankedResult]:
                    base = llm_rerank_fn(q, ps, backend=backend)
                    return citation_rerank_global(base, ps)

                rankers["llm+citation(global)"] = _llm_cite_global

    return rankers


def evaluate_stage(
    gold: GoldStage,
    rankers: dict[str, Callable[[str, list[Paper]], list[RankedResult]]],
    limit: int = 50,
) -> list[EvalRow]:
    # Retrieve with the short keyword form (Set A), which is what produced the
    # analyst's gold labels. The rankers themselves still see the NL form
    # (Set B) as the user-facing query.
    logger.info(
        "Stage %d retrieval_query=%r | reranker_query=%r",
        gold.stage,
        gold.retrieval_query,
        gold.query,
    )
    # Each retriever picks up its own key from env (SERPAPI_KEY or
    # SEMANTIC_SCHOLAR_API_KEY).
    papers = search(gold.retrieval_query, limit=limit)
    if not papers:
        logger.warning("Stage %d: no papers retrieved", gold.stage)
        return []

    n_labeled = sum(1 for p in papers if _is_labeled(gold, p.title))
    n_relevant = sum(1 for p in papers if grade_for(gold, p.title) > 0)

    logger.info(
        "Stage %d: %d retrieved, %d labeled, %d relevant in pool",
        gold.stage,
        len(papers),
        n_labeled,
        n_relevant,
    )

    rows: list[EvalRow] = []
    for name, ranker in rankers.items():
        ranking = ranker(gold.query, papers)
        grades = [grade_for(gold, r.title) for r in ranking]
        rows.append(
            EvalRow(
                stage=gold.stage,
                topic=gold.topic,
                ranker=name,
                ndcg10=ndcg_at_k(grades, 10),
                mrr=mrr(grades),
                recall10=recall_at_k(grades, n_relevant, 10),
                n_retrieved=len(papers),
                n_labeled_in_pool=n_labeled,
                n_relevant_in_pool=n_relevant,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_table(rows: list[EvalRow]) -> str:
    import pandas as pd

    if not rows:
        return "(no rows)"
    df = pd.DataFrame([r.__dict__ for r in rows])
    pivot = (
        df.pivot_table(
            index=["stage", "topic", "ranker"],
            values=["ndcg10", "mrr", "recall10"],
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["stage", "ranker"])
    )
    return pivot.to_markdown(index=False, floatfmt=".3f")


def format_summary(rows: list[EvalRow]) -> str:
    import pandas as pd

    if not rows:
        return "(no rows)"
    df = pd.DataFrame([r.__dict__ for r in rows])
    summary = (
        df.groupby("ranker", as_index=False)
        .agg(
            ndcg10=("ndcg10", "mean"),
            mrr=("mrr", "mean"),
            recall10=("recall10", "mean"),
        )
        .sort_values("ndcg10", ascending=False)
    )
    return summary.to_markdown(index=False, floatfmt=".3f")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Run reranker eval over the gold set.")
    parser.add_argument("--limit", type=int, default=30, help="SS candidates per query")
    parser.add_argument(
        "--stage", type=int, default=None, help="Run only this stage number"
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM reranker")
    parser.add_argument(
        "--no-citation",
        action="store_true",
        help="Skip the citation-graph reranker (also skipped if --no-llm)",
    )
    parser.add_argument(
        "--no-cross",
        action="store_true",
        help="Skip the cross-encoder reranker (skips model download on first run)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPORTS_DIR / "q3_results.md",
        help="Markdown report path",
    )
    args = parser.parse_args()

    stages = load_gold()
    if args.stage is not None:
        stages = [s for s in stages if s.stage == args.stage]
        if not stages:
            print(f"No gold stage matches --stage {args.stage}")
            return

    rankers = _build_rankers(
        enable_llm=not args.no_llm,
        enable_citation=not args.no_citation,
        enable_cross=not args.no_cross,
    )

    all_rows: list[EvalRow] = []
    for stage in stages:
        print(f"\n--- Stage {stage.stage}: {stage.topic} ---")
        rows = evaluate_stage(stage, rankers, limit=args.limit)
        all_rows.extend(rows)
        for r in rows:
            print(
                f"  {r.ranker:14s}  NDCG@10={r.ndcg10:.3f}  "
                f"MRR={r.mrr:.3f}  Recall@10={r.recall10:.3f}  "
                f"(pool relevant={r.n_relevant_in_pool}/{r.n_labeled_in_pool} labeled)"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "# Q3 Reranker Evaluation\n\n"
        "Generated by `python -m q3_reranker.eval`. Metrics computed against "
        "analyst-labeled gold from `comparison_report.md`.\n\n"
        "## Per-stage results\n\n"
        + format_table(all_rows)
        + "\n\n## Cross-stage summary (mean over stages)\n\n"
        + format_summary(all_rows)
        + "\n"
    )
    args.out.write_text(body)
    print(f"\nWrote report -> {args.out}")


if __name__ == "__main__":
    main()
