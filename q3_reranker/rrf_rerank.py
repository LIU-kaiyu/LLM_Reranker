"""Reciprocal Rank Fusion (RRF) ensemble ranker.

Combines multiple ranked lists using the formula from Cormack et al. (2009):

    score(d) = Σ_r  1 / (k + rank_r(d))

where rank_r(d) is the 1-based position of document d in ranker r's list,
and k=60 is the standard damping constant.  Documents absent from a
ranker's list are penalised with rank n+1 (last place in that list).

Because every upstream ranker in this project ranks the full candidate pool,
the n+1 fallback is rarely triggered, but it is handled correctly.
"""

from __future__ import annotations

from .baselines import RankedResult

DEFAULT_K = 60


def rrf(
    rankings: dict[str, list[RankedResult]],
    k: int = DEFAULT_K,
) -> list[RankedResult]:
    """Fuse ranked lists via Reciprocal Rank Fusion.

    Args:
        rankings: ranker-name → ranked list (best-first).  At least one entry
            required.
        k: RRF damping constant (default 60).

    Returns:
        RankedResult list sorted best-first by RRF score.
    """
    if not rankings:
        return []

    # Index of paper_id → RankedResult (for title lookup)
    paper_index: dict[str, RankedResult] = {}
    for ranking in rankings.values():
        for r in ranking:
            if r.paper_id not in paper_index:
                paper_index[r.paper_id] = r

    # Per-ranker: paper_id → 1-based rank
    rank_maps: list[dict[str, int]] = []
    max_n = 0
    for ranking in rankings.values():
        rank_maps.append({r.paper_id: i for i, r in enumerate(ranking, 1)})
        max_n = max(max_n, len(ranking))

    rrf_scores: dict[str, float] = {}
    for paper_id in paper_index:
        rrf_scores[paper_id] = sum(
            1.0 / (k + rm.get(paper_id, max_n + 1)) for rm in rank_maps
        )

    return [
        RankedResult(
            paper_id=pid,
            title=paper_index[pid].title,
            score=rrf_scores[pid],
        )
        for pid in sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    ]


def main() -> None:
    import argparse
    import logging
    from pathlib import Path

    from .baselines import bm25, dense, ss_default
    from .sources import search

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="RRF ensemble of ss_default+bm25+dense.")
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    papers = search(args.query, limit=args.limit)
    if not papers:
        print(f"No papers found for: {args.query!r}")
        return

    fused = rrf({
        "ss_default": ss_default(papers),
        "bm25": bm25(args.query, papers),
        "dense": dense(args.query, papers),
    })
    print(f"\n=== RRF top-{args.top_k} for {args.query!r} ===")
    for i, r in enumerate(fused[: args.top_k], 1):
        print(f"{i:2}. ({r.score:.4f})  {r.title[:90]}")


if __name__ == "__main__":
    main()
