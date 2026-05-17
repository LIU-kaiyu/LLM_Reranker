"""Baseline rankers for the Q3 demo.

Three baselines are provided so that Phase 3's LLM reranker has something
honest to beat:

- ``ss_default``: keep the order Semantic Scholar returned.
- ``bm25``: BM25 over ``title + abstract`` via the ``bm25s`` library.
- ``dense``: cosine similarity in a sentence-transformer embedding space
  (``all-MiniLM-L6-v2`` by default; swap to ``allenai/specter2_base`` if
  you have the ``adapters`` library installed).

Each ranker returns a list of ``RankedResult`` sorted from most to least
relevant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .retriever import Paper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedResult:
    paper_id: str
    title: str
    score: float


def _paper_text(p: Paper) -> str:
    """Concatenated title + abstract for ranking."""
    parts: list[str] = [p.title or ""]
    if p.abstract:
        parts.append(p.abstract)
    return " ".join(parts).strip()


def ss_default(papers: Sequence[Paper]) -> list[RankedResult]:
    """Trivial baseline: keep the order Semantic Scholar returned."""
    n = len(papers)
    return [
        RankedResult(paper_id=p.paper_id, title=p.title, score=float(n - i))
        for i, p in enumerate(papers)
    ]


def bm25(query: str, papers: Sequence[Paper]) -> list[RankedResult]:
    """BM25 (Okapi) over title + abstract."""
    import bm25s

    corpus = [_paper_text(p) for p in papers]
    tokens = bm25s.tokenize(corpus, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokens)

    q_tokens = bm25s.tokenize([query], stopwords="en")
    results, scores = retriever.retrieve(q_tokens, k=len(papers))

    ranked: list[RankedResult] = []
    for idx, score in zip(results[0], scores[0]):
        p = papers[int(idx)]
        ranked.append(
            RankedResult(paper_id=p.paper_id, title=p.title, score=float(score))
        )
    return ranked


def dense(
    query: str,
    papers: Sequence[Paper],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[RankedResult]:
    """Dense baseline: cosine similarity in a sentence-transformer space."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    corpus = [_paper_text(p) for p in papers]
    doc_emb = model.encode(
        corpus, normalize_embeddings=True, show_progress_bar=False
    )
    q_emb = model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )
    scores = np.asarray(doc_emb) @ np.asarray(q_emb).T
    scores = scores.squeeze(axis=-1)

    order = np.argsort(-scores)
    return [
        RankedResult(
            paper_id=papers[int(i)].paper_id,
            title=papers[int(i)].title,
            score=float(scores[int(i)]),
        )
        for i in order
    ]


def main() -> None:
    import argparse

    from .sources import search

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run baseline rankers on a query.")
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip dense baseline (avoids downloading the sentence-transformers model).",
    )
    args = parser.parse_args()

    papers = search(args.query, limit=args.limit)
    if not papers:
        print(f"No papers found for: {args.query!r}")
        return

    rankers: dict[str, list[RankedResult]] = {
        "ss_default": ss_default(papers),
        "bm25": bm25(args.query, papers),
    }
    if not args.skip_dense:
        rankers["dense"] = dense(args.query, papers)

    for name, ranking in rankers.items():
        print(f"\n=== {name} top-{args.top_k} ===")
        for i, r in enumerate(ranking[: args.top_k], 1):
            print(f"{i:2}. ({r.score:7.3f})  {r.title[:90]}")


if __name__ == "__main__":
    main()
