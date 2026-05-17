"""Cross-encoder reranker using sentence-transformers.

Scores each (query, title + abstract) pair with a cross-encoder model that
performs full attention over the concatenated input — strictly more expressive
than the bi-encoder dense baseline which encodes query and document
independently.

Default model: ``cross-encoder/ms-marco-MiniLM-L-12-v2`` (~33 MB, CPU-fast).
The model is kept in a module-level cache so it is loaded once per process.

Trade-offs vs. the LLM reranker:
- Faster: ~0.5 s for 30 papers on CPU.
- No API key or LLM call required.
- Trained on MS MARCO (web passages), not scientific papers — a SPECTER2
  cross-encoder would be stronger for this domain but is not publicly
  available as a sentence-transformers cross-encoder at the time of writing.
"""

from __future__ import annotations

import logging
from typing import Sequence

from .baselines import RankedResult
from .retriever import Paper

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
MAX_INPUT_CHARS = 1000  # trim to avoid exceeding 512-token limit

_model_cache: dict[str, object] = {}


def _load_model(model_name: str):
    if model_name not in _model_cache:
        logger.info("Loading cross-encoder: %s", model_name)
        from sentence_transformers import CrossEncoder  # type: ignore[import]
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


def cross_encoder_rerank(
    query: str,
    papers: Sequence[Paper],
    model_name: str = DEFAULT_MODEL,
) -> list[RankedResult]:
    """Rerank ``papers`` by cross-encoder relevance score to ``query``.

    Each paper is represented as ``title + " " + abstract`` (truncated to
    ``MAX_INPUT_CHARS`` characters to stay inside the 512-token window).
    """
    if not papers:
        return []

    model = _load_model(model_name)
    pairs: list[tuple[str, str]] = []
    for p in papers:
        abstract = (p.abstract or "")
        text = f"{p.title or ''} {abstract}".strip()[:MAX_INPUT_CHARS]
        pairs.append((query, text))

    scores = model.predict(pairs, show_progress_bar=False)

    ranked = sorted(zip(scores, papers), key=lambda x: float(x[0]), reverse=True)
    return [
        RankedResult(paper_id=p.paper_id, title=p.title, score=float(s))
        for s, p in ranked
    ]


def main() -> None:
    import argparse
    import logging
    from pathlib import Path

    from .sources import search

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Cross-encoder reranker.")
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    papers = search(args.query, limit=args.limit)
    if not papers:
        print(f"No papers found for: {args.query!r}")
        return

    ranking = cross_encoder_rerank(args.query, papers, model_name=args.model)
    print(f"\n=== cross-encoder top-{args.top_k} for {args.query!r} ===")
    for i, r in enumerate(ranking[: args.top_k], 1):
        print(f"{i:2}. ({r.score:8.4f})  {r.title[:90]}")


if __name__ == "__main__":
    main()
