"""Phase 5: side-by-side demo CLI.

Run::

    python -m q3_reranker.demo \
        --query "agentic retrieval augmented generation scientific QA" \
        --nl-query "What are the key papers on agentic RAG for science?" \
        --top-k 5

``--query`` is the keyword form sent to Scholar for retrieval.
``--nl-query`` is the natural-language form passed to the LLM reranker;
it defaults to ``--query`` when omitted.

For a given query, runs all enabled rankers and prints a column-aligned
top-K table so you can see which papers each method surfaces.

If the query matches one of the gold stages, also prints a small metric
strip (NDCG@10 / MRR / Recall@10) at the bottom.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .baselines import RankedResult
from .eval import _build_rankers, mrr, ndcg_at_k, recall_at_k
from .gold import GoldStage, gold_for_query, grade_for
from .query_expand import QueryExpansion, expand
from .sources import get_source, search

_SOURCE_LABELS = {
    "arxiv": "arXiv",
    "serpapi": "SerpAPI Google Scholar",
    "semantic_scholar": "Semantic Scholar",
    "ss": "Semantic Scholar",
}


def _title(r: RankedResult, max_len: int = 70) -> str:
    t = r.title.strip()
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def _print_table(
    rankings: dict[str, list[RankedResult]],
    top_k: int,
    gold: GoldStage | None = None,
) -> None:
    names = list(rankings.keys())
    rows: list[list[str]] = []
    for i in range(top_k):
        row = [f"{i + 1:2}"]
        for name in names:
            ranking = rankings[name]
            if i < len(ranking):
                r = ranking[i]
                grade = grade_for(gold, r.title) if gold else 0
                tag = f" [{grade}]" if gold and grade > 0 else ""
                row.append(_title(r) + tag)
            else:
                row.append("")
        rows.append(row)

    col_widths = [
        max(len(rows[i][j]) for i in range(len(rows))) for j in range(len(names) + 1)
    ]
    col_widths[0] = max(col_widths[0], 4)
    for i, name in enumerate(names, start=1):
        col_widths[i] = max(col_widths[i], len(name))

    header = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(["#"] + names))
    print(header)
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print("  ".join(c.ljust(col_widths[i]) for i, c in enumerate(row)))


def _print_metrics(
    rankings: dict[str, list[RankedResult]], gold: GoldStage, top_k: int = 10
) -> None:
    n_relevant = sum(
        1 for r in next(iter(rankings.values())) if grade_for(gold, r.title) > 0
    )
    print(
        f"\nMetrics (gold = Stage {gold.stage}, "
        f"{n_relevant} relevant docs in candidate pool):"
    )
    print(f"  {'ranker':14s}  NDCG@{top_k}  MRR    Recall@{top_k}")
    for name, ranking in rankings.items():
        grades = [grade_for(gold, r.title) for r in ranking]
        n = ndcg_at_k(grades, top_k)
        m = mrr(grades)
        rec = recall_at_k(grades, n_relevant, top_k)
        print(f"  {name:14s}  {n:.3f}    {m:.3f}  {rec:.3f}")


def _review_expansion(exp: QueryExpansion, auto_approve: bool) -> tuple[str, str]:
    """Show the LLM expansion and let the user approve / edit / reject.

    Returns ``(retrieval_query, nl_query)``. Raises ``SystemExit`` if the
    user rejects.
    """
    kind_label = {
        "nl": "natural-language question",
        "keyword": "keyword query",
        "unknown": "unclassified (LLM expansion failed; using raw text for both forms)",
    }.get(exp.kind, exp.kind)

    print()
    print(f"LLM classification: {kind_label}")
    print(f"  retrieval_query: {exp.retrieval_query}")
    print(f"  nl_query:        {exp.nl_query}")
    print()

    if auto_approve:
        print("--yes given; running without review.")
        return exp.retrieval_query, exp.nl_query

    retrieval = exp.retrieval_query
    nl = exp.nl_query
    while True:
        choice = input("Approve and run? [y]es / [e]dit / [n]o: ").strip().lower()
        if choice in ("y", "yes", ""):
            return retrieval, nl
        if choice in ("n", "no"):
            raise SystemExit("Aborted by user.")
        if choice in ("e", "edit"):
            new_r = input(f"  retrieval_query [{retrieval}]: ").strip()
            new_n = input(f"  nl_query        [{nl}]: ").strip()
            if new_r:
                retrieval = new_r
            if new_n:
                nl = new_n
            print()
            print(f"  retrieval_query: {retrieval}")
            print(f"  nl_query:        {nl}")
            print()
            continue
        print("Please answer y / e / n.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Side-by-side reranker demo.")
    parser.add_argument(
        "--query",
        default=None,
        help="Keyword retrieval query (sent to Scholar). Required unless --text is given.",
    )
    parser.add_argument(
        "--nl-query",
        default=None,
        help="Natural-language query for the LLM reranker (defaults to --query).",
    )
    parser.add_argument(
        "--text",
        default=None,
        help=(
            "Single user query in either form (NL question or keywords). "
            "An LLM expander classifies it and produces the counterpart; "
            "you review/edit/approve before retrieval runs."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive review prompt when using --text.",
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-citation", action="store_true")
    args = parser.parse_args()

    if args.text and args.query:
        parser.error("Use --text OR --query, not both.")
    if not args.text and not args.query:
        parser.error("One of --text or --query is required.")

    if args.text:
        exp = expand(args.text)
        retrieval_query, nl_query = _review_expansion(exp, auto_approve=args.yes)
    else:
        retrieval_query = args.query
        nl_query = args.nl_query or args.query

    papers = search(retrieval_query, limit=args.limit)
    if not papers:
        print(f"No papers found for: {retrieval_query!r}")
        return

    rankers = _build_rankers(
        enable_llm=not args.no_llm,
        enable_citation=not args.no_citation,
    )

    rankings: dict[str, list[RankedResult]] = {
        name: ranker(nl_query, papers) for name, ranker in rankers.items()
    }

    gold = gold_for_query(retrieval_query)
    source_label = _SOURCE_LABELS.get(get_source(), get_source())
    print(
        f"\nQuery: {retrieval_query!r}  "
        f"({len(papers)} candidates from {source_label})\n"
    )
    _print_table(rankings, top_k=args.top_k, gold=gold)

    if gold is not None:
        _print_metrics(rankings, gold, top_k=args.top_k)


if __name__ == "__main__":
    main()
