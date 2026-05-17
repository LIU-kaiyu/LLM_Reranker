"""arXiv Atom feed retriever (drop-in replacement for SerpAPI / Semantic Scholar).

Uses the free arXiv query API — no API key required. Returns **full abstracts**,
unlike SerpAPI which gives a ~200-char snippet.

Key trade-offs vs. other backends:
- ``citation_count`` is always 0: arXiv carries no citation data. The
  citation-graph reranker's global fallback becomes a no-op for these results
  (all authority scores are equal, so the blend collapses to pure LLM score).
- Coverage is excellent for CS / ML / physics / math / quantitative-bio
  preprints, but may miss papers published journal-only without an arXiv copy.
- Rate limit: arXiv asks for ≤3 req/s. A 1-second courtesy sleep is added
  after each live fetch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

from .retriever import Paper

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "arxiv"

_ATOM = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_NS = {"a": _ATOM, "arxiv": _ARXIV_NS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_key(query: str, limit: int) -> str:
    return hashlib.sha256(f"{query}\n{limit}".encode()).hexdigest()[:32]


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _parse_arxiv_id(raw_id: str) -> str:
    """'http://arxiv.org/abs/2305.12345v2' → '2305.12345'."""
    part = raw_id.rstrip("/").split("/")[-1]
    if "v" in part:
        part = part[: part.rfind("v")]
    return part


def _parse_year(date_str: str | None) -> int | None:
    """'2023-05-25T00:00:00Z' → 2023."""
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# XML → serialisable dict  (stored in the disk cache)
# ---------------------------------------------------------------------------


def _entry_to_dict(entry: ET.Element) -> dict[str, Any]:
    raw_id = (entry.findtext("a:id", namespaces=_NS) or "").strip()
    paper_id = _parse_arxiv_id(raw_id)

    title = (
        (entry.findtext("a:title", namespaces=_NS) or "")
        .strip()
        .replace("\n", " ")
    )
    abstract = (
        (entry.findtext("a:summary", namespaces=_NS) or "")
        .strip()
        .replace("\n", " ")
    )
    published = (entry.findtext("a:published", namespaces=_NS) or "").strip()

    authors = [
        (el.findtext("a:name", namespaces=_NS) or "").strip()
        for el in entry.findall("a:author", namespaces=_NS)
    ]
    authors = [a for a in authors if a]

    journal_ref = (
        entry.findtext("arxiv:journal_ref", namespaces=_NS) or ""
    ).strip()
    if journal_ref:
        venue: str | None = journal_ref
    else:
        pcat = entry.find("arxiv:primary_category", namespaces=_NS)
        venue = pcat.get("term") if pcat is not None else None

    doi = (entry.findtext("arxiv:doi", namespaces=_NS) or "").strip()

    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract or None,
        "published": published,
        "authors": authors,
        "venue": venue,
        "doi": doi or None,
    }


def _dict_to_paper(d: dict[str, Any]) -> Paper:
    external_ids: dict[str, Any] = {"arxiv": d["paper_id"]}
    if d.get("doi"):
        external_ids["DOI"] = d["doi"]

    return Paper(
        paper_id=d["paper_id"],
        corpus_id=None,
        title=d["title"],
        abstract=d.get("abstract"),
        authors=d.get("authors") or [],
        year=_parse_year(d.get("published")),
        venue=d.get("venue"),
        citation_count=0,  # arXiv carries no citation counts
        external_ids=external_ids,
    )


# ---------------------------------------------------------------------------
# Live fetch
# ---------------------------------------------------------------------------


def _fetch(query: str, limit: int) -> list[dict[str, Any]]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(1)  # arXiv polite-pool courtesy delay

    root = ET.fromstring(resp.content)
    entries = root.findall("a:entry", namespaces=_NS)
    return [_entry_to_dict(e) for e in entries]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def search(
    query: str,
    limit: int = 20,
    use_cache: bool = True,
    **kwargs: Any,
) -> list[Paper]:
    """Search arXiv via the Atom feed; returns ``Paper`` objects.

    No API key required. ``citation_count`` is always 0 — the citation-graph
    reranker's global fallback has no effect on arXiv results.
    """
    key = _cache_key(query, limit)
    cache_path = _cache_path(key)

    if use_cache and cache_path.exists():
        logger.info(
            "arXiv cache hit for query=%r limit=%d (%s)", query, limit, cache_path.name
        )
        entries = json.loads(cache_path.read_text())
    else:
        logger.info("Fetching arXiv results for query=%r limit=%d", query, limit)
        entries = _fetch(query, limit)
        cache_path.write_text(json.dumps(entries, indent=2))
        logger.info("Cached %d arXiv entries to %s", len(entries), cache_path.name)

    papers: list[Paper] = []
    for d in entries:
        try:
            papers.append(_dict_to_paper(d))
        except Exception as exc:
            logger.warning(
                "Skipping malformed arXiv entry %r: %s", d.get("paper_id"), exc
            )
    return papers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Search arXiv (free, no API key).")
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    papers = search(args.query, limit=args.limit, use_cache=not args.no_cache)
    print(f"\nGot {len(papers)} papers for: {args.query!r}\n")
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:3]) + ("…" if len(p.authors) > 3 else "")
        snippet = (p.abstract or "")[:140].replace("\n", " ")
        print(f"{i:2}. [{p.year or '----'}] {p.title}")
        print(f"    Authors: {authors}")
        print(f"    Venue: {p.venue or '-'} | id: {p.paper_id}")
        print(f"    Abstract: {snippet}…")
        print()


if __name__ == "__main__":
    main()
