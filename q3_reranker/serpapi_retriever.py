"""SerpAPI Google Scholar retriever (drop-in replacement for SS).

Maps SerpAPI's Scholar response shape into the same ``Paper`` dataclass
the rest of the pipeline uses, so the dispatcher in ``sources.py`` can
swap retrievers without touching the rankers.

Key trade-offs vs. Semantic Scholar:
- ``abstract`` is populated from the SerpAPI ``snippet`` (~200 chars,
  truncated with "..."). BM25 / dense baselines have less text to work
  with; the LLM reranker is fine on title + snippet.
- There is no per-paper reference list. The citation-graph reranker
  (Phase 6) falls back to the global ``citation_count`` signal — see
  ``citation_rerank.citation_rerank_global``.
- Each non-cached SerpAPI request costs 1 credit. Results are cached on
  disk keyed by SHA256(query + limit + year filters).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from .retriever import Paper

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "serpapi"

MAX_NUM_PER_CALL = 20  # SerpAPI hard cap per call


def _cache_key(query: str, limit: int, year_low: int | None, year_high: int | None) -> str:
    payload = f"{query}\n{limit}\n{year_low}\n{year_high}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _parse_summary(summary: str | None) -> tuple[list[str], str | None, int | None]:
    """Parse 'A Author, B Author - Venue, 2024 - publisher' into (authors, venue, year).

    Returns empty values for any segment we can't parse. We never raise
    on malformed summaries — Scholar's format varies a lot.
    """
    if not summary:
        return [], None, None

    parts = re.split(r"\s+-\s+", summary)
    authors: list[str] = []
    venue: str | None = None
    year: int | None = None

    if parts:
        authors = [a.strip() for a in re.split(r",\s*", parts[0]) if a.strip()]

    if len(parts) >= 2:
        middle = parts[1]
        year_match = re.search(r"\b(19|20)\d{2}\b", middle)
        if year_match:
            year = int(year_match.group(0))
            venue = middle.replace(year_match.group(0), "").strip(", ").strip()
            if not venue:
                venue = None
        else:
            venue = middle.strip(", ").strip() or None

    return authors, venue, year


def _to_paper(raw: dict[str, Any]) -> Paper:
    """Map one SerpAPI organic_results entry to our Paper dataclass."""
    title = (raw.get("title") or "").strip()
    snippet = raw.get("snippet") or None

    pub = raw.get("publication_info") or {}
    summary = pub.get("summary")
    structured_authors = pub.get("authors") or []
    parsed_authors, parsed_venue, parsed_year = _parse_summary(summary)

    if structured_authors:
        authors = [a.get("name", "").strip() for a in structured_authors if a.get("name")]
    else:
        authors = parsed_authors

    inline = raw.get("inline_links") or {}
    cited_by = inline.get("cited_by") or {}
    versions = inline.get("versions") or {}

    external_ids = {
        "serpapi_result_id": raw.get("result_id"),
        "serpapi_cites_id": cited_by.get("cites_id"),
        "serpapi_cluster_id": versions.get("cluster_id"),
    }
    external_ids = {k: v for k, v in external_ids.items() if v}

    return Paper(
        paper_id=str(raw.get("result_id") or ""),
        corpus_id=None,
        title=title,
        abstract=snippet,
        authors=authors,
        year=parsed_year,
        venue=parsed_venue,
        citation_count=int(cited_by.get("total") or 0),
        external_ids=external_ids,
    )


def _fetch_page(
    query: str,
    start: int,
    num: int,
    api_key: str,
    year_low: int | None,
    year_high: int | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key,
        "start": start,
        "num": min(num, MAX_NUM_PER_CALL),
        "hl": "en",
    }
    if year_low is not None:
        params["as_ylo"] = year_low
    if year_high is not None:
        params["as_yhi"] = year_high

    backoff = 2.0
    last_status: int | None = None
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            resp = requests.get(SERPAPI_URL, params=params, timeout=45)
        except requests.RequestException as exc:
            last_err = exc
            logger.warning("SerpAPI network error on attempt %d: %s", attempt + 1, exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        last_status = resp.status_code
        if resp.status_code == 200:
            payload = resp.json()
            status = (payload.get("search_metadata") or {}).get("status")
            if status == "Success":
                return payload
            logger.warning(
                "SerpAPI returned status=%r; payload error=%r",
                status,
                (payload.get("error") or payload.get("search_metadata")),
            )
            return payload
        if resp.status_code in (429, 502, 503, 504):
            logger.warning(
                "SerpAPI HTTP %d on attempt %d; backing off %.1fs",
                resp.status_code,
                attempt + 1,
                backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        resp.raise_for_status()

    raise RuntimeError(
        f"SerpAPI search failed after retries (last status: {last_status}, "
        f"last error: {last_err})"
    )


def search(
    query: str,
    limit: int = 20,
    use_cache: bool = True,
    api_key: str | None = None,
    year_low: int | None = None,
    year_high: int | None = None,
) -> list[Paper]:
    """Search Google Scholar via SerpAPI; returns ``Paper`` objects.

    Costs ceil(limit / 20) SerpAPI credits on a cache miss. Cache hit is free.
    """
    key = _cache_key(query, limit, year_low, year_high)
    cache_path = _cache_path(key)

    if use_cache and cache_path.exists():
        logger.info(
            "SerpAPI cache hit for query=%r limit=%d (%s)",
            query,
            limit,
            cache_path.name,
        )
        payload = json.loads(cache_path.read_text())
    else:
        api_key = api_key or os.environ.get("SERPAPI_KEY")
        if not api_key:
            raise RuntimeError(
                "SERPAPI_KEY is not set; cannot perform live SerpAPI search."
            )

        all_results: list[dict[str, Any]] = []
        full_metadata: list[dict[str, Any]] = []
        for start in range(0, limit, MAX_NUM_PER_CALL):
            num = min(MAX_NUM_PER_CALL, limit - start)
            page = _fetch_page(query, start, num, api_key, year_low, year_high)
            full_metadata.append(page.get("search_metadata") or {})
            page_results = page.get("organic_results") or []
            if not page_results:
                logger.info(
                    "SerpAPI returned empty page at start=%d; stopping pagination",
                    start,
                )
                break
            all_results.extend(page_results)

        payload = {
            "query": query,
            "limit": limit,
            "organic_results": all_results,
            "search_metadata_pages": full_metadata,
        }
        cache_path.write_text(json.dumps(payload, indent=2))
        logger.info(
            "Cached %d organic_results to %s",
            len(all_results),
            cache_path.name,
        )

    return [_to_paper(item) for item in payload.get("organic_results", [])]


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Search Google Scholar via SerpAPI (1 credit per 20 results)."
    )
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--year-low", type=int, default=None)
    parser.add_argument("--year-high", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    papers = search(
        args.query,
        limit=args.limit,
        use_cache=not args.no_cache,
        year_low=args.year_low,
        year_high=args.year_high,
    )

    print(f"\nGot {len(papers)} papers for: {args.query!r}\n")
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:3]) + ("..." if len(p.authors) > 3 else "")
        snippet = (p.abstract or "").replace("\n", " ")[:140]
        print(f"{i:2}. [{p.year or '----'}] {p.title}")
        print(f"    Authors: {authors}")
        print(
            f"    Venue: {p.venue or '-'} | "
            f"Citations: {p.citation_count} | id: {p.paper_id}"
        )
        print(f"    Snippet: {snippet}...")
        print()


if __name__ == "__main__":
    main()
