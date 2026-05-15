"""Semantic Scholar Graph API client with disk cache.

The free /graph/v1/paper/search endpoint requires no auth (~1 req/s rate
limit) and returns the same kind of bibliographic records the Asta paper
finder consumes upstream. Results are cached on disk keyed by SHA256 of
(query, limit) so repeated runs of the demo are free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = ",".join(
    [
        "paperId",
        "corpusId",
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "citationCount",
        "referenceCount",
        "externalIds",
        "publicationTypes",
        "publicationDate",
    ]
)
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "retrieve"
MAX_LIMIT = 100


@dataclass(frozen=True)
class Paper:
    """A single bibliographic record returned by Semantic Scholar."""

    paper_id: str
    corpus_id: str | None
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    citation_count: int
    external_ids: dict[str, Any]

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "Paper":
        return cls(
            paper_id=str(raw.get("paperId") or ""),
            corpus_id=str(raw["corpusId"]) if raw.get("corpusId") is not None else None,
            title=str(raw.get("title") or "").strip(),
            abstract=raw.get("abstract"),
            authors=[a.get("name", "") for a in (raw.get("authors") or [])],
            year=raw.get("year"),
            venue=raw.get("venue"),
            citation_count=int(raw.get("citationCount") or 0),
            external_ids=dict(raw.get("externalIds") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cache_key(query: str, limit: int) -> str:
    return hashlib.sha256(f"{query}\n{limit}".encode("utf-8")).hexdigest()[:32]


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def search(
    query: str,
    limit: int = 50,
    fields: str = DEFAULT_FIELDS,
    use_cache: bool = True,
    api_key: str | None = None,
) -> list[Paper]:
    """Search Semantic Scholar for papers matching `query`.

    Hits the free /graph/v1/paper/search endpoint. Results are disk-cached
    by SHA256(query+limit) so reruns are free.
    """
    key = _cache_key(query, limit)
    cache_path = _cache_path(key)

    if use_cache and cache_path.exists():
        logger.info("Cache hit for query=%r limit=%d (%s)", query, limit, cache_path.name)
        payload = json.loads(cache_path.read_text())
    else:
        logger.info("Fetching SS results for query=%r limit=%d", query, limit)
        payload = _fetch(query, limit, fields, api_key)
        cache_path.write_text(json.dumps(payload, indent=2))

    return [Paper.from_api(item) for item in payload.get("data", []) or []]


def _fetch(
    query: str, limit: int, fields: str, api_key: str | None
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": min(limit, MAX_LIMIT),
        "fields": fields,
    }

    backoff = 2.0
    last_status: int | None = None
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            resp = requests.get(
                SS_SEARCH_URL, params=params, headers=headers, timeout=30
            )
        except requests.RequestException as exc:
            last_err = exc
            logger.warning("Network error on attempt %d: %s", attempt + 1, exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        if resp.status_code == 200:
            return resp.json()
        last_status = resp.status_code
        if resp.status_code in (429, 502, 503, 504):
            logger.warning(
                "SS API returned %d on attempt %d; backing off %.1fs",
                resp.status_code,
                attempt + 1,
                backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        resp.raise_for_status()

    hint = (
        " (HTTP 429 = free pool throttled; request a free Semantic Scholar API "
        "key at https://www.semanticscholar.org/product/api and set "
        "SEMANTIC_SCHOLAR_API_KEY to bump the limit ~100x)"
        if last_status == 429
        else ""
    )
    raise RuntimeError(
        f"Semantic Scholar search failed after retries: query={query!r} "
        f"(last status: {last_status}, last error: {last_err}).{hint}"
    )


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Search Semantic Scholar from the command line."
    )
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    papers = search(
        args.query,
        limit=args.limit,
        use_cache=not args.no_cache,
        api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
    )

    print(f"\nFound {len(papers)} papers for: {args.query!r}\n")
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:3]) + ("..." if len(p.authors) > 3 else "")
        print(f"{i:2}. [{p.year or '----'}] {p.title}")
        print(f"    Authors: {authors}")
        print(
            f"    Venue: {p.venue or '-'} | "
            f"Citations: {p.citation_count} | "
            f"id: {p.paper_id}"
        )
        print()


if __name__ == "__main__":
    main()
