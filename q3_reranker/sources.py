"""Retrieval source dispatcher.

Picks between Semantic Scholar and SerpAPI Google Scholar based on the
``RETRIEVAL_SOURCE`` env var (default: ``serpapi``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .retriever import Paper

logger = logging.getLogger(__name__)


def get_source() -> str:
    return (os.environ.get("RETRIEVAL_SOURCE") or "serpapi").lower()


def search(query: str, limit: int = 20, **kwargs: Any) -> list[Paper]:
    """Dispatch to the configured retriever; returns a list of ``Paper``."""
    source = get_source()
    logger.info("Retrieval source: %s", source)
    if source == "serpapi":
        from .serpapi_retriever import search as _search

        return _search(query, limit=limit, **kwargs)
    if source in ("semantic_scholar", "ss"):
        from .retriever import search as _search

        return _search(query, limit=limit, **kwargs)
    if source == "arxiv":
        from .arxiv_retriever import search as _search

        return _search(query, limit=limit, **kwargs)
    raise ValueError(
        f"Unknown RETRIEVAL_SOURCE={source!r}; "
        f"expected 'serpapi', 'semantic_scholar', or 'arxiv'."
    )
