"""Phase 6 (novel angle): citation-graph reranker.

Re-scores a base ranker's top-K output by exploiting one-hop citation
edges *within the candidate set*. Intuition: foundational papers in a
topic are cited by other relevant papers in the same candidate pool, so
we can detect them without relying on global citation counts (which
favour old papers regardless of topical fit).

Procedure:
  1. Take the base ranker's top-K candidates.
  2. Fetch each candidate's reference list from Semantic Scholar via the
     ``/graph/v1/paper/batch`` endpoint (one batch call for the whole
     window, cached on disk).
  3. Build a directed graph: edge j -> i if paper j cites paper i.
  4. Authority(i) = in-degree of i in the top-K induced subgraph.
  5. Final score = blend * base_norm + (1 - blend) * authority_norm.
  6. Re-sort; append untouched tail.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import requests

from .baselines import RankedResult
from .retriever import Paper

logger = logging.getLogger(__name__)

SS_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "refs"

DEFAULT_TOP_K = 20
DEFAULT_BLEND = 0.7


def _cache_key(paper_ids: Sequence[str]) -> str:
    joined = "|".join(sorted(paper_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _post_batch(
    paper_ids: list[str], api_key: str | None
) -> list[dict[str, Any] | None]:
    """POST to /paper/batch with retry/backoff. Returns aligned with input order."""
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key
    params = {"fields": "paperId,references.paperId"}
    body = {"ids": list(paper_ids)}

    backoff = 2.0
    last_status: int | None = None
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            resp = requests.post(
                SS_BATCH_URL,
                params=params,
                headers=headers,
                json=body,
                timeout=60,
            )
        except requests.RequestException as exc:
            last_err = exc
            logger.warning("Network error on batch attempt %d: %s", attempt + 1, exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        last_status = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                return data
            logger.warning("Unexpected batch payload type: %s", type(data).__name__)
            return []
        if resp.status_code in (429, 502, 503, 504):
            logger.warning(
                "SS batch returned %d on attempt %d; backing off %.1fs",
                resp.status_code,
                attempt + 1,
                backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        resp.raise_for_status()

    raise RuntimeError(
        f"SS batch failed after retries (last status: {last_status}, "
        f"last error: {last_err})"
    )


def fetch_references(
    paper_ids: list[str],
    api_key: str | None = None,
    use_cache: bool = True,
) -> dict[str, set[str]]:
    """Fetch references for each paper; returns {paper_id: set(cited_ids)}."""
    if not paper_ids:
        return {}

    key = _cache_key(paper_ids)
    cache_path = _cache_path(key)

    if use_cache and cache_path.exists():
        logger.info("Refs cache hit (%s)", cache_path.name)
        raw = json.loads(cache_path.read_text())
        return {pid: set(cited) for pid, cited in raw.items()}

    logger.info("Refs cache miss; fetching batch for %d papers", len(paper_ids))
    data = _post_batch(paper_ids, api_key)

    refs: dict[str, set[str]] = {pid: set() for pid in paper_ids}
    for paper_id, item in zip(paper_ids, data):
        if not item:
            continue
        ref_list = item.get("references") or []
        refs[paper_id] = {
            r.get("paperId") for r in ref_list if r and r.get("paperId")
        }

    serializable = {pid: sorted(cited) for pid, cited in refs.items()}
    cache_path.write_text(json.dumps(serializable, indent=2))
    return refs


def _normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def citation_rerank_global(
    base_ranking: list[RankedResult],
    papers: Sequence[Paper],
    top_k: int = DEFAULT_TOP_K,
    blend: float = DEFAULT_BLEND,
) -> list[RankedResult]:
    """Cheap fallback: blend base ranker with global ``citation_count``.

    Used when per-paper reference lists are unavailable (e.g., SerpAPI
    Google Scholar). Trade-off: this leans on the source's own citation-
    count signal rather than the in-set graph, so it partially recovers
    Scholar's default secondary sort. Honest framing for the write-up:
    "approximate in-set authority via global citations, log-normalized
    within the candidate pool."
    """
    if not base_ranking:
        return []

    paper_by_id = {p.paper_id: p for p in papers}
    top = [r for r in base_ranking[:top_k] if r.paper_id in paper_by_id]
    if not top:
        return list(base_ranking)

    raw_cites = np.array(
        [float(paper_by_id[r.paper_id].citation_count) for r in top], dtype=float
    )
    # log1p flattens the heavy tail of citation counts (some papers 5000+, most <100)
    auth = np.log1p(raw_cites)

    base_scores = np.array([r.score for r in top], dtype=float)
    base_n = _normalize(base_scores)
    auth_n = _normalize(auth)
    blended = blend * base_n + (1.0 - blend) * auth_n

    order = np.argsort(-blended)
    reranked_top = [
        RankedResult(
            paper_id=top[int(i)].paper_id,
            title=top[int(i)].title,
            score=float(blended[int(i)]),
        )
        for i in order
    ]

    seen = {r.paper_id for r in reranked_top}
    tail = [r for r in base_ranking if r.paper_id not in seen]
    return reranked_top + tail


def citation_rerank(
    base_ranking: list[RankedResult],
    papers: Sequence[Paper],
    top_k: int = DEFAULT_TOP_K,
    blend: float = DEFAULT_BLEND,
    api_key: str | None = None,
    use_cache: bool = True,
) -> list[RankedResult]:
    """Blend base ranker scores with in-set citation authority.

    ``blend`` controls the mix:
        1.0 -> pure base ranker (no change)
        0.7 -> base-dominant with citation lift (default)
        0.5 -> equal mix
        0.0 -> pure authority (rarely useful alone)
    """
    if not base_ranking:
        return []

    paper_by_id = {p.paper_id: p for p in papers}
    top = [r for r in base_ranking[:top_k] if r.paper_id in paper_by_id]
    if not top:
        return list(base_ranking)

    ids = [r.paper_id for r in top]
    refs = fetch_references(ids, api_key=api_key, use_cache=use_cache)
    id_set = set(ids)

    auth = np.zeros(len(top), dtype=float)
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    for citing_id, cited in refs.items():
        for cited_id in cited:
            if cited_id in id_set and cited_id != citing_id:
                auth[id_to_idx[cited_id]] += 1.0

    base_scores = np.array([r.score for r in top], dtype=float)
    base_n = _normalize(base_scores)
    auth_n = _normalize(auth)
    blended = blend * base_n + (1.0 - blend) * auth_n

    order = np.argsort(-blended)
    reranked_top = [
        RankedResult(
            paper_id=top[int(i)].paper_id,
            title=top[int(i)].title,
            score=float(blended[int(i)]),
        )
        for i in order
    ]

    seen = {r.paper_id for r in reranked_top}
    tail = [r for r in base_ranking if r.paper_id not in seen]
    return reranked_top + tail
