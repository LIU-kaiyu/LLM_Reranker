"""Smart query expansion (Phase 7).

The GUI search box and CLI ``--text`` flag accept a single user-supplied
string in either form:

  * a natural-language question, e.g. *"What are the key papers on agentic
    RAG for science?"*, or
  * a short keyword query, e.g. *"agentic retrieval augmented generation
    scientific QA"*.

This module asks the configured LLM backend to classify which form the
user typed and emit the counterpart, so downstream code can keep the
two-query design (keyword form for retrieval, NL form for the LLM
reranker prompt) without forcing the user to provide both.

The expansion is cheap (~300 tokens), cached on disk via the same
``_cached_complete`` machinery as the listwise reranker, and degrades
gracefully: on any parse failure we fall back to using the raw text for
both forms (which is the current behaviour today, so worst case is no
worse than the status quo).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from .llm_rerank import LLMBackend, _cached_complete, get_backend

logger = logging.getLogger(__name__)

QueryKind = Literal["nl", "keyword", "unknown"]


@dataclass(frozen=True)
class QueryExpansion:
    """Both forms of a user query plus the LLM's classification of which
    form the user originally supplied.

    ``kind == "unknown"`` indicates that the LLM call failed or returned
    an unparseable response; in that case ``retrieval_query`` and
    ``nl_query`` are both copies of the raw input so the downstream
    pipeline still works.
    """

    kind: QueryKind
    retrieval_query: str
    nl_query: str
    raw: str


SYSTEM_PROMPT = (
    "You help a scientific paper search tool. The user supplies a single "
    "string that is either (a) a natural-language question (a full "
    "sentence or question, e.g. 'What papers established self-supervised "
    "monocular depth estimation?') OR (b) a short keyword query (a noun "
    "phrase or a few keywords, e.g. 'self-supervised monocular depth "
    "estimation'). Classify which kind the user wrote, then produce the "
    "counterpart so we have BOTH a keyword form (used for API retrieval) "
    "and a natural-language form (used as the reranker prompt). The two "
    "forms must describe the same information need.\n\n"
    "Return ONLY a JSON object with these exact keys:\n"
    '  {"kind": "nl" | "keyword",\n'
    '   "retrieval_query": "<5-15 word keyword query, no question marks>",\n'
    '   "nl_query": "<one-sentence natural-language question>"}\n\n'
    "If the user already wrote a keyword query, copy it verbatim into "
    "retrieval_query and write a faithful NL question into nl_query. If "
    "the user wrote a natural-language question, copy it verbatim into "
    "nl_query and write a faithful keyword query into retrieval_query. "
    "Output the JSON object and nothing else."
)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(raw: str) -> dict | None:
    """Lenient JSON extraction: try direct parse, then first ``{...}`` blob."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _fallback(raw_text: str) -> QueryExpansion:
    cleaned = raw_text.strip()
    return QueryExpansion(
        kind="unknown",
        retrieval_query=cleaned,
        nl_query=cleaned,
        raw=cleaned,
    )


def expand(
    raw_text: str,
    backend: LLMBackend | None = None,
) -> QueryExpansion:
    """Classify ``raw_text`` and return both keyword + NL forms.

    On any failure (empty input, backend error, unparseable response) we
    log and return a fallback so the caller can still run retrieval and
    reranking using the raw text for both forms.
    """
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return _fallback("")

    backend = backend or get_backend()
    user = f"User input: {cleaned!r}"

    try:
        raw = _cached_complete(backend, SYSTEM_PROMPT, user)
    except Exception as exc:  # noqa: BLE001 - graceful fallback by design
        logger.warning("Query expansion backend failed: %s", exc)
        return _fallback(cleaned)

    parsed = _parse_response(raw)
    if not parsed:
        logger.warning(
            "Query expander returned unparseable response (first 120 chars): %r",
            raw[:120],
        )
        return _fallback(cleaned)

    kind_val = str(parsed.get("kind", "")).strip().lower()
    kind: QueryKind = kind_val if kind_val in ("nl", "keyword") else "unknown"

    retrieval = str(parsed.get("retrieval_query", "")).strip()
    nl = str(parsed.get("nl_query", "")).strip()

    # Either field empty -> degrade to fallback rather than ship a
    # half-broken expansion to the user.
    if not retrieval or not nl:
        logger.warning(
            "Query expander returned an empty field: kind=%r retrieval=%r nl=%r",
            kind, retrieval, nl,
        )
        return _fallback(cleaned)

    return QueryExpansion(
        kind=kind,
        retrieval_query=retrieval,
        nl_query=nl,
        raw=cleaned,
    )
