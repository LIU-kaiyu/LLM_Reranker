"""LLM-based listwise reranker (Phase 3).

Implements a RankGPT-style sliding-window listwise ranker (Sun et al.,
EMNLP 2023) on top of a pluggable LLM backend. The default backend is
Anthropic Claude; OpenAI and Gemini can be added without changing the
caller surface.

The public function ``rerank`` takes a query and a list of ``Paper``
candidates and returns a list of ``RankedResult`` sorted from most to
least relevant. Internally it:

  1. Slides a window of ``window_size`` candidates from bottom to top.
  2. For each window, asks the LLM to return the ranked identifiers as
     JSON.
  3. Re-stitches the per-window orders into a single global order.
  4. Optionally runs a second pass (``num_passes=2``) for stability.

All LLM responses are cached on disk by ``hash(model + prompt)`` so
re-running the eval is free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .baselines import RankedResult
from .retriever import Paper

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "llm"

DEFAULT_MODEL = "claude-haiku-4-5"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
DEFAULT_WINDOW_SIZE = 20
DEFAULT_STRIDE = 10
DEFAULT_NUM_PASSES = 1
MAX_ABSTRACT_CHARS = 800

SYSTEM_PROMPT = (
    "You are an expert academic literature reviewer. Your job is to rank "
    "candidate papers by how relevant they are to a user's research query. "
    "Relevance means: the paper directly addresses the query topic, uses "
    "methods central to it, or reports core results on it. Avoid being "
    "swayed by paper age or citation count; rank on topical relevance to "
    "the query as written. Return ONLY a JSON array of identifier integers "
    "in ranked order from most to least relevant, with no other text."
)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """A minimal interface for chat-style LLM calls."""

    name: str

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Send a single turn and return the model's text response."""
        ...


@dataclass(frozen=True)
class AnthropicBackend:
    """Anthropic Claude backend."""

    model: str = DEFAULT_MODEL
    name: str = "claude"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        import anthropic  # local import so other backends remain optional

        client = anthropic.Anthropic()
        # Use prompt caching on the system prompt: it's stable across windows
        # so we get cache hits across all calls for a single query.
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        parts: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "".join(parts).strip()


@dataclass(frozen=True)
class DeepSeekBackend:
    """DeepSeek (OpenAI-compatible) backend.

    DeepSeek exposes an OpenAI-compatible Chat Completions endpoint at
    ``https://api.deepseek.com``. We use the official ``openai`` SDK with
    a custom ``base_url`` and the ``DEEPSEEK_API_KEY`` env var.
    """

    model: str = DEEPSEEK_DEFAULT_MODEL
    name: str = "deepseek"
    base_url: str = "https://api.deepseek.com"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to your .env file "
                "or export it before running the reranker."
            )

        client = OpenAI(api_key=api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def get_backend(name: str | None = None) -> LLMBackend:
    """Resolve a backend by name (defaults to env RERANKER_BACKEND, then claude)."""
    name = (name or os.environ.get("RERANKER_BACKEND") or "claude").lower()
    model_env = os.environ.get("RERANKER_MODEL")
    if name == "claude":
        return AnthropicBackend(model=model_env or DEFAULT_MODEL)
    if name == "deepseek":
        return DeepSeekBackend(model=model_env or DEEPSEEK_DEFAULT_MODEL)
    raise NotImplementedError(
        f"Backend {name!r} not implemented yet. Add a class implementing "
        f"LLMBackend.complete() and extend get_backend()."
    )


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _truncate(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _format_candidate(idx: int, p: Paper) -> str:
    title = (p.title or "Untitled").strip()
    year = p.year or "----"
    venue = p.venue or "-"
    snippet = _truncate(p.abstract, MAX_ABSTRACT_CHARS)
    return f"[{idx}] ({year}, {venue}) {title}\n    {snippet}"


def _build_user_prompt(query: str, window: list[tuple[int, Paper]]) -> str:
    listing = "\n\n".join(_format_candidate(idx, p) for idx, p in window)
    ids = [idx for idx, _ in window]
    sample = ids[: min(3, len(ids))]
    return (
        f"Query: {query}\n\n"
        f"Candidate papers (each labeled with an integer identifier):\n\n"
        f"{listing}\n\n"
        f"Rank these {len(window)} candidates from most to least relevant to "
        f"the query. Respond with a single JSON array of the identifiers in "
        f"ranked order, e.g. {sample + ['...']}. Output ONLY the JSON array."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_JSON_ARRAY_RE = re.compile(r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]", re.DOTALL)


def parse_ranked_ids(text: str, valid_ids: set[int]) -> list[int]:
    """Parse a JSON array of integer ids from the model's response.

    Falls back to extracting the first bracketed list if the model wraps
    its answer in prose. Filters to ``valid_ids`` only.
    """
    text = text.strip()
    candidates: list[str] = [text]
    candidates.extend(_JSON_ARRAY_RE.findall(text))

    for snippet in candidates:
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            ids: list[int] = []
            seen: set[int] = set()
            for item in data:
                try:
                    i = int(item)
                except (TypeError, ValueError):
                    continue
                if i in valid_ids and i not in seen:
                    ids.append(i)
                    seen.add(i)
            if ids:
                return ids
    return []


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_key(model: str, system: str, user: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    return h.hexdigest()[:32]


def _cached_complete(backend: LLMBackend, system: str, user: str) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = getattr(backend, "model", backend.name)
    key = _cache_key(model, system, user)
    path = CACHE_DIR / f"{key}.json"

    if path.exists():
        cached = json.loads(path.read_text())
        logger.info("LLM cache hit (%s)", key)
        return cached["response"]

    logger.info("LLM cache miss (%s); calling backend %s", key, backend.name)
    text = backend.complete(system, user)
    path.write_text(
        json.dumps(
            {
                "model": model,
                "backend": backend.name,
                "system": system,
                "user": user,
                "response": text,
            },
            indent=2,
        )
    )
    return text


# ---------------------------------------------------------------------------
# Sliding-window ranker
# ---------------------------------------------------------------------------


def _rank_window(
    query: str,
    indexed_window: list[tuple[int, Paper]],
    backend: LLMBackend,
) -> list[int]:
    """Rank a single window and return ids in ranked (best-first) order."""
    user = _build_user_prompt(query, indexed_window)
    raw = _cached_complete(backend, SYSTEM_PROMPT, user)
    valid = {idx for idx, _ in indexed_window}
    ranked = parse_ranked_ids(raw, valid)
    missing = [idx for idx, _ in indexed_window if idx not in ranked]
    return ranked + missing


def rerank(
    query: str,
    papers: Sequence[Paper],
    backend: LLMBackend | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    num_passes: int = DEFAULT_NUM_PASSES,
) -> list[RankedResult]:
    """Rerank ``papers`` by relevance to ``query`` using a listwise LLM.

    Implements RankGPT-style bottom-to-top sliding-window reranking. With
    ``window_size=20`` and ``stride=10`` we get ~2 overlapping windows per
    10 papers; running ``num_passes=2`` improves stability at 2x cost.
    """
    if not papers:
        return []

    backend = backend or get_backend()
    n = len(papers)
    order: list[int] = list(range(n))

    for pass_idx in range(num_passes):
        start = max(0, n - window_size)
        while True:
            end = start + window_size
            window_indices = order[start:end]
            indexed_window = [(i, papers[i]) for i in window_indices]
            ranked_ids = _rank_window(query, indexed_window, backend)
            order[start:end] = ranked_ids
            if start == 0:
                break
            start = max(0, start - stride)
        logger.info("Pass %d/%d complete", pass_idx + 1, num_passes)

    return [
        RankedResult(
            paper_id=papers[orig_idx].paper_id,
            title=papers[orig_idx].title,
            score=float(n - rank),
        )
        for rank, orig_idx in enumerate(order)
    ]


def main() -> None:
    import argparse

    from .retriever import search

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Auto-load .env for the CLI so users don't have to `source` it
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="LLM listwise reranker (RankGPT-style).")
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--passes", type=int, default=DEFAULT_NUM_PASSES)
    args = parser.parse_args()

    papers = search(args.query, limit=args.limit)
    if not papers:
        print(f"No papers found for: {args.query!r}")
        return

    ranking = rerank(
        args.query,
        papers,
        window_size=args.window,
        stride=args.stride,
        num_passes=args.passes,
    )

    print(f"\n=== LLM rerank top-{args.top_k} for {args.query!r} ===")
    for i, r in enumerate(ranking[: args.top_k], 1):
        print(f"{i:2}. ({r.score:7.1f})  {r.title[:90]}")


if __name__ == "__main__":
    main()
