#!/usr/bin/env python3
"""Synthetic validation of the citation-authority blend.

WHAT THIS PROVES (and does NOT prove)
-------------------------------------
This is a *correctness* test, not an *efficacy* test. On a hand-built
citation subgraph with a ground-truth ordering derived by hand, it
confirms that ``citation_rerank.py``:

  * computes in-set in-degree correctly,
  * min-max normalizes it,
  * blends it with base-ranker scores at lambda = 0.7, and
  * reorders candidates exactly as the formula dictates.

It carries ZERO evidential weight on whether the blend improves ranking
quality on real papers -- synthetic data can be built to make the blend
look arbitrarily good or bad. Efficacy on real citation graphs and gold
labels remains future work, pending a Semantic Scholar API key.

HAND-DERIVED GROUND TRUTH (in-set, 6-node graph)
------------------------------------------------
Query: "the foundational method behind technique X."

Citation edges (citing -> cited), in-set only:
    P2 -> P1
    P3 -> P1, P2
    P4 -> P1
    P5 -> P1, P4
  => in-set in-degree:  P1=4  P2=1  P3=0  P4=1  P5=0  P6=0

Synthetic base-ranker (LLM) scores -- topical wording, not authority:
    P3=0.95  P2=0.90  P1=0.60  P4=0.55  P5=0.30  P6=0.10

min-max norm(LLM):  P3=1.0000 P2=0.9412 P1=0.5882 P4=0.5294 P5=0.2353 P6=0
min-max norm(auth): P1=1.0000 P2=0.2500 P4=0.2500 P3=0      P5=0      P6=0

final = 0.7*norm(LLM) + 0.3*norm(auth):
    P2 = 0.7*0.9412 + 0.3*0.25 = 0.733824
    P1 = 0.7*0.5882 + 0.3*1.00 = 0.711765
    P3 = 0.7*1.0000 + 0.3*0.00 = 0.700000
    P4 = 0.7*0.5294 + 0.3*0.25 = 0.445588
    P5 = 0.7*0.2353 + 0.3*0.00 = 0.164706
    P6 = 0.7*0.0000 + 0.3*0.00 = 0.000000

Pure-LLM order : P3, P2, P1, P4, P5, P6
Blended  order : P2, P1, P3, P4, P5, P6   <-- recorded expectation

Falsifiable prediction: P1 (lowest LLM score among the relevant papers,
highest authority) RISES from rank 3 to rank 2; P3 (high LLM, zero
authority) slips from rank 1 to rank 3.

HAND-DERIVED GROUND TRUTH (global fallback, 4-node)
---------------------------------------------------
No reference graph -- authority approximated by log1p(citation_count).
    base scores : G2=0.95 G3=0.70 G1=0.55 G4=0.20
    citations   : G1=20000 G3=10 G2=5 G4=0
Pure-base order : G2, G3, G1, G4
Blended  order  : G2, G1, G3, G4   <-- recorded expectation
(G1's 20000 citations lift it past G3 despite a lower base score.)

Run standalone:  python q3_reranker/tests/test_citation_blend_synthetic.py
Or under pytest: pytest q3_reranker/tests/test_citation_blend_synthetic.py
"""
from __future__ import annotations

import contextlib
import math
import shutil
import sys
import tempfile
from pathlib import Path

# Make the q3_reranker package importable when this file is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from q3_reranker import citation_rerank  # noqa: E402
from q3_reranker.baselines import RankedResult  # noqa: E402
from q3_reranker.retriever import Paper  # noqa: E402

TOL = 1e-6

# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------

INSET_TITLES = {
    "P1": "Foundational Method X",
    "P2": "Application of X to Task A",
    "P3": "A Survey of X Variants",
    "P4": "X Applied to Domain B",
    "P5": "A Niche Extension of X",
    "P6": "An Unrelated Study of Topic Y",
}

INSET_ABSTRACTS = {
    "P1": "We introduce Method X, a foundational technique for the core "
          "problem. Subsequent work builds directly on this formulation.",
    "P2": "We apply Method X to Task A and report strong empirical gains, "
          "demonstrating the practical value of the foundational technique.",
    "P3": "This survey catalogs the many variants of Method X proposed to "
          "date and organizes them into a unified taxonomy.",
    "P4": "Method X is adapted to Domain B. We describe the modifications "
          "required and evaluate them on domain-specific benchmarks.",
    "P5": "We propose a niche extension of Method X targeting an uncommon "
          "setting with limited general applicability.",
    "P6": "An unrelated empirical study of Topic Y, included as a distractor "
          "with no connection to Method X.",
}

# Citation edges: citing-id -> list of cited-ids (in-set edges only).
INSET_GRAPH = {
    "P1": [],
    "P2": ["P1"],
    "P3": ["P1", "P2"],
    "P4": ["P1"],
    "P5": ["P1", "P4"],
    "P6": [],
}

# Synthetic base-ranker (LLM) scores: topical wording, NOT authority.
INSET_LLM_SCORES = {
    "P1": 0.60, "P2": 0.90, "P3": 0.95, "P4": 0.55, "P5": 0.30, "P6": 0.10,
}

# Hand-derived expectations, recorded independently of the production code
# (see the derivation table in the module docstring).
HAND_INSET_INDEGREE = {"P1": 4, "P2": 1, "P3": 0, "P4": 1, "P5": 0, "P6": 0}
HAND_INSET_ORDER = ["P2", "P1", "P3", "P4", "P5", "P6"]
HAND_GLOBAL_ORDER = ["G2", "G1", "G3", "G4"]

GLOBAL_TITLES = {
    "G1": "Foundational Method X (heavily cited)",
    "G2": "A Topically Precise Application of X",
    "G3": "A Mid-Relevance Study of X",
    "G4": "An Unrelated Topic Y",
}
GLOBAL_BASE_SCORES = {"G1": 0.55, "G2": 0.95, "G3": 0.70, "G4": 0.20}
GLOBAL_CITATIONS = {"G1": 20000, "G2": 5, "G3": 10, "G4": 0}


def make_paper(pid: str, title: str, abstract: str,
               citation_count: int = 0) -> Paper:
    """Build a fully-populated synthetic ``Paper`` (all 9 fields required)."""
    return Paper(
        paper_id=pid,
        corpus_id=None,
        title=title,
        abstract=abstract,
        authors=["Synthetic Author"],
        year=2020,
        venue="Synthetic Venue",
        citation_count=citation_count,
        external_ids={},
    )


# --------------------------------------------------------------------------
# Independent reference implementation (pure Python; no production import).
# Used to cross-check production scores. The hand-recorded orderings above
# are the primary ground truth; this catches numeric bugs as well.
# --------------------------------------------------------------------------

def reference_normalize(values: list[float]) -> list[float]:
    """Min-max normalize; all-equal input collapses to zeros (mirrors spec)."""
    vals = [float(v) for v in values]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.0] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def reference_blend(base_scores: list[float], authority: list[float],
                    blend: float) -> list[float]:
    """final = blend * norm(base) + (1 - blend) * norm(authority)."""
    bn = reference_normalize(base_scores)
    an = reference_normalize(authority)
    return [blend * b + (1.0 - blend) * a for b, a in zip(bn, an)]


def order_by_score(scores: list[float], ids: list[str]) -> list[str]:
    """Stable descending sort of ids by paired score."""
    return [pid for _, pid in sorted(
        zip(scores, ids), key=lambda t: -t[0])]


# --------------------------------------------------------------------------
# Test harness: patch the Semantic Scholar network seam offline
# --------------------------------------------------------------------------

@contextlib.contextmanager
def synthetic_citation_graph(graph: dict[str, list[str]]):
    """Serve ``graph`` through the real SS-batch parsing path, offline.

    ``graph`` maps citing-id -> list of cited-ids. The fake ``_post_batch``
    returns payloads in the real ``/graph/v1/paper/batch`` shape, so
    ``fetch_references`` exercises its genuine parsing code. ``CACHE_DIR`` is
    redirected to a temp dir so the real refs cache is never written to.
    """
    def fake_post_batch(paper_ids, api_key=None):
        return [
            {
                "paperId": pid,
                "references": [{"paperId": c} for c in graph.get(pid, [])],
            }
            for pid in paper_ids
        ]

    orig_post = citation_rerank._post_batch
    orig_cache = citation_rerank.CACHE_DIR
    tmp = Path(tempfile.mkdtemp(prefix="q3_synth_refs_"))
    citation_rerank._post_batch = fake_post_batch
    citation_rerank.CACHE_DIR = tmp
    try:
        yield
    finally:
        citation_rerank._post_batch = orig_post
        citation_rerank.CACHE_DIR = orig_cache
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# Per-assertion report machinery
# --------------------------------------------------------------------------

_REPORT: list[tuple[str, str, bool, str]] = []


def _check(local: list, name: str, passed: bool, detail: str = "") -> None:
    local.append((name, bool(passed), detail))


def _finish(section: str, local: list) -> None:
    for name, passed, detail in local:
        _REPORT.append((section, name, passed, detail))
    failed = [n for n, p, _ in local if not p]
    assert not failed, f"{section}: {len(failed)} check(s) failed: {failed}"


def _inset_inputs():
    """Build the 6 synthetic papers and the pure-LLM base ranking."""
    papers = [
        make_paper(pid, INSET_TITLES[pid], INSET_ABSTRACTS[pid])
        for pid in INSET_TITLES
    ]
    order = sorted(INSET_LLM_SCORES, key=lambda k: -INSET_LLM_SCORES[k])
    base_ranking = [
        RankedResult(pid, INSET_TITLES[pid], INSET_LLM_SCORES[pid])
        for pid in order
    ]
    return papers, base_ranking, order


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_inset_parsing() -> None:
    """fetch_references parses /paper/batch payloads into the in-set graph."""
    local: list = []
    _, base_ranking, _ = _inset_inputs()
    ids = [r.paper_id for r in base_ranking]
    with synthetic_citation_graph(INSET_GRAPH):
        refs = citation_rerank.fetch_references(ids, use_cache=False)
    expected = {pid: set(INSET_GRAPH[pid]) for pid in ids}
    _check(local, "fetch_references returns one entry per input id",
           set(refs) == set(ids), f"keys={sorted(refs)}")
    for pid in ids:
        _check(local, f"references parsed for {pid}",
               refs.get(pid) == expected[pid],
               f"expected={sorted(expected[pid])} got={sorted(refs.get(pid, []))}")
    _finish("In-set: SS batch-payload parsing", local)


def test_inset_indegree() -> None:
    """At blend=0 the output score equals normalized in-set in-degree."""
    local: list = []
    papers, base_ranking, order = _inset_inputs()
    with synthetic_citation_graph(INSET_GRAPH):
        out = citation_rerank.citation_rerank(
            base_ranking, papers, blend=0.0, use_cache=False)
    score_by_id = {r.paper_id: r.score for r in out}
    # Hand-derived in-degree -> normalized via the independent reference.
    expected_norm = reference_normalize(
        [HAND_INSET_INDEGREE[pid] for pid in order])
    expected_by_id = {pid: expected_norm[i] for i, pid in enumerate(order)}
    for pid in order:
        ok = abs(score_by_id[pid] - expected_by_id[pid]) < TOL
        _check(local,
               f"in-degree({pid})={HAND_INSET_INDEGREE[pid]} "
               f"-> norm {expected_by_id[pid]:.4f}",
               ok,
               f"blend=0 output score={score_by_id[pid]:.6f}")
    _finish("In-set: in-degree computation (via blend=0)", local)


def test_inset_blend() -> None:
    """At blend=0.7 the output order matches the hand-derived ranking."""
    local: list = []
    papers, base_ranking, order = _inset_inputs()
    with synthetic_citation_graph(INSET_GRAPH):
        out = citation_rerank.citation_rerank(
            base_ranking, papers, blend=0.7, use_cache=False)
    actual_order = [r.paper_id for r in out]

    # Independent re-derivation (pure Python, no production blend imported).
    ref_blended = reference_blend(
        [INSET_LLM_SCORES[p] for p in order],
        [HAND_INSET_INDEGREE[p] for p in order],
        0.7)
    ref_order = order_by_score(ref_blended, order)

    _check(local, "hand-recorded order == independent reference order",
           ref_order == HAND_INSET_ORDER,
           f"reference={ref_order} hand={HAND_INSET_ORDER}")
    _check(local, "production order == hand-derived order",
           actual_order == HAND_INSET_ORDER,
           f"production={actual_order} hand={HAND_INSET_ORDER}")

    score_by_id = {r.paper_id: r.score for r in out}
    ref_by_id = {p: ref_blended[i] for i, p in enumerate(order)}
    for pid in order:
        ok = abs(score_by_id[pid] - ref_by_id[pid]) < TOL
        _check(local, f"blended score {pid}", ok,
               f"production={score_by_id[pid]:.6f} reference={ref_by_id[pid]:.6f}")

    pure_order = [r.paper_id for r in base_ranking]
    p1_pure, p1_blended = pure_order.index("P1"), actual_order.index("P1")
    _check(local, "FALSIFIABLE: P1 rises above its pure-LLM rank",
           p1_blended < p1_pure,
           f"P1 pure index {p1_pure} -> blended index {p1_blended}")
    p3_pure, p3_blended = pure_order.index("P3"), actual_order.index("P3")
    _check(local, "FALSIFIABLE: P3 slips below its pure-LLM rank",
           p3_blended > p3_pure,
           f"P3 pure index {p3_pure} -> blended index {p3_blended}")
    _finish("In-set: lambda=0.7 blend & reorder", local)


def test_global_fallback() -> None:
    """citation_rerank_global blends base scores with log1p(citation_count)."""
    local: list = []
    papers = [
        make_paper(pid, GLOBAL_TITLES[pid], "Synthetic abstract.",
                   citation_count=GLOBAL_CITATIONS[pid])
        for pid in GLOBAL_TITLES
    ]
    order = sorted(GLOBAL_BASE_SCORES, key=lambda k: -GLOBAL_BASE_SCORES[k])
    base_ranking = [
        RankedResult(pid, GLOBAL_TITLES[pid], GLOBAL_BASE_SCORES[pid])
        for pid in order
    ]
    out = citation_rerank.citation_rerank_global(base_ranking, papers, blend=0.7)
    actual_order = [r.paper_id for r in out]

    ref_blended = reference_blend(
        [GLOBAL_BASE_SCORES[p] for p in order],
        [math.log1p(GLOBAL_CITATIONS[p]) for p in order],
        0.7)
    ref_order = order_by_score(ref_blended, order)

    _check(local, "hand-recorded order == independent reference order",
           ref_order == HAND_GLOBAL_ORDER,
           f"reference={ref_order} hand={HAND_GLOBAL_ORDER}")
    _check(local, "production order == hand-derived order",
           actual_order == HAND_GLOBAL_ORDER,
           f"production={actual_order} hand={HAND_GLOBAL_ORDER}")

    score_by_id = {r.paper_id: r.score for r in out}
    ref_by_id = {p: ref_blended[i] for i, p in enumerate(order)}
    for pid in order:
        ok = abs(score_by_id[pid] - ref_by_id[pid]) < TOL
        _check(local, f"log-blended score {pid}", ok,
               f"production={score_by_id[pid]:.6f} reference={ref_by_id[pid]:.6f}")

    g1_pure, g1_blended = order.index("G1"), actual_order.index("G1")
    _check(local, "G1 rises on global-citation authority",
           g1_blended < g1_pure,
           f"G1 pure index {g1_pure} -> blended index {g1_blended}")
    _finish("Global fallback: log1p(citation_count) blend", local)


def test_edge_all_equal_authority() -> None:
    """arXiv case: no citation edges => zero authority => pure base order."""
    local: list = []
    titles = {"A": "Paper A", "B": "Paper B", "C": "Paper C"}
    scores = {"A": 0.9, "B": 0.6, "C": 0.3}
    papers = [make_paper(p, titles[p], "abstract") for p in titles]
    order = ["A", "B", "C"]
    base_ranking = [RankedResult(p, titles[p], scores[p]) for p in order]
    with synthetic_citation_graph({p: [] for p in titles}):
        out = citation_rerank.citation_rerank(
            base_ranking, papers, blend=0.7, use_cache=False)
    actual = [r.paper_id for r in out]
    _check(local, "all-zero authority leaves base order unchanged",
           actual == order, f"expected={order} got={actual}")
    _finish("Edge case: all-equal (zero) authority", local)


def test_edge_single_paper() -> None:
    """A one-paper pool cannot be reordered and must not crash."""
    local: list = []
    papers = [make_paper("S", "Solo Paper", "abstract")]
    base_ranking = [RankedResult("S", "Solo Paper", 0.5)]
    with synthetic_citation_graph({"S": []}):
        out = citation_rerank.citation_rerank(
            base_ranking, papers, blend=0.7, use_cache=False)
    actual = [r.paper_id for r in out]
    _check(local, "single-paper pool returns exactly that paper",
           actual == ["S"], f"got={actual}")
    _finish("Edge case: single paper", local)


def test_edge_empty_pool() -> None:
    """An empty pool returns an empty list from both code paths."""
    local: list = []
    out_inset = citation_rerank.citation_rerank([], [], use_cache=False)
    out_global = citation_rerank.citation_rerank_global([], [])
    _check(local, "in-set path returns [] on empty pool", out_inset == [],
           f"got={out_inset}")
    _check(local, "global path returns [] on empty pool", out_global == [],
           f"got={out_global}")
    _finish("Edge case: empty pool", local)


ALL_TESTS = [
    test_inset_parsing,
    test_inset_indegree,
    test_inset_blend,
    test_global_fallback,
    test_edge_all_equal_authority,
    test_edge_single_paper,
    test_edge_empty_pool,
]


# --------------------------------------------------------------------------
# Standalone report runner
# --------------------------------------------------------------------------

def _run_report() -> bool:
    for t in ALL_TESTS:
        try:
            t()
        except AssertionError:
            pass  # per-check results already recorded in _REPORT
        except Exception as exc:  # noqa: BLE001  -- a crash is itself a failure
            _REPORT.append((t.__name__, f"unexpected error: {exc!r}", False, ""))

    width = 74
    print("=" * width)
    print("SYNTHETIC VALIDATION  --  citation-authority blend")
    print("=" * width)
    print()
    print("Synthetic in-set citation graph (citing -> cited, in-set edges):")
    for pid, cited in INSET_GRAPH.items():
        print(f"   {pid} -> {', '.join(cited) if cited else '(none)'}")
    print()
    pure_inset = sorted(INSET_LLM_SCORES, key=lambda k: -INSET_LLM_SCORES[k])
    pure_global = sorted(GLOBAL_BASE_SCORES, key=lambda k: -GLOBAL_BASE_SCORES[k])
    print(f"   hand-derived in-set in-degree : {HAND_INSET_INDEGREE}")
    print(f"   in-set  pure-LLM order        : {pure_inset}")
    print(f"   in-set  hand-derived blended  : {HAND_INSET_ORDER}")
    print(f"   global  pure-base order       : {pure_global}")
    print(f"   global  hand-derived blended  : {HAND_GLOBAL_ORDER}")

    section = None
    passed = failed = 0
    for sec, name, ok, detail in _REPORT:
        if sec != section:
            section = sec
            print()
            print(section)
            print("-" * width)
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}")
        if detail:
            print(f"         -> {detail}")
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)

    print()
    print("=" * width)
    print(f"RESULT: {passed} passed, {failed} failed "
          f"({passed + failed} assertions total)")
    print("=" * width)
    print()
    if failed == 0:
        print("PROVEN (implementation correctness only):")
        print("  In-set in-degree computation and the lambda=0.7 blend reorder")
        print("  candidates exactly as specified, verified on a 6-node synthetic")
        print("  citation graph; the log1p global-citation fallback verified on")
        print("  a 4-node case. Efficacy on real citation data was NOT tested --")
        print("  that requires a Semantic Scholar API key and is future work.")
    else:
        print("FAILURES above: the implementation does NOT match the")
        print("hand-derived specification. See the detail lines.")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_report() else 1)
