"""Gold-label loader for Phase 4 evaluation.

Loads analyst-labeled relevance judgments from ``data/gold/queries.json``
(hand-encoded from ``/home/kaiyul3/ASAT/comparison_report.md``).

Grading scheme:
    3 = "✓ Core" / "✓ Core anchor"
    2 = "✓ Related" / "✓ Background/survey"
    1 = "✓ Partial"
    0 = "✗ *" (off-topic, irrelevant, marginal, etc.)

Title matching is fuzzy via canonicalization: lowercase, strip
non-alphanumeric, collapse whitespace. Substring fallback is allowed
only when the candidate title is meaningfully long (>= 30 chars).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

GOLD_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "queries.json"

_MIN_FALLBACK_LEN = 30


@dataclass(frozen=True)
class GoldStage:
    stage: int
    topic: str
    query: str  # NL question (Set B) — used as input to the LLM reranker
    retrieval_query: str  # short keyword form (Set A) — used to retrieve candidates
    papers: dict[str, int] = field(default_factory=dict)


def normalize_title(title: str) -> str:
    """Canonicalize a title for fuzzy matching."""
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def load_gold(path: Path | None = None) -> list[GoldStage]:
    """Load all gold stages from disk."""
    path = path or GOLD_PATH
    raw = json.loads(path.read_text())
    stages: list[GoldStage] = []
    for stage in raw.get("stages", []):
        nl_query = stage["query"]
        stages.append(
            GoldStage(
                stage=int(stage["stage"]),
                topic=stage.get("topic", ""),
                query=nl_query,
                retrieval_query=stage.get("retrieval_query") or nl_query,
                papers={
                    normalize_title(t): int(g)
                    for t, g in stage.get("papers", {}).items()
                },
            )
        )
    return stages


def gold_for_query(query: str, stages: list[GoldStage] | None = None) -> GoldStage | None:
    """Find a gold stage by exact query match."""
    stages = stages or load_gold()
    for stage in stages:
        if stage.query == query:
            return stage
    return None


def grade_for(gold: GoldStage, title: str) -> int:
    """Look up the graded relevance for a paper title; returns 0 if unlabeled.

    Tries strict normalized match first, then a length-guarded substring
    fallback to handle minor truncation or extra subtitles.
    """
    norm = normalize_title(title)
    if not norm:
        return 0
    if norm in gold.papers:
        return gold.papers[norm]
    if len(norm) >= _MIN_FALLBACK_LEN:
        for gold_title, grade in gold.papers.items():
            if len(gold_title) < _MIN_FALLBACK_LEN:
                continue
            if gold_title in norm or norm in gold_title:
                return grade
    return 0


def main() -> None:
    """Print a small summary of the loaded gold set."""
    stages = load_gold()
    print(f"Loaded {len(stages)} gold stages from {GOLD_PATH.name}\n")
    for s in stages:
        total = len(s.papers)
        by_grade: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for g in s.papers.values():
            by_grade[g] = by_grade.get(g, 0) + 1
        rel = sum(by_grade[g] for g in (1, 2, 3))
        print(f"Stage {s.stage} — {s.topic}")
        print(f"  query: {s.query[:90]}...")
        print(
            f"  papers: {total} total | grade 3: {by_grade[3]:2} | "
            f"grade 2: {by_grade[2]:2} | grade 1: {by_grade[1]:2} | "
            f"grade 0: {by_grade[0]:2} | relevant (>0): {rel}"
        )
        print()


if __name__ == "__main__":
    main()
