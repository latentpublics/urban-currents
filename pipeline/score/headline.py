"""Headline score and quiet-day decision (PRD §5.6).

    headline = 0.40*relevance + 0.20*source_multiplicity
             + 0.20*artifact_completeness + 0.20*novelty

The threshold is **not** guessed from a two-week sample — it comes from the
90-day backfill distribution, picked so the headline rate lands in 30-50%
(`uc calibrate`). A quiet day is not an empty day: every card still publishes.
It only means no item earned the top slot.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..config import scoring_config
from ..models import Item


def _weights() -> dict[str, float]:
    return (scoring_config().get("weights") or {})


def source_multiplicity(item: Item) -> float:
    sat = float(scoring_config().get("source_multiplicity_saturation", 3) or 3)
    n = max(1, len(item.cluster.members))
    return min(1.0, (n - 1) / max(1.0, sat - 1))


def artifact_completeness(item: Item) -> float:
    cfg = scoring_config().get("artifact_completeness") or {}
    total = 0.0
    if "code" in item.badges:
        total += float(cfg.get("code", 0.4))
    if "data" in item.badges:
        total += float(cfg.get("data", 0.4))
    if "published" in item.badges:
        total += float(cfg.get("published", 0.2))
    return min(1.0, total)


def novelty(item: Item, seen_entity_ids: Optional[set[str]] = None) -> float:
    """Share of this item's method/data/tool tags never seen before.

    A paper introducing tags the archive has not carried is, by construction,
    doing something the archive has not covered.
    """
    overlay = [e.id for e in (item.entities.methods + item.entities.data + item.entities.tools)]
    if not overlay:
        return 0.0
    if seen_entity_ids is None:
        return 0.0
    fresh = sum(1 for eid in overlay if eid not in seen_entity_ids)
    return round(fresh / len(overlay), 4)


def score_item(item: Item, seen_entity_ids: Optional[set[str]] = None) -> Item:
    w = _weights()
    comp = item.scores.components
    comp.relevance = item.scores.relevance
    comp.source_multiplicity = round(source_multiplicity(item), 4)
    comp.artifact_completeness = round(artifact_completeness(item), 4)
    comp.novelty = novelty(item, seen_entity_ids)
    item.scores.headline = round(
        float(w.get("relevance", 0.40)) * comp.relevance
        + float(w.get("source_multiplicity", 0.20)) * comp.source_multiplicity
        + float(w.get("artifact_completeness", 0.20)) * comp.artifact_completeness
        + float(w.get("novelty", 0.20)) * comp.novelty,
        4,
    )
    return item


def score_all(items: Iterable[Item], seen_entity_ids: Optional[set[str]] = None) -> list[Item]:
    return [score_item(it, seen_entity_ids) for it in items]


def headline_threshold() -> float:
    return float(scoring_config().get("headline_threshold", 0.5))


def pick_headline(items: Sequence[Item], threshold: Optional[float] = None) -> Optional[Item]:
    """Top-scoring item, if any clears the threshold. Otherwise it is a quiet day."""
    if not items:
        return None
    thr = headline_threshold() if threshold is None else threshold
    best = max(items, key=lambda it: (it.scores.headline, it.work_key))
    return best if best.scores.headline >= thr else None


def headline_line(item: Item) -> str:
    """One line drawn from the summary — never invented here, and never from a
    field the LLM could have hallucinated bibliography into."""
    what = (item.summary.en.what if item.summary.en else "") or ""
    first = what.split(". ")[0].strip()
    if first:
        return first if first.endswith(".") else first + "."
    return item.bibliography.title
