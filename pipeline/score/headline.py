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


def best_candidate(items: Sequence[Item]) -> Optional[Item]:
    """The top-scoring item, **whether or not it clears the bar** (0Z, Z1).

    `pick_headline` computed this and threw it away, so a day where nothing
    cleared the threshold recorded `headline.work_key: null` and the archive
    row lost its representative title — the lead column went blank on a day
    that had published nine papers. `Headline` already keeps `present` and
    `work_key` as separate fields, so the day can say "nothing cleared the bar"
    and still name what came closest.
    """
    if not items:
        return None
    return max(items, key=lambda it: (it.scores.headline, it.work_key))


def pick_headline(items: Sequence[Item], threshold: Optional[float] = None) -> Optional[Item]:
    """Top-scoring item, if any clears the threshold.

    Returning None means **no item cleared the headline bar**. It does not mean
    the day was quiet — that is a question about how much was published, and
    `Issue.is_quiet` answers it (0Z-A, Z1) — and since 0Z-B it does not mean the
    day gets no headline either. See `headline_form`.
    """
    best = best_candidate(items)
    if best is None:
        return None
    thr = headline_threshold() if threshold is None else threshold
    return best if best.scores.headline >= thr else None


def headline_form(items: Sequence[Item], threshold: Optional[float] = None) -> str:
    """`"lead"` or `"day"` — which shape the day's headline takes (0Z-B, B0).

    ## What the threshold now decides

    It used to decide **whether** a day had a headline at all: no item over the
    bar, no headline. That was right while a headline compressed one paper —
    with no representative paper there was nothing to compress.

    Once the line summarises the day, the premise stops holding. A day of nine
    evenly solid papers is not a day with nothing to say; it is the day a
    summary is *most* useful. So the threshold chooses the **form** instead:

      `lead`  one paper stands out — the line is about that paper
      `day`   nothing stands out — the line is about the day's papers

    ## And the measurement that forced it

    The score is degenerate at the published tier. Of 2,315 items in the archive
    on 2026-08-22: **33.8% score exactly 0.44** and only **3.1%** clear the
    0.444 threshold. Every "no headline" day in the archive — 08-21, 08-20,
    08-13, 08-09, 08-02 — has *every* item sitting on that same 0.44, while the
    days that got one have a single item at 0.46 to 0.67.

    So the bar was not separating a standout day from an ordinary one. It was
    separating "one paper happened to pick up a small component bonus" from
    "none did", by a margin of 0.004 over a plateau holding a third of the
    archive. That is a coin toss deciding whether the most visible line on the
    page exists.

    **The weights are untouched** and so is `scoring.yaml`; what changed is how
    the number is used. What it now predicts is the share of days led by a
    single paper rather than the share of days with a headline — see the report
    for what that does to the calibrated `headline_rate`.
    """
    return "lead" if pick_headline(items, threshold) is not None else "day"


def headline_line(item: Item) -> str:
    """One line drawn from the summary — never invented here, and never from a
    field the LLM could have hallucinated bibliography into."""
    what = (item.summary.en.what if item.summary.en else "") or ""
    first = what.split(". ")[0].strip()
    if first:
        return first if first.endswith(".") else first + "."
    return item.bibliography.title
