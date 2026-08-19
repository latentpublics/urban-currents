"""The held queue — what we would not publish unattended (phase 0L, M2-2).

The operating assumption changed. Q4 used to ask whether a human could review a
day in fifteen minutes; it now asks whether the thing can run for a week with
nobody watching and still not publish something the editor would retract. Nobody
is checking before the mail goes out, so **the selection policy has to do the
job the daily review was doing**.

That makes conservatism the correct setting. When we are not sure, we publish
less. An item that trips a suspicion rule is not published and not discarded —
it is held, and the day goes out with a hole where it would have been. **The
hole is the right answer**: a digest that fills its slots with things it is
unsure about is worth less than a shorter one, and a reader cannot see the
difference between a confident item and a slot that needed filling.

Two kinds of doubt land here, and they are not the same fact:

- `withheld` — it was going to be published and a rule pulled it. This one costs
  the issue an item. It is counted against the published total, because a rule
  that withholds a third of the day is not a filter, it is a different editorial
  policy adopted by accident.
- `near_miss` — it was never going to be published, but it sits close enough to
  the line that a judgement would be worth having. Costs the issue nothing.

Both are the same thing to a labeller, which is the point of the design: **the
held queue is the labelling queue is the training set.** It routes the rare
attention of one person at exactly the cases where the pipeline is least sure,
instead of at the top of a ranking it already gets right.

Held items are **not carried into the next issue**. They are not a backlog of
things owed to readers; they are waiting for a verdict. A held item that is
later judged `keep` tells us the rule is too wide — it does not get published
late.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

from . import paths
from .config import cfg, vocab_file
from .metrics import utcnow
from .models import Item

# --------------------------------------------------------------------------
# The suspicion rules
# --------------------------------------------------------------------------

# R1. The paper's own subfield is outside the ones the whitelist was built from.
#
# The journal path has no gate at all: membership of `journals.yaml` is the whole
# test, so a paper in a covered journal is published whatever it is about. The
# 08-11 top fifteen contained car insurance, V2G bus economics, Vietnamese green
# logistics, a shipping-lane CBA, nano-TiO2 pore structure, asphalt ageing and
# Arctic route cost competitiveness — and almost all of them are identified by
# the paper's **own** `primary_topic.subfield`, which we already collect and have
# never looked at. This is the cheapest hypothesis in L3-1 and it costs nothing
# to evaluate.
RULE_OFF_SUBFIELD = "off_subfield"

# ...and it is **enforced again**, against a list derived from articles.
#
# The original rule failed because of its list, not its logic: it asked whether
# a paper's subfield was in `openalex.whitelist_subfields`, which was built to
# choose *journals*. Measured, that lost 27 of 44 keeps — 61.4% — and took 59%
# and 79% of two backfilled days.
#
# `vocab/paper_subfields.yaml` is derived from our own labels instead
# (phase 0N, P2). Under it the same rule loses **6.8%** of keeps and withholds
# **0.2217** across the backfilled archive, both inside the pre-registered
# limits (10% and 0.30), so it removes items again.
#
# It is a weak gate by construction — it excludes only the four subfields our
# labels have seen at least three times and rejected more often than not. That
# is the point: its job is to stop taking our own subject matter, not to filter
# hard.
#
# Against the 75 labelled journal items, gating on the paper's own subfield
# loses **27 of 44 keeps (61.4%)** and leaves no day with enough labelled items
# in its top ten to compute a precision at all. Against two backfilled days it
# withheld 59% and 79% of the day. The papers it takes are the subject matter:
# GIS accessibility analysis, park climate adaptation, food deserts, urban heat
# risk, hurricane recovery of points of interest.
#
# The cause is not the rule's logic but its premise. `whitelist_subfields`
# (3305, 3313, 3322) was built to choose *journals*; OpenAlex scatters
# individual urban papers across 2215, 2307, 2214, 1110, 2213 and a long tail.
# A journal-selection list is the wrong instrument for judging an article, which
# is the same mistake the canon scope rule made in phase 0e with `journals.yaml`.
#
# So the rule still runs and still files what it finds — that is the labelling
# queue, and the queue is the point — but it no longer removes anything from an
# issue. One config line turns enforcement back on.

# R2. The classifier is closest to a coin flip here.
#
# Not a publication rule — items in this band are already below the 0.80 arXiv
# floor and are not published either way. They are held because this is where a
# judgement buys the most: the label file's own precision by band is near 0.5
# through here, which is another way of saying the model does not know. Holding
# them turns a silent drop into a question.
RULE_UNCERTAIN = "uncertain_score"

# R3. Scored right at the floor.
#
# Above the line by a margin smaller than the model's own calibration error is
# not meaningfully above the line.
RULE_AT_THE_FLOOR = "at_the_floor"

WITHHELD = "withheld"
NEAR_MISS = "near_miss"


def held_dir() -> Path:
    return paths.CONTENT / "held"


def held_path(d: date) -> Path:
    return held_dir() / f"{d}.json"


def _uncertain_band() -> tuple[float, float]:
    lo = float(cfg("held.uncertain_from", 0.50))
    hi = float(cfg("held.uncertain_to", 0.80))
    return lo, hi


def _floor_margin() -> float:
    return float(cfg("held.floor_margin", 0.03))


def _off_subfield_withholds() -> bool:
    """Whether R1 removes an item or merely files it. Default: files it.

    See RULE_OFF_SUBFIELD. Set `held.off_subfield_withholds: true` to enforce,
    once `whitelist_subfields` has been re-derived from articles rather than
    from journals — that is the fix, and it is YJUN's call (§N3).
    """
    return bool(cfg("held.off_subfield_withholds", False))


def enabled() -> bool:
    """`held.enabled` was declared in config and read by nothing."""
    return bool(cfg("held.enabled", True))


def whitelist_subfield_ids() -> set[str]:
    """The journal-selection list. Kept for reference; **not** the gate."""
    return {str(s) for s in (cfg("openalex.whitelist_subfields", []) or [])}


def rejected_subfield_ids() -> set[str]:
    """Subfields our own labels have seen enough of, and rejected.

    **A deny-list, not an allow-list, and the direction is the whole point.**

    The first attempt at this used the derived *inclusion* list — the 42
    subfields our labels have actually seen. That silently withheld every paper
    in a subfield the labels had never seen at all, which is the harshest
    possible treatment of the most complete absence of evidence, and the exact
    opposite of the rule it claimed to implement ("thin evidence is not evidence
    against"). A test asking about an unseen subfield caught it.

    We have evidence that **four** subfields are not ours — seen at least three
    times and kept less than half the time. We have no evidence about the
    hundreds we have never labelled, and acting on evidence we do not have is
    what `openalex.whitelist_subfields` did wrong in the first place.

    Empty if the file is missing, which means the rule holds nothing back —
    failing open, because a gate built on a list that failed to load should not
    quietly start rejecting things.
    """
    doc = vocab_file("paper_subfields.yaml") or {}
    return {
        str(entry["id"])
        for entry in (doc.get("excluded") or [])
        if isinstance(entry, dict) and entry.get("id")
    }


def paper_subfield(item: Item) -> Optional[str]:
    """The paper's own primary subfield id, not its venue's.

    The primary topic when one is flagged, else the highest-scoring one. This is
    the field the whole L3-1 hypothesis rests on and we have been collecting it
    since phase 0 without ever reading it.

    Returns None when OpenAlex has not classified the paper, and **None is not
    treated as off-subfield**: an unclassified paper is one we could not check,
    and holding it would be the measured-zero-versus-could-not-measure mistake
    this project keeps making.
    """
    topics = list(item.entities.topics or [])
    if not topics:
        return None
    primary = next((t for t in topics if getattr(t, "is_primary", False)), None)
    if primary is None:
        primary = max(topics, key=lambda t: float(getattr(t, "score", 0.0) or 0.0))
    sid = getattr(primary, "subfield", None)
    return str(sid).rsplit("/", 1)[-1] if sid else None


@dataclass
class Suspicion:
    """One reason to hold one item."""

    work_key: str
    rule: str
    kind: str
    detail: str
    score: Optional[float] = None
    source: Optional[str] = None
    title: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "work_key": self.work_key,
            "rule": self.rule,
            "kind": self.kind,
            "detail": self.detail,
            "score": self.score,
            "source": self.source,
            "title": self.title,
        }


def inspect(
    item: Item, source: str, selected: bool, floor: Optional[float] = None
) -> Optional[Suspicion]:
    """Whether this item is doubtful, and why. None means publish it as normal."""
    if not enabled():
        return None
    score = float(getattr(item.scores, "relevance", 0.0) or 0.0)
    floor = float(cfg("selection.arxiv_floor", 0.80)) if floor is None else floor
    title = item.bibliography.title or ""

    if source == "journal" and selected:
        rejected = rejected_subfield_ids()
        own = paper_subfield(item)
        if rejected and own and own in rejected:
            return Suspicion(
                work_key=item.work_key,
                rule=RULE_OFF_SUBFIELD,
                kind=WITHHELD if _off_subfield_withholds() else NEAR_MISS,
                detail=(
                    f"the paper's own subfield {own} is one our labels have "
                    f"seen and rejected {sorted(rejected)}"
                ),
                score=score,
                source=source,
                title=title,
            )

    if selected and source == "arxiv" and score < floor + _floor_margin():
        return Suspicion(
            work_key=item.work_key,
            rule=RULE_AT_THE_FLOOR,
            kind=WITHHELD,
            detail=f"{score:.3f} is within {_floor_margin()} of the {floor} floor",
            score=score,
            source=source,
            title=title,
        )

    if not selected and source == "arxiv":
        lo, hi = _uncertain_band()
        if lo <= score < hi:
            return Suspicion(
                work_key=item.work_key,
                rule=RULE_UNCERTAIN,
                kind=NEAR_MISS,
                detail=f"{score:.3f} sits in the {lo}–{hi} band where the model is least sure",
                score=score,
                source=source,
                title=title,
            )

    return None


# --------------------------------------------------------------------------
# Writing and reading the queue
# --------------------------------------------------------------------------


def over_warn_threshold(published: int, withheld: int) -> Optional[str]:
    """A day that withholds too much has replaced the editorial policy.

    59% and 79% were not issues, they were wreckage. Returns a sentence when the
    rate crosses `held.withheld_rate_warn`, and **nothing is blocked** — refusing
    to publish on this would lose the whole day instead of part of it. The point
    is that the drift becomes visible on the day it happens rather than three
    batches later.
    """
    denom = published + withheld
    if not denom:
        return None
    rate = withheld / denom
    limit = float(cfg("held.withheld_rate_warn", 0.30))
    if rate <= limit:
        return None
    return (
        f"held: withheld {withheld} of {denom} ({rate:.2%}), over the {limit:.0%} "
        f"line — the suspicion rules are acting as the editorial policy, not as "
        f"a filter"
    )


def record(d: date, suspicions: Iterable[Suspicion], published: int) -> Optional[Path]:
    """Write the day's held queue. No suspicions, no file."""
    rows = [s.as_dict() for s in suspicions]
    if not rows:
        return None

    held_dir().mkdir(parents=True, exist_ok=True)
    withheld = [r for r in rows if r["kind"] == WITHHELD]
    doc = {
        "date": str(d),
        # No timestamp. The held queue states what was doubtful *about a day*,
        # and re-running that day must produce the same file — `content/` being
        # byte-identical on a re-run is a PRD guarantee, and check 6 caught this
        # one moving. `runs_log` is different and keeps its timestamps because
        # it records *runs*, where a second attempt is a new fact.
        "published": published,
        "withheld": len(withheld),
        "near_miss": len(rows) - len(withheld),
        # The rate that says whether the rules are filters or a policy change.
        # Denominator is what the day would have published had nothing been
        # held, so it is comparable across days of different sizes.
        "withheld_rate": (
            round(len(withheld) / (published + len(withheld)), 4)
            if (published + len(withheld))
            else None
        ),
        "items": rows,
    }
    path = held_path(d)
    path.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def load(d: date) -> Optional[dict]:
    path = held_path(d)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def all_held() -> list[dict]:
    if not held_dir().exists():
        return []
    out = []
    for path in sorted(held_dir().glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def pending(since: Optional[date] = None) -> list[dict]:
    """Every held item not yet judged: withheld first, then oldest first.

    Oldest first within a kind, because a week away should be worked through in
    the order it happened. But **withheld before near-miss**, because they are
    not equally urgent: a withheld item cost an issue a slot and its judgement
    decides whether the rule that took it is right, while a near-miss cost
    nothing and only ever buys training signal.

    That ordering matters more than it looks. The first real day produced 7
    withheld and **36 near-misses**; a week away at that rate is 250 items, and
    a queue that has to be worked front to back would bury the seven that
    actually changed an issue under a fortnight of preprints that did not.
    """
    from .labeling import load_labels, superseded

    judged = {r.get("work_key") for r in superseded(load_labels("relevance"))}
    rows = []
    for day in all_held():
        if since and day["date"] < str(since):
            continue
        for row in day["items"]:
            if row["work_key"] in judged:
                continue
            rows.append({**row, "date": day["date"]})
    rows.sort(key=lambda r: (0 if r["kind"] == WITHHELD else 1, r["date"]))
    return rows


def counts() -> dict[str, Any]:
    """What the weekly summary reports: how much is waiting."""
    waiting = pending()
    return {
        "days_with_held_items": len(all_held()),
        "waiting": len(waiting),
        "withheld": sum(1 for r in waiting if r["kind"] == WITHHELD),
        "near_miss": sum(1 for r in waiting if r["kind"] == NEAR_MISS),
        "oldest": min((r["date"] for r in waiting), default=None),
    }
