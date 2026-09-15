"""The held queue — what we would not publish unattended (phase 0L, M2-2).

The operating assumption changed. Q4 used to ask whether a human could review a
day in fifteen minutes; it now asks whether the thing can run for a week with
nobody watching and still not publish something the editor would retract. Nobody
is checking before the mail goes out, so **the selection policy has to do the
job the daily review was doing**.

That makes conservatism the correct setting. When we are not sure, we publish
less. An item that a *withholding* rule trips is not published and not discarded
— it is held, and the day goes out with a hole where it would have been. **The
hole is the right answer**: a digest that fills its slots with things it is
unsure about is worth less than a shorter one, and a reader cannot see the
difference between a confident item and a slot that needed filling.

What that does not license is holding on doubt we have not measured. Each rule
has had to earn the right to leave a hole, and both rules that could have been
asked to earn it failed: `off_subfield` on the wrong list (0N, 0Q) and
`at_the_floor` on an argument that the calibration numbers turned out to
contradict (1H). Being unsure is the reason to *ask*, and asking is what this
file is for; it is not on its own a reason to publish less.

Two kinds of doubt land here, and they are not the same fact:

- `withheld` — it was going to be published and a rule pulled it. This one costs
  the issue an item. It is counted against the published total, because a rule
  that withholds a third of the day is not a filter, it is a different editorial
  policy adopted by accident.
- `near_miss` — it was never going to be published, but it sits close enough to
  the line that a judgement would be worth having. Costs the issue nothing.

**As of 1H every rule files rather than withholds, so `withheld` is 0 and stays
0 until a switch is turned back on.** That is the resting state of the queue,
not a fault in it: `off_subfield` has an empty deny-list (0Q) and
`at_the_floor` was demoted because the argument behind it did not survive being
measured (1H). The kinds stay, because the distinction is about what a
judgement is worth, and the rules can be switched back one config line each.

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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

from . import paths
from .config import cfg, vocab_file
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
# It used to read: *"Above the line by a margin smaller than the model's own
# calibration error is not meaningfully above the line."* **That argument is
# wrong, and it is left here rather than overwritten because the way it failed
# is the useful part.** It is kept as a quotation, not as a rule.
#
# It failed twice over, on the same measurement that was supposed to support it
# (D196, and see `held.floor_margin` in config):
#
#  - **The error at the top is under-confidence, not noise.** In [0.80,0.90) the
#    mean score is 0.861 and the observed keep rate is 1.00 (n=5). The argument
#    assumed the error was symmetric — that a 0.81 might really be a 0.78. It
#    points the other way at the top of the range, and withholding is the wrong
#    response to an item being *better* than its score says.
#  - **The withheld window was not measured when the margin was set.** Zero
#    relevance labels fall in [0.80,0.83): a margin justified by a calibration
#    curve was cutting at the one place the curve had no observations.
#
# It has been measured since, in a different frame, and the number is not the
# one the demotion was argued from. See `held.floor_margin` in
# `config/pipeline.yaml` for the 35 `held_review` judgements of 2026-08-20 —
# 16 keep, 19 drop — and for why they can neither be pooled with the relevance
# labels nor ignored.
#
# So the rule is **demoted, not deleted** (1H). It still runs and still files,
# because `content/held/` is the labelling queue and that unmeasured window can
# only be measured by keeping these items on record. What it no longer does is
# remove anything from an issue: arXiv is a single 0.80 floor again.
#
# `RULE_AT_THE_FLOOR` keeps its name. 79 days of `content/held/` files carry
# this string and renaming it would split the record in two.
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


def _at_the_floor_withholds() -> bool:
    """Whether R3 removes an item or merely files it. Default: files it.

    See RULE_AT_THE_FLOOR. The same switch R1 has, for the same reason and with
    the opposite history: R1 was demoted because its list was wrong and was
    restored once the list was rebuilt, while R3 is demoted because the
    *argument* was wrong. Restoring it would need a new argument, not a new
    number — the margin is not what failed.
    """
    return bool(cfg("held.at_the_floor_withholds", False))


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
    # `selection.arxiv.floor`, with the dot. This read `selection.arxiv_floor`
    # from phase 0L until 1H — a key that has never existed, so the lookup
    # always missed and always fell back to the literal below. It was invisible
    # because the two values happen to be equal and because `run_stages.py`
    # passes `floor=` explicitly on every real call. Moving the floor would have
    # silently left this path on 0.80.
    floor = float(cfg("selection.arxiv.floor", 0.80)) if floor is None else floor
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
            kind=WITHHELD if _at_the_floor_withholds() else NEAR_MISS,
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

    This is a one-sided test and has to stay one: since 1H `withheld` is 0 on a
    normal day, and 0 must produce silence. The zero denominator — a day that
    published nothing and withheld nothing — is silence for the same reason. A
    day with no issue is reported by the outcome record, not by a rate with
    nothing in it.
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

    # Both files, because either can settle an item: `held_review` is where
    # `--pending` writes, and a paper judged in the ranked relevance sample has
    # been judged whatever queue it also sat in. Asking again would be asking a
    # question we have the answer to.
    judged = {r.get("work_key") for r in superseded(load_labels("held_review"))}
    judged |= {r.get("work_key") for r in superseded(load_labels("relevance"))}
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


def _rule_can_fire(rule: str) -> bool:
    """Could this rule **withhold** something on the next run?

    Not "could it file something" — `at_the_floor` still fills the labelling
    queue every day it fires, and `uncertain_score` only ever did that. This
    asks the narrower question the `withheld` total depends on.

    Both switchable rules are off as of 1H, by different routes: `off_subfield`
    has nothing left to deny *and* is switched on, `at_the_floor` has plenty to
    catch and is switched off. `uncertain_score` sits below the floor and has
    never withheld anything, so it cannot go from live to inert — it was never
    live in this sense, and saying it can fire would put it in `inert_rules`'s
    complement as though it were holding the queue open.
    """
    if rule == RULE_OFF_SUBFIELD:
        return bool(rejected_subfield_ids()) and _off_subfield_withholds()
    if rule == RULE_AT_THE_FLOOR:
        return _at_the_floor_withholds()
    return True


def counts() -> dict[str, Any]:
    """What the weekly summary reports: how much is waiting, and **from which
    rule**.

    The per-rule breakdown was added in 0Q because one rule had come to own the
    whole withheld queue: the `off_subfield` deny-list emptied, leaving
    `at_the_floor` as the only rule that could take anything out of an issue. A
    single total hides that — three rules sharing a queue and one rule owning it
    are the same number and completely different situations.

    **1H made that count zero.** `at_the_floor` was demoted to `near_miss`, so
    no rule withholds anything today and `withheld` is expected to sit at 0
    while `near_miss` grows. Read the three fields accordingly:

    - `withheld: 0` is the **normal resting state**, not a broken queue. If it
      ever becomes an alert, that is the regression.
    - `withheld_by_one_rule` is None when no rule withholds *and* when several
      do. Only a single name means "one rule is the queue"; the absence of a
      name is not a diagnosis either way.
    - `inert_rules` names rules that hold items here and can withhold no more of
      them. `at_the_floor` is on that list now while remaining the busiest rule
      in the file — inert is about withholding, not about filing.
    """
    waiting = pending()
    by_rule: dict[str, dict[str, int]] = {}
    for row in waiting:
        bucket = by_rule.setdefault(row["rule"], {"withheld": 0, "near_miss": 0})
        bucket["withheld" if row["kind"] == WITHHELD else "near_miss"] += 1

    withheld = sum(1 for r in waiting if r["kind"] == WITHHELD)
    withheld_rules = sorted(
        (rule for rule, b in by_rule.items() if b["withheld"]),
        key=lambda rule: -by_rule[rule]["withheld"],
    )
    return {
        "days_with_held_items": len(all_held()),
        "waiting": len(waiting),
        "withheld": withheld,
        "near_miss": sum(1 for r in waiting if r["kind"] == NEAR_MISS),
        "oldest": min((r["date"] for r in waiting), default=None),
        "by_rule": dict(sorted(by_rule.items())),
        # A rule can hold items in the queue and be unable to withhold another
        # one. `off_subfield` is that as of 0Q — the deny-list is empty, so its
        # 81 withholdings are history rather than standing policy — and
        # `at_the_floor` is that as of 1H, by the switch rather than by running
        # out of things to catch. Left unmarked, past withholdings read as
        # standing policy.
        "inert_rules": sorted(r for r in by_rule if not _rule_can_fire(r)),
        # Named rather than left to be worked out from `by_rule`: this is the
        # fact that has to be noticeable without reading a table.
        #
        # None for zero rules as well as for several, and the two are not the
        # same thing — but neither of them is "one rule is the queue", which is
        # the only claim this field makes. Since 1H zero is the usual case.
        "withheld_by_one_rule": withheld_rules[0] if len(withheld_rules) == 1 else None,
    }
