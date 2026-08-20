"""What a day's run concluded, and what earns the right to say "quiet" (X3).

**A failed day is not a quiet day.** This module exists for that one sentence.

The service's only real asset is that when it says a day was quiet, it looked.
"Quiet days are declared, not padded" is a claim about *having seen*, and an
issue written on a day when OpenAlex returned 500s says we saw nothing and
therefore there was nothing — which is the same shape of lie as inventing a
paper, told with a straight face.

Phase 0h produced exactly that file. `content/issues/2026-08-14.json`: zero
items, `quiet_day: true`, `candidates_scanned: 0`. Nothing in the data
distinguishes it from a genuinely thin Tuesday. It now lives in
`content/_retired/` as the marker for where this changed.

Three outcomes, and the middle one has to be earned:

| outcome | meaning | issue file | email |
|---|---|---|---|
| `published` | we looked, and there was something | written | sent |
| `quiet` | **we looked**, and there was almost nothing | written | sent |
| `not_published` | **we did not look** | none | none, alert instead |

## What proves we looked

A `quiet` claim requires all of:

1. **Every required source finished successfully.** Not "did not raise" —
   finished, with the stage recording OK. A source that was skipped for a
   missing key has not looked either.
2. **A candidate population was actually counted**, even if the count is 0.
   Counting zero and failing to count are different facts and the difference is
   the whole point of this module.
3. **No stage failed**, including the ones after collection. A day summarised
   half way is not a quiet day, it is an incomplete one.
4. **The budget held.** A run stopped by the spend cap saw only part of the day,
   so what it did not see is unknown rather than absent.

Anything short of that is `not_published`, which writes no issue, sends no
email, and leaves a row in `content/runs_log/` saying which of the four failed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from . import paths
from .metrics import Run, utcnow

PUBLISHED = "published"
QUIET = "quiet"
NOT_PUBLISHED = "not_published"

# A fourth state, and it is not a fourth verdict (hotfix 2, H5-2).
#
# `not_published` is a conclusion the pipeline reached: it looked, something was
# wrong, it said so. `interrupted` means **no conclusion was reached at all** —
# the process was killed before it could decide. CI hitting `timeout-minutes`
# produced exactly this and recorded nothing, so the day was missing from
# `runs_log` entirely and looked identical to a day the schedule never fired on.
#
# Kept apart from `not_published` because the remedy differs. A `not_published`
# day is retried and usually succeeds. An `interrupted` day says the run does
# not fit in the time it was given, and retrying it unchanged will do the same
# thing again.
INTERRUPTED = "interrupted"

# Sources that must answer before a day can be called quiet. Both, because the
# issue claims a scope — "what appeared in urban data science today" — and half
# that scope is a different claim.
REQUIRED_SOURCES = ("collect.arxiv", "collect.openalex")

# The count each source writes when it succeeds. Used to catch the failure that
# reports success: a source that finishes OK and contributes nothing.
SOURCE_COUNTS = {"collect.arxiv": "arxiv_fetched", "collect.openalex": "openalex_fetched"}

# ★ Stages whose absence makes the day unpublishable (phase 0U, U1).
#
# `looked()` only ever inspected `FAILED`, and `StageSkipped` is a *different*
# outcome by design: `skips.py` draws the line at "a stage that cannot run for a
# reason that is not a failure — no key, no model". That line is right, and it
# had a hole in it. When the LLM is unavailable the summarize stage returns
# `SKIPPED`, no reason is added, the verdict is `published`, and the issue goes
# out with every card reading **"Summary pending review."**
#
# So the distinction is not skipped-versus-failed; it is **which stage**. Some
# stages are optional by nature and some are the product.
#
#   `summarize`  Required. A digest of unread summaries is not this product.
#                It is the single thing a reader is here for.
#   `classify`   Required. Without it the arXiv path has no score, so the
#                floor cannot be applied and "what appeared" becomes "what we
#                happened to fetch".
#   `select`     Required. Without it nothing decided what the issue contains,
#                and U4 shows what fills the gap: the whole candidate pool.
#   `issue`      Required. It is the artefact.
#
# Deliberately **not** required, and each for a stated reason:
#
#   `enrich.springer`  A key nobody has is the normal state, not a fault.
#   `link` / `extract` Overlay tags enrich a card and no card depends on them.
#   `synthesis`        Already designed to be absent; silence is a finding.
#   `canon`            0T's chore. It must never decide a day (D223).
#   `preview`          Rendering, downstream of the verdict.
#
# `PARTIAL` is not refused. A day where some items got summaries is a day worth
# publishing — the rule is about a stage that produced *nothing*.
REQUIRED_STAGES = ("summarize", "classify", "select", "issue")


@dataclass
class Outcome:
    """One day's verdict, with the reasons that produced it."""

    date: date
    status: str
    reasons: list[str] = field(default_factory=list)
    failed_stages: list[str] = field(default_factory=list)
    # Recorded beside the failures because a required stage being skipped is
    # now a verdict-changing fact and the log has to show which one (U1).
    skipped_stages: list[str] = field(default_factory=list)
    candidates: Optional[int] = None
    published: int = 0
    attempts: int = 1
    spend_usd: float = 0.0
    silent_sources: list[str] = field(default_factory=list)
    # ★ Whether the day had a headline (phase 0U, U5).
    #
    # `status` and this are **two different facts** and were being written into
    # one field. `decide()` grants `quiet` for "published nothing"; `daily.py`
    # then overwrote it with `issue.quiet_day`, which means "nothing cleared the
    # headline threshold". 2026-08-09 is the proof: `"published": 11` and
    # `"status": "quiet"` in the same row — eleven papers on a day the log calls
    # silent. That value feeds `uc weekly`, `uc status` and the archive strip.
    #
    # The headline fact is not discarded; `quiet_day` is what the renderer keys
    # off and it is real. It just stops pretending to be the verdict.
    headline_present: Optional[bool] = None

    @property
    def writes_issue(self) -> bool:
        return self.status in (PUBLISHED, QUIET)

    @property
    def sends_email(self) -> bool:
        return self.status in (PUBLISHED, QUIET)

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": str(self.date),
            "status": self.status,
            "reasons": self.reasons,
            "failed_stages": self.failed_stages,
            "skipped_stages": self.skipped_stages,
            "candidates": self.candidates,
            "published": self.published,
            "attempts": self.attempts,
            "spend_usd": round(self.spend_usd, 6),
            "silent_sources": self.silent_sources,
            "headline_present": self.headline_present,
            "recorded_at": utcnow().isoformat(),
        }


def looked(run: Run, required: tuple[str, ...] = REQUIRED_SOURCES) -> tuple[bool, list[str]]:
    """Did this run actually observe the day? Returns (verdict, reasons against).

    Deliberately strict and deliberately explicit: every reason it says no is a
    sentence a human can check against the run's own metrics.
    """
    reasons: list[str] = []
    stages = run.metrics.stages

    for source in required:
        status = stages.get(source)
        if status is None:
            reasons.append(f"{source} did not run")
        elif status not in ("OK",):
            reasons.append(f"{source} finished {status}")

    collect = stages.get("collect")
    if collect == "EMPTY":
        # The collect stage says EMPTY when it got zero candidates. That is not
        # by itself a failure — but it is the exact case where "quiet" and
        # "blind" look alike, so it is only allowed through when every required
        # source above reported OK.
        if reasons:
            reasons.append("collect returned nothing and a source had failed")
    elif collect != "OK":
        reasons.append(f"collect finished {collect or 'not at all'}")

    failed = [name for name, status in stages.items() if status == "FAILED"]
    if failed:
        reasons.append(f"stages failed: {', '.join(sorted(failed))}")

    # A required stage that did not run at all, or ran and produced nothing, is
    # as disqualifying as one that raised. See `REQUIRED_STAGES`.
    for stage in REQUIRED_STAGES:
        status = stages.get(stage)
        if status is None:
            reasons.append(f"{stage} did not run")
        elif status == "SKIPPED":
            reasons.append(
                f"{stage} was skipped, and an issue without it is not the "
                f"product — check that the keys and model it needs are present"
            )

    if getattr(run.metrics, "budget_exceeded", False):
        reasons.append("the daily LLM budget stopped the run part-way")

    return (not reasons), reasons


def silent_sources(run: Run, window_days: int = 1) -> list[str]:
    """Sources that reported OK and returned nothing anyway.

    The failure that reports success. A run covering a week fetched zero arXiv
    items with every stage green, because the window sat inside arXiv's indexing
    lag — a whole half of our declared scope missing, and nothing in the data
    saying so. Over a single day zero is ordinary; over a window it is evidence
    that the window is wrong, that the source is down, or that our query is.

    This does **not** change the verdict. Journal items collected on a day arXiv
    was unreachable are real papers, and withholding them would trade a partial
    issue for none. But it is recorded on every run, so the next time a source
    goes quiet it is a line in the log rather than a slow drift nobody notices.
    """
    if window_days < 2:
        return []

    out = []
    for source in REQUIRED_SOURCES:
        if run.metrics.stages.get(source) != "OK":
            continue  # already counted as a failure, not as silence
        field = SOURCE_COUNTS.get(source)
        if field and not int(getattr(run.metrics.counts, field, 0) or 0):
            out.append(source)
    return out


def decide(
    run: Run,
    d: date,
    published_count: int,
    budget_exceeded: bool = False,
    window_days: int = 1,
) -> Outcome:
    """The day's verdict. The only place `quiet` can be granted."""
    if budget_exceeded:
        setattr(run.metrics, "budget_exceeded", True)

    ok, reasons = looked(run)
    silent = silent_sources(run, window_days)
    if silent:
        # Loud in the log, and loud in the run's own errors, but not a veto —
        # see `silent_sources` for why the verdict is left alone.
        run.error(
            f"outcome: {', '.join(silent)} reported OK and returned nothing over "
            f"a {window_days}-day window"
        )
    counts = run.metrics.counts
    candidates = None
    if ok or run.metrics.stages.get("collect") in ("OK", "EMPTY"):
        candidates = int(getattr(counts, "arxiv_candidates", 0) or 0) + int(
            getattr(counts, "journal_candidates", 0) or 0
        )

    failed = sorted(n for n, s in run.metrics.stages.items() if s == "FAILED")
    skipped = sorted(n for n, s in run.metrics.stages.items() if s == "SKIPPED")

    if not ok:
        return Outcome(
            date=d,
            status=NOT_PUBLISHED,
            reasons=reasons,
            failed_stages=failed,
            skipped_stages=skipped,
            candidates=candidates,
            published=0,
            spend_usd=float(run.metrics.cost.total_usd or 0.0),
            silent_sources=silent,
        )

    status = PUBLISHED if published_count else QUIET
    return Outcome(
        date=d,
        status=status,
        reasons=["nothing cleared the bar"] if status == QUIET else [],
        failed_stages=[],
        skipped_stages=skipped,
        candidates=candidates,
        published=published_count,
        spend_usd=float(run.metrics.cost.total_usd or 0.0),
        silent_sources=silent,
    )


# --------------------------------------------------------------------------
# The log of days with no issue
# --------------------------------------------------------------------------


def log_dir() -> Path:
    return paths.CONTENT / "runs_log"


def log_path(d: date) -> Path:
    return log_dir() / f"{d}.json"


def record(outcome: Outcome) -> Path:
    """Every outcome is logged, including the successful ones.

    Logging only failures would make the log's own silence ambiguous — the same
    ambiguity this module exists to remove one level up.
    """
    log_dir().mkdir(parents=True, exist_ok=True)
    path = log_path(outcome.date)

    previous = None
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = None
    if previous:
        outcome.attempts = int(previous.get("attempts", 1)) + 1

    path.write_text(
        json.dumps(outcome.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def load_log(d: date) -> Optional[dict]:
    path = log_path(d)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def all_logs() -> list[dict]:
    if not log_dir().exists():
        return []
    out = []
    for path in sorted(log_dir().glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def unpublished_dates(limit: Optional[int] = None) -> list[dict]:
    """Days we could not see, newest first — the catch-up queue."""
    rows = [r for r in all_logs() if r.get("status") == NOT_PUBLISHED]
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows[:limit] if limit else rows


def record_interrupted(d: date, reason: str) -> Path:
    """Write the row a killed run could not write itself.

    Called by the workflow's `always()` step when no row exists for the day.
    Deliberately dumb — it takes no measurements and reaches no verdict, because
    by definition nobody knows what happened. It asserts one fact: **something
    started and never finished**, which is strictly more than the silence it
    replaces.

    Never overwrites. If the pipeline did manage to record a verdict, that
    verdict is better than this one and stands.
    """
    existing = load_log(d)
    if existing:
        return log_path(d)

    return record(
        Outcome(
            date=d,
            status=INTERRUPTED,
            reasons=[reason],
            candidates=None,
            published=0,
        )
    )


def interrupted_dates(limit: Optional[int] = None) -> list[dict]:
    rows = [r for r in all_logs() if r.get("status") == INTERRUPTED]
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows[:limit] if limit else rows
