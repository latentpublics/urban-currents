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

# Sources that must answer before a day can be called quiet. Both, because the
# issue claims a scope — "what appeared in urban data science today" — and half
# that scope is a different claim.
REQUIRED_SOURCES = ("collect.arxiv", "collect.openalex")


@dataclass
class Outcome:
    """One day's verdict, with the reasons that produced it."""

    date: date
    status: str
    reasons: list[str] = field(default_factory=list)
    failed_stages: list[str] = field(default_factory=list)
    candidates: Optional[int] = None
    published: int = 0
    attempts: int = 1
    spend_usd: float = 0.0

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
            "candidates": self.candidates,
            "published": self.published,
            "attempts": self.attempts,
            "spend_usd": round(self.spend_usd, 6),
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

    if getattr(run.metrics, "budget_exceeded", False):
        reasons.append("the daily LLM budget stopped the run part-way")

    return (not reasons), reasons


def decide(run: Run, d: date, published_count: int, budget_exceeded: bool = False) -> Outcome:
    """The day's verdict. The only place `quiet` can be granted."""
    if budget_exceeded:
        setattr(run.metrics, "budget_exceeded", True)

    ok, reasons = looked(run)
    counts = run.metrics.counts
    candidates = None
    if ok or run.metrics.stages.get("collect") in ("OK", "EMPTY"):
        candidates = int(getattr(counts, "arxiv_candidates", 0) or 0) + int(
            getattr(counts, "journal_candidates", 0) or 0
        )

    failed = sorted(n for n, s in run.metrics.stages.items() if s == "FAILED")

    if not ok:
        return Outcome(
            date=d,
            status=NOT_PUBLISHED,
            reasons=reasons,
            failed_stages=failed,
            candidates=candidates,
            published=0,
            spend_usd=float(run.metrics.cost.total_usd or 0.0),
        )

    status = PUBLISHED if published_count else QUIET
    return Outcome(
        date=d,
        status=status,
        reasons=["nothing cleared the bar"] if status == QUIET else [],
        failed_stages=[],
        candidates=candidates,
        published=published_count,
        spend_usd=float(run.metrics.cost.total_usd or 0.0),
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
