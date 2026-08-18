"""Filling the archive — one issue per past day (phase 0L, N1).

Almost every second-order measurement in the last four batches ended in "too
few": coupling anchors n=3, canon lines n=2, review articles n=3, `tag shift`
with no row on any of five days. **The cause is one thing — the archive is five
days long.** Ninety days of candidates are on disk and only five of them were
ever made into issues, so everything that could be built has been built and
almost nothing can be measured.

This walks backwards through the calendar making the missing issues.

## Four rules, and the reasons

**One-day windows.** A live run covers seven days because arXiv indexes three
days late and we must not miss things. A historical issue has no such problem —
the candidates are already collected — and dating it by a seven-day sweep would
make it claim papers that belong to its neighbours. `covers_from == covers_to`,
and `backfilled: true` so no aggregate mixes the two kinds without saying so.

**Oldest first.** If the budget runs out half way, an unbroken run of days is
worth more than sixty days with holes in it: every window-based measurement
(coupling, tag shift, first-internal-citation) needs *consecutive* days.

**Never rewrite.** The five existing issues and anything CI has published are
skipped outright. They are immutable, and the journal count they carry (159)
is a fact about the day they were made.

**Checkpoint every day, commit every ten.** A run that dies at day 41 must not
start again at day 1, and a single commit of sixty days is a revert that is all
or nothing.

## Two budgets, and why the wall clock is not one of them

The spend budget is real and enforced here: the batch ceiling is $9.00 and the
run stops when it is reached, reporting how far it got.

The **wall clock is deliberately not inherited**. `daily.max_minutes` is 30,
sized for a single live run, and a backfill is expected to take hours —
Gemini's free tier is roughly 10-15 requests a minute and this needs about
3,270 calls, so most of the elapsed time is spent *waiting for permission to
ask*. Applying a one-day deadline here would kill the job for succeeding
slowly. The backfill therefore runs on its own clock, and the throttling it
meets is measured (`llm_rate_limited`, `llm_backoff_s`) rather than treated as
failure — that measurement is the evidence for whether a paid tier is needed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from . import paths, store
from .config import cfg

CHECKPOINT = "backfill_issues.json"

# Days a live run owns. The backfill never touches today or the recent window,
# because those are CI's to publish and two machines writing one date is the
# conflict this project has already had once.
LIVE_MARGIN_DAYS = 8


def checkpoint_path() -> Path:
    return paths.STATE / CHECKPOINT


def load_checkpoint() -> dict:
    p = checkpoint_path()
    if not p.exists():
        return {"done": [], "skipped": [], "spend_usd": 0.0, "calls": 0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"done": [], "skipped": [], "spend_usd": 0.0, "calls": 0}


def save_checkpoint(state: dict) -> None:
    p = checkpoint_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def existing_issue_dates() -> set[str]:
    directory = paths.CONTENT / "issues"
    if not directory.exists():
        return set()
    return {p.stem for p in directory.glob("*.json")}


def target_dates(days: int, today: Optional[date] = None) -> list[date]:
    """The days to fill, oldest first.

    Ends `LIVE_MARGIN_DAYS` back so the backfill and the live pipeline never
    reach for the same date.
    """
    today = today or date.today()
    end = today - timedelta(days=LIVE_MARGIN_DAYS)
    return [end - timedelta(days=i) for i in range(days - 1, -1, -1)]



def _throttling_for(day: date) -> tuple[int, float]:
    """(times throttled, seconds slept) for one day's run, from its metrics."""
    from .metrics import Run

    try:
        run = Run.for_date(day)
    except Exception:  # noqa: BLE001
        return 0, 0.0
    counts = run.metrics.counts
    return (
        int(getattr(counts, "llm_rate_limited", 0) or 0),
        float(run.metrics.timing.get("llm_backoff_s", 0.0) or 0.0),
    )

@dataclass
class BackfillResult:
    attempted: int = 0
    published: int = 0
    quiet: int = 0
    not_published: int = 0
    skipped: int = 0
    spend_usd: float = 0.0
    calls: int = 0
    rate_limited: int = 0
    backoff_s: float = 0.0
    seconds: float = 0.0
    stopped_on: Optional[str] = None
    days: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "published": self.published,
            "quiet": self.quiet,
            "not_published": self.not_published,
            "skipped": self.skipped,
            "spend_usd": round(self.spend_usd, 6),
            "calls": self.calls,
            "rate_limited": self.rate_limited,
            "backoff_s": round(self.backoff_s, 1),
            "minutes": round(self.seconds / 60, 1),
            "stopped_on": self.stopped_on,
            "days": self.days,
        }


def backfill(
    days: int = 60,
    budget_usd: Optional[float] = None,
    today: Optional[date] = None,
    on_checkpoint: Optional[Callable[[int, dict], None]] = None,
    commit_every: int = 10,
    runner=None,
) -> BackfillResult:
    """Make the missing issues, oldest first, within the spend budget."""
    from .daily import run_daily
    from .llm import UsageState

    budget = float(budget_usd if budget_usd is not None else cfg("backfill.max_usd", 9.0))
    state = load_checkpoint()
    done = set(state.get("done") or [])
    skipped = set(state.get("skipped") or [])
    existing = existing_issue_dates()

    # The budget is for the whole backfill, not for one pass of it.
    #
    # A sixty-day run outlives a single process, so it is driven in passes that
    # each resume from the checkpoint. Measuring spend from "cumulative cost
    # when this pass started" gives every pass a fresh ceiling — fifteen passes,
    # fifteen budgets. Caught when the checkpoint reported *less* spend after a
    # day was added than before it.
    #
    # So the baseline is written once, on the first pass, and every later pass
    # measures against the same number.
    usage_now = UsageState.load()
    baseline_key = "baseline_cost_usd"
    if baseline_key not in state:
        state[baseline_key] = usage_now.cost_usd
        state.setdefault("baseline_calls", usage_now.calls)
        save_checkpoint(state)
    start_spend = float(state[baseline_key])
    start_calls = int(state.get("baseline_calls", usage_now.calls))

    result = BackfillResult()
    started = time.monotonic()
    run_one = runner or run_daily

    for day in target_dates(days, today):
        key = str(day)
        if key in done:
            continue
        if key in existing or key in skipped:
            # Immutable: the five original issues and anything CI published.
            result.skipped += 1
            skipped.add(key)
            continue

        usage = UsageState.load()
        spent = usage.cost_usd - start_spend
        if spent >= budget:
            result.stopped_on = (
                f"spend budget reached: ${spent:.4f} of ${budget:.2f} after "
                f"{result.attempted} day(s)"
            )
            break

        # Its own clock. `daily.max_minutes` is 30, sized for one live run, and
        # a backfill day that is being throttled can legitimately take longer
        # than that — killing it would be punishing the run for succeeding
        # slowly. Not `install()`ed: replacing the signal handler once per day
        # for sixty days is churn, and a local backfill is stopped with Ctrl-C.
        from .daily import Deadline

        outcome = run_one(
            d=day,
            window=(day, day),
            backfilled=True,
            today=today,
            deadline=Deadline(minutes=float(cfg("backfill.max_minutes_per_day", 60))),
        )
        result.attempted += 1
        status = (outcome or {}).get("status")
        if status == "published":
            result.published += 1
        elif status == "quiet":
            result.quiet += 1
        else:
            result.not_published += 1

        done.add(key)

        # Throttling is read back out of the day's own metrics rather than
        # tracked here, because the client that did the waiting is created and
        # discarded inside the stage. Summed across the backfill it is the
        # measured answer to "does this need a paid tier".
        throttled, slept = _throttling_for(day)
        result.rate_limited += throttled
        result.backoff_s += slept

        result.days.append({
            "date": key,
            "status": status,
            "published": (outcome or {}).get("published"),
            "candidates": (outcome or {}).get("candidates"),
            "rate_limited": throttled,
            "backoff_s": round(slept, 1),
        })

        usage = UsageState.load()
        state = {
            baseline_key: start_spend,
            "baseline_calls": start_calls,
            "done": sorted(done),
            "skipped": sorted(skipped),
            "spend_usd": round(usage.cost_usd - start_spend, 6),
            "calls": usage.calls - start_calls,
            "updated": key,
        }
        save_checkpoint(state)

        if on_checkpoint and result.attempted % commit_every == 0:
            on_checkpoint(result.attempted, state)

    usage = UsageState.load()
    result.spend_usd = usage.cost_usd - start_spend
    result.calls = usage.calls - start_calls
    result.seconds = time.monotonic() - started
    state = {
        baseline_key: start_spend,
        "baseline_calls": start_calls,
        "done": sorted(done),
        "skipped": sorted(skipped),
        "spend_usd": round(result.spend_usd, 6),
        "calls": result.calls,
    }
    save_checkpoint(state)
    return result
