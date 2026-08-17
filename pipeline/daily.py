"""`uc daily` — the one command that runs itself (phase 0k, X2).

Until now a human chose the date and chained the stages. This is the same
pipeline with three properties it did not need while a person was watching:

**It knows what to run.** The issue date is the day we publish, not the day the
papers did (X1: journal indexing is p50 1 day, p90 2 days, and no arXiv item has
ever been visible on its own publication day). So the window is
`[today - lookback, today - 1]`, and anything already published is skipped by
the existing published index rather than by date arithmetic.

**It cannot run twice at once.** A file lock with the holder's PID and start
time, and a reclaim path for the case that has already bitten this repo twice —
a lock left behind by a process that is gone.

**It resumes.** A run that dies after summarising has already spent the money;
the next attempt reuses every completed stage. Summaries are additionally cached
by prompt version and work key, so even a full re-run of a day pays nothing for
items it has already described.

And the property that matters most, which lives in `outcome.py`: **if it could
not see the day, it writes no issue.**
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from . import paths, store
from .config import cfg
from .metrics import Run
from .outcome import NOT_PUBLISHED, Outcome, decide, record

# Stages `uc daily` owns, in order. `collect` is handled separately because it
# takes the window.
STAGES = ("dedup", "gate", "enrich", "classify", "select", "link", "summarize", "score")


class DailyLocked(RuntimeError):
    """Another run holds the lock and is still alive."""


class BudgetExceeded(RuntimeError):
    """The day's LLM allowance ran out mid-run."""


# --------------------------------------------------------------------------
# Locking
# --------------------------------------------------------------------------


def lock_path() -> Path:
    return paths.RUNS / "daily.lock"


def _pid_alive(pid: int) -> bool:
    """Is that process still there? Windows and POSIX differ; both are handled."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import subprocess

            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False
    except Exception:  # noqa: BLE001 - an unanswerable question is not a live process
        return False


@dataclass
class Lock:
    path: Path
    acquired: bool
    reclaimed_from: Optional[dict] = None

    def release(self) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink(missing_ok=True)


def acquire_lock(stale_after_s: int = 6 * 3600) -> Lock:
    """Take the daily lock, reclaiming one whose owner is gone.

    Two lock files left by dead git processes have already cost this project
    time (phase 0i found `.git/index.lock` and `.git/HEAD.lock` from a crash two
    days earlier). A lock with no liveness information is a lock that eventually
    wedges the thing it protects, so this one carries the PID and the start
    time, and says in the log which of the two let it reclaim.
    """
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            holder = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            holder = {}
        pid = int(holder.get("pid", 0) or 0)
        started = float(holder.get("started_at", 0) or 0)
        age = time.time() - started if started else None

        if _pid_alive(pid):
            if age is not None and age > stale_after_s:
                # Alive but far past any plausible run: reclaiming would risk two
                # writers, so this refuses and says why rather than guessing.
                raise DailyLocked(
                    f"pid {pid} still alive and holding the lock for "
                    f"{age / 3600:.1f}h — inspect it before running again"
                )
            raise DailyLocked(f"another run is in progress (pid {pid})")

        reclaimed = {"pid": pid, "age_hours": round(age / 3600, 2) if age else None}
        path.unlink(missing_ok=True)
    else:
        reclaimed = None

    path.write_text(
        json.dumps({"pid": os.getpid(), "started_at": time.time()}, indent=2),
        encoding="utf-8",
    )
    return Lock(path=path, acquired=True, reclaimed_from=reclaimed)


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------


def target_window(today: Optional[date] = None) -> tuple[date, date]:
    """The publication-date window a run started today should collect.

    From X1: journals reach 95% coverage at D+2 and arXiv is never visible on
    D+0. The default lookback is 3 days, which covers the p99 for neither — it
    covers the mass, and the tail is caught by re-collection on later days,
    because an item already published is skipped rather than published twice.
    """
    today = today or date.today()
    lookback = int(cfg("daily.lookback_days", 3))
    end = today - timedelta(days=int(cfg("daily.min_lag_days", 1)))
    return end - timedelta(days=lookback - 1), end


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _spend_since(baseline: float) -> float:
    from .llm import UsageState

    return round(UsageState.load().cost_usd - baseline, 6)


def run_daily(
    d: Optional[date] = None,
    dry_run: bool = False,
    use_llm: bool = True,
    today: Optional[date] = None,
) -> dict[str, Any]:
    """One day, end to end, with the outcome decided rather than assumed."""
    from .llm import UsageState
    from .run_stages import (
        StageSkipped,
        stage_collect,
        stage_issue,
        stage_preview,
        _guard,
    )
    from . import run_stages

    issue_date = d or (today or date.today())
    covers_from, covers_to = target_window(today)
    budget_cap = float(cfg("daily.max_llm_usd", 1.0))
    baseline_spend = UsageState.load().cost_usd

    lock = acquire_lock()
    started = time.monotonic()
    run = Run.for_date(issue_date)
    if lock.reclaimed_from:
        run.error(
            f"daily: reclaimed a stale lock from pid "
            f"{lock.reclaimed_from['pid']} (dead)"
        )

    budget_exceeded = False
    try:
        # Collect the window. `backfill_from` already exists for exactly this.
        _guard(run, "collect", lambda: stage_collect(
            run, covers_to, backfill_from=covers_from
        ))

        for name in STAGES:
            if name == "summarize":
                spent = _spend_since(baseline_spend)
                if spent >= budget_cap:
                    budget_exceeded = True
                    run.error(
                        f"daily: LLM budget of ${budget_cap:.2f} reached before "
                        f"summarize (${spent:.4f} spent); not publishing"
                    )
                    break
            fn = getattr(run_stages, f"stage_{name}")
            if name == "summarize":
                _guard(run, name, lambda: fn(run, use_llm=use_llm))
            elif name in ("dedup",):
                _guard(run, name, lambda: fn(run, covers_to))
            else:
                _guard(run, name, lambda: fn(run))

        spent = _spend_since(baseline_spend)
        if spent > budget_cap:
            budget_exceeded = True
            run.error(
                f"daily: LLM spend ${spent:.4f} exceeded the ${budget_cap:.2f} cap"
            )

        # The verdict comes before the issue, never after.
        selected = run_stages.read_stage(run, "score") or []
        outcome = decide(run, issue_date, len(selected), budget_exceeded=budget_exceeded)

        if outcome.status == NOT_PUBLISHED or dry_run:
            if dry_run and outcome.status != NOT_PUBLISHED:
                outcome.reasons.append("dry run: nothing written")
            record(outcome)
            alert = None
            if outcome.status == NOT_PUBLISHED and not dry_run:
                # Recorded first, then announced: the log is the fact and the
                # mail is only a copy of it. An alert that cannot be sent leaves
                # the record intact (X7).
                from .notify import notify_failure

                alert = notify_failure(issue_date, outcome.reasons, run=run)
            run.metrics.timing["daily_s"] = round(time.monotonic() - started, 1)
            run.save()
            result = _result(outcome, run, started, covers_from, covers_to, dry_run)
            if alert is not None:
                result["alert"] = alert
            return result

        issue = stage_issue(run, issue_date)
        issue.covers_from = covers_from
        issue.covers_to = covers_to
        store.save_issue(issue)
        try:
            stage_preview(run, issue_date)
        except StageSkipped:
            pass

        outcome.published = len(issue.items)
        outcome.status = "quiet" if issue.quiet_day else "published"
        record(outcome)

        delivery = _deliver_issue(run, issue)
        run.metrics.timing["daily_s"] = round(time.monotonic() - started, 1)
        run.save()
        result = _result(outcome, run, started, covers_from, covers_to, dry_run)
        result["delivery"] = delivery
        return result
    finally:
        lock.release()


def catch_up(today: Optional[date] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Retry the days we could not see, oldest first.

    A scheduled run that only ever attempts today leaves every outage permanent:
    the morning the API was down stays blank forever, and the archive's gap is
    then a fact about our uptime rather than about the field. So each run also
    reaches back over `daily.catch_up_days` and retries what is still missing.

    Oldest first, because a retry of the 3rd that succeeds may satisfy the
    coupling window used by the 4th. And bounded, because a day whose sources
    have moved on cannot be recovered by asking again — past the horizon a
    missed day stays missed, and stays in the log saying so.
    """
    from .outcome import unpublished_dates

    today = today or date.today()
    horizon = today - timedelta(days=int(cfg("daily.catch_up_days", 7)))

    pending = [
        row for row in unpublished_dates()
        if horizon <= date.fromisoformat(row["date"]) < today
    ]
    pending.sort(key=lambda r: r["date"])
    if limit:
        pending = pending[:limit]

    results = []
    for row in pending:
        d = date.fromisoformat(row["date"])
        try:
            results.append(run_daily(d=d, today=today))
        except DailyLocked:
            raise
        except Exception as e:  # noqa: BLE001 - one bad day must not stop the rest
            results.append({
                "date": row["date"],
                "status": "retry_failed",
                "error": f"{type(e).__name__}: {e}",
            })
    return results


def _deliver_issue(run: Run, issue) -> dict[str, Any]:
    """Send the issue, and never let a send failure cost the issue.

    The issue is already written and already on the site by this point. A
    delivery that fails is a delivery to retry, not a reason to unwind a day's
    work — so this catches, records, and lets the run finish reporting success
    for the part that succeeded.
    """
    from .deliver import Message, DeliveryError, deliver, get_backend, recipients
    from .render.plaintext import render_text
    from .render.preview import email_subject, render_issue
    from .render.inline import to_email

    try:
        items = [it for it in (store.load_item(k) for k in issue.items) if it]
        unreadable = [it for it in (store.load_item(k) for k in issue.unreadable) if it]
        message = Message(
            subject=email_subject(issue),
            html=to_email(render_issue(issue, items, unreadable)),
            text=render_text(issue, items, unreadable),
            issue_date=issue.date,
            recipients=recipients(),
        )
        return deliver(issue.date, message, backend=get_backend())
    except (DeliveryError, OSError) as e:
        run.error(f"deliver: {type(e).__name__}: {e}")
        return {"status": "failed", "error": f"{type(e).__name__}: {e}"}


def _result(
    outcome: Outcome,
    run: Run,
    started: float,
    covers_from: date,
    covers_to: date,
    dry_run: bool,
) -> dict[str, Any]:
    from .llm import UsageState

    usage = UsageState.load()
    return {
        "date": str(outcome.date),
        "status": outcome.status,
        "covers_from": str(covers_from),
        "covers_to": str(covers_to),
        "reasons": outcome.reasons,
        "failed_stages": outcome.failed_stages,
        "candidates": outcome.candidates,
        "published": outcome.published,
        "dry_run": dry_run,
        "seconds": round(time.monotonic() - started, 1),
        "llm_calls_total": usage.calls,
        "llm_cost_total_usd": round(usage.cost_usd, 6),
    }
