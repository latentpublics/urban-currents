"""`uc daily` — the one command that runs itself (phase 0k, X2).

Until now a human chose the date and chained the stages. This is the same
pipeline with three properties it did not need while a person was watching:

**It knows what to run.** The issue date is the day we publish, not the day the
papers did (X1: journal indexing is p50 1 day, p90 2 days, and no arXiv item has
ever been visible on its own publication day). So the window is
`[today - lookback, today - 1]`, and anything already published is skipped by
the existing published index rather than by date arithmetic. The lookback is set
by the slower source — see `target_window`, and the measurement that caught a
window sitting inside arXiv's indexing lag.

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


class TimeBudgetExceeded(RuntimeError):
    """The run took longer than `daily.max_minutes` and stopped itself."""


class Interrupted(RuntimeError):
    """SIGTERM arrived — the platform is about to kill us."""


# --------------------------------------------------------------------------
# Stopping before something else stops us (hotfix 2, H5)
# --------------------------------------------------------------------------


class Deadline:
    """A wall clock the run checks between stages, plus a SIGTERM handler.

    The failure this exists for: CI killed the job at `timeout-minutes: 45` and
    **nothing was recorded**. No run-log row, no alert, no commit. To `uc status`
    the day simply did not exist, which is indistinguishable from the schedule
    never having fired — X3's whole problem, reappearing one level up, because
    **a pipeline killed by the platform cannot record its own death.**

    Two defences, and they cover different failures:

    - The **deadline** is ours. Between stages we ask whether there is time for
      another one, and if not we stop, write the run log, and exit. Stopping
      yourself always beats being stopped.
    - The **signal handler** is for when the deadline was too generous or a
      single stage overran it. GitHub sends SIGTERM and waits before SIGKILL;
      that gap is enough to write one JSON file.

    `daily.max_minutes` must stay **comfortably under** the workflow's
    `timeout-minutes`, or the platform wins the race again and we are back to
    silence. The two numbers are commented in both places, because changing one
    alone is exactly how this returns.
    """

    def __init__(self, minutes: Optional[float] = None):
        self.limit_s = float(minutes if minutes is not None else cfg("daily.max_minutes", 30)) * 60
        self.started = time.monotonic()
        self.signalled = False
        self._previous: dict[int, Any] = {}

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return self.limit_s - self.elapsed

    def install(self) -> "Deadline":
        """Catch SIGTERM (and SIGINT) without swallowing them.

        The handler only sets a flag. Doing real work inside a signal handler is
        how you get a half-written JSON file, and a corrupt run log is worse
        than none — it is the same lie with a timestamp on it.
        """
        import signal

        def handler(signum, _frame):
            self.signalled = True

        for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
            if sig is None:
                continue
            try:
                self._previous[sig] = signal.getsignal(sig)
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not the main thread, or the platform disagrees. The deadline
                # still works; only the signal path is unavailable.
                pass
        return self

    def restore(self) -> None:
        import signal

        for sig, previous in self._previous.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass
        self._previous.clear()

    def check(self, stage: str) -> None:
        """Raise if we are out of time or have been asked to stop."""
        if self.signalled:
            raise Interrupted(
                f"SIGTERM received after {self.elapsed:.0f}s, before stage {stage!r}"
            )
        if self.remaining <= 0:
            raise TimeBudgetExceeded(
                f"time budget exceeded after {self.elapsed:.0f}s in stage {stage!r} "
                f"(limit {self.limit_s / 60:.0f} min)"
            )


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

    Set by the **slower** of the two sources. Journals reach 95% coverage at D+2
    (X1), but arXiv's `submittedDate` index is measurably three days behind:
    asked on 2026-08-18, it returned 0 for D-1, D-2 and D-3 and 221–453 per day
    from D-4 back, weekends included (`scripts/arxiv_visibility.py`).

    The first version of this used 3 days, from the journal figure alone, and
    that window sat entirely inside arXiv's blind zone — a scheduled run would
    have published journal-only issues every morning with every stage green.
    The default is now 7. Re-collecting days already seen costs a few requests:
    an item already published is skipped rather than published twice.
    """
    today = today or date.today()
    lookback = int(cfg("daily.lookback_days", 3))
    end = today - timedelta(days=int(cfg("daily.min_lag_days", 1)))
    return end - timedelta(days=lookback - 1), end


def smoke_window(today: Optional[date] = None) -> tuple[date, date]:
    """A narrow window for the "does this install work at all" run (H7).

    Step 1 of the turn-on checklist exists to confirm the install, the model
    cache and the keys. It was collecting the full seven-day window and
    summarising everything — **the most expensive thing the pipeline does, with
    the result thrown away** — and it is what ran for 44 minutes and got killed.

    Narrow, not shallow. The window ends past arXiv's three-day indexing lag so
    both sources actually return something; a smoke test that collects nothing
    proves nothing. Summaries are capped rather than skipped, because 0k was
    right that a dry run which skips the expensive stage does not test the thing
    most likely to break — but three summaries exercise the key, the schema and
    the parsing exactly as well as thirty do.
    """
    today = today or date.today()
    end = today - timedelta(days=int(cfg("daily.smoke_lag_days", 4)))
    span = max(1, int(cfg("daily.smoke_days", 2)))
    return end - timedelta(days=span - 1), end


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
    smoke: bool = False,
) -> dict[str, Any]:
    """One day, end to end, with the outcome decided rather than assumed.

    `smoke` narrows the window and caps summaries — the cheap path for step 1 of
    the turn-on checklist, which only needs to prove the install, the model and
    the keys work. See `smoke_window`.
    """
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
    covers_from, covers_to = (smoke_window(today) if smoke else target_window(today))
    summarize_limit = int(cfg("daily.smoke_summaries", 3)) if smoke else None
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
    deadline = Deadline().install()
    stopped_early: Optional[str] = None
    try:
        # Collect the window. `backfill_from` already exists for exactly this.
        _guard(run, "collect", lambda: stage_collect(
            run, covers_to, backfill_from=covers_from
        ))

        for name in STAGES:
            # Between stages, not inside them: a stage is the smallest unit we
            # can abandon without leaving half-written output behind.
            deadline.check(name)
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
                _guard(run, name, lambda: fn(run, use_llm=use_llm, limit=summarize_limit))
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
        outcome = decide(
            run,
            issue_date,
            len(selected),
            budget_exceeded=budget_exceeded,
            window_days=(covers_to - covers_from).days + 1,
        )

        if outcome.status == NOT_PUBLISHED or dry_run:
            if dry_run and outcome.status != NOT_PUBLISHED:
                outcome.reasons.append("dry run: nothing written")
            # A rehearsal does not enter the record. Writing `status: published`
            # for a date where nothing was published would make `uc status`
            # report a success that produced no issue — the run log exists to
            # answer "did this day get covered", and a dry run's answer is no.
            # The rehearsal is still on disk in `runs/{run_id}/metrics.json`,
            # which is where runs that wrote nothing belong.
            if not dry_run:
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

    except (TimeBudgetExceeded, Interrupted) as e:
        # The point of the whole Deadline machinery: a run that is out of time,
        # or being killed, still leaves a row saying so. A day with no record is
        # indistinguishable from a day the scheduler never fired, and telling
        # those apart is what `runs_log` is for.
        stopped_early = type(e).__name__
        run.error(f"daily: {stopped_early} — {e}")
        outcome = Outcome(
            date=issue_date,
            status=NOT_PUBLISHED,
            reasons=[str(e)],
            failed_stages=sorted(
                n for n, s in run.metrics.stages.items() if s == "FAILED"
            ),
            candidates=None,
            published=0,
            spend_usd=_spend_since(baseline_spend),
        )
        if not dry_run:
            record(outcome)
            from .notify import notify_failure

            notify_failure(issue_date, outcome.reasons, run=run)
        run.metrics.timing["daily_s"] = round(time.monotonic() - started, 1)
        run.save()
        result = _result(outcome, run, started, covers_from, covers_to, dry_run)
        result["stopped_early"] = stopped_early
        return result
    finally:
        deadline.restore()
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
