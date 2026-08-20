"""A run that is killed still has to leave a record (hotfix 2, H5).

CI hit `timeout-minutes: 45` and the platform cancelled the job. What survived:
no run-log row, no alert, no commit. To `uc status` the day did not exist, which
is indistinguishable from the schedule never having fired — **X3's problem
reappearing one level up**, because a process killed from outside cannot record
its own death.

Three defences, tested here in the order they fire:

1. a wall-clock budget the run checks between stages and stops itself on;
2. a SIGTERM handler for when one stage overruns the budget;
3. `uc record-interrupted`, which the workflow calls when neither of those got
   the chance — SIGKILL, OOM, a runner that vanished.

And one property that matters as much: `interrupted` is kept apart from
`not_published`, because one is a verdict and the other is the absence of one.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from pipeline import daily as daily_mod
from pipeline import run_stages
from pipeline.daily import Deadline, Interrupted, TimeBudgetExceeded
from pipeline.outcome import (
    INTERRUPTED,
    NOT_PUBLISHED,
    PUBLISHED,
    interrupted_dates,
    load_log,
    record_interrupted,
    unpublished_dates,
)

DAY = date(2026, 8, 20)
# The stages a healthy day reports. The four after the sources are
# `REQUIRED_STAGES` (0U, U1) — a run that never summarised is not a published
# day, so a fixture claiming to be one has to say it summarised.
GOOD = {
    "collect": "OK",
    "collect.arxiv": "OK",
    "collect.openalex": "OK",
    "classify": "OK",
    "summarize": "OK",
    "select": "OK",
    "issue": "OK",
}


# --------------------------------------------------------------------------
# 1. The deadline
# --------------------------------------------------------------------------


def test_the_deadline_raises_once_the_budget_is_gone():
    deadline = Deadline(minutes=0)
    with pytest.raises(TimeBudgetExceeded) as excinfo:
        deadline.check("summarize")

    # The message names the stage, because "it was slow" is not actionable and
    # "it was slow in summarize" is.
    assert "summarize" in str(excinfo.value)
    assert "time budget" in str(excinfo.value)


def test_a_deadline_with_time_left_says_nothing():
    Deadline(minutes=60).check("classify")  # does not raise


def test_a_signal_is_noticed_between_stages():
    """The handler only sets a flag.

    Writing JSON inside a signal handler is how you get a half-written file, and
    a corrupt run log is worse than none — it is the same lie with a timestamp.
    """
    deadline = Deadline(minutes=60)
    deadline.signalled = True

    with pytest.raises(Interrupted) as excinfo:
        deadline.check("link")
    assert "SIGTERM" in str(excinfo.value)


def test_installing_and_restoring_leaves_the_handlers_as_they_were():
    import signal

    before = signal.getsignal(signal.SIGTERM)
    deadline = Deadline(minutes=1).install()
    try:
        assert signal.getsignal(signal.SIGTERM) is not before
    finally:
        deadline.restore()
    assert signal.getsignal(signal.SIGTERM) is before


# --------------------------------------------------------------------------
# 2. The run records its own stop
# --------------------------------------------------------------------------


@pytest.fixture
def wiring(monkeypatch):
    """A `run_daily` whose stages are instant, so only the deadline decides."""
    state: dict = {"collected": []}

    def collect(run, d, backfill_from=None, **kw):
        run.metrics.stages.update(GOOD)
        run.metrics.counts.arxiv_fetched = 300
        run.metrics.counts.openalex_fetched = 40
        return []

    monkeypatch.setattr(run_stages, "stage_collect", collect)
    monkeypatch.setattr(run_stages, "read_stage", lambda run, name: [1, 2, 3])
    monkeypatch.setattr(daily_mod, "STAGES", ("dedup", "summarize", "score"))
    for name in ("dedup", "summarize", "score"):
        monkeypatch.setattr(
            run_stages, f"stage_{name}", lambda run, *a, **k: [], raising=False
        )
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")
    return state


def test_running_out_of_time_is_recorded_rather_than_silent(repo, wiring, monkeypatch):
    """The whole point. Being stopped by the platform loses the record; stopping
    yourself keeps it."""
    monkeypatch.setattr(daily_mod, "Deadline", lambda *a, **k: Deadline(minutes=0))

    result = daily_mod.run_daily(d=DAY, use_llm=False)

    assert result["status"] == NOT_PUBLISHED
    assert result["stopped_early"] == "TimeBudgetExceeded"

    logged = load_log(DAY)
    assert logged is not None, "a run that stopped itself must leave a row"
    assert logged["status"] == NOT_PUBLISHED
    assert any("time budget" in r for r in logged["reasons"])
    # And it names where the time went.
    assert any("dedup" in r or "summarize" in r or "score" in r for r in logged["reasons"])


def test_a_dry_run_that_times_out_writes_no_row(repo, wiring, monkeypatch):
    """A rehearsal still does not enter the record, even when it is cut short."""
    monkeypatch.setattr(daily_mod, "Deadline", lambda *a, **k: Deadline(minutes=0))

    result = daily_mod.run_daily(d=DAY, dry_run=True, use_llm=False)

    assert result["stopped_early"] == "TimeBudgetExceeded"
    assert load_log(DAY) is None


def test_a_sigterm_mid_run_still_leaves_a_row(repo, wiring, monkeypatch):
    """Simulates the cancellation GitHub actually sends."""
    deadline = Deadline(minutes=60)

    original = deadline.check
    calls = {"n": 0}

    def check(stage: str):
        calls["n"] += 1
        if calls["n"] == 2:
            deadline.signalled = True   # the platform asks us to stop
        original(stage)

    deadline.check = check  # type: ignore[method-assign]
    monkeypatch.setattr(daily_mod, "Deadline", lambda *a, **k: deadline)

    result = daily_mod.run_daily(d=DAY, use_llm=False)

    assert result["stopped_early"] == "Interrupted"
    logged = load_log(DAY)
    assert logged["status"] == NOT_PUBLISHED
    assert any("SIGTERM" in r for r in logged["reasons"])


def test_the_lock_is_released_when_the_run_is_cut_short(repo, wiring, monkeypatch):
    """A lock left behind by a killed run is how the next day fails too."""
    from pipeline.daily import lock_path

    monkeypatch.setattr(daily_mod, "Deadline", lambda *a, **k: Deadline(minutes=0))
    daily_mod.run_daily(d=DAY, use_llm=False)

    assert not lock_path().exists()


# --------------------------------------------------------------------------
# 3. The last net
# --------------------------------------------------------------------------


def test_record_interrupted_writes_the_row_nobody_else_could(repo):
    record_interrupted(DAY, "the job ended without a verdict")

    logged = load_log(DAY)
    assert logged["status"] == INTERRUPTED
    assert logged["published"] == 0
    assert logged["candidates"] is None  # nobody counted anything
    assert "without a verdict" in logged["reasons"][0]


def test_record_interrupted_never_overwrites_a_real_verdict(repo):
    """If the pipeline concluded anything, its conclusion is the better one."""
    from pipeline.metrics import Run
    from pipeline.outcome import decide, record

    run = Run.for_date(DAY)
    run.metrics.stages.update(GOOD)
    run.metrics.counts.arxiv_candidates = 12
    record(decide(run, DAY, published_count=12))

    record_interrupted(DAY, "the job ended without a verdict")

    assert load_log(DAY)["status"] == PUBLISHED


def test_interrupted_is_not_counted_as_a_failure(repo):
    """Different states, different remedies.

    `not_published` is retried and usually succeeds. `interrupted` says the run
    does not fit the time it was given, and retrying it unchanged repeats it.
    """
    record_interrupted(DAY, "cancelled")

    assert [r["date"] for r in interrupted_dates()] == [str(DAY)]
    assert unpublished_dates() == []


def test_status_reports_the_two_apart(repo):
    from pipeline.metrics import Run
    from pipeline.outcome import decide, record
    from pipeline.notify import status

    record_interrupted(DAY, "cancelled")
    run = Run.for_date(DAY - timedelta(days=1))
    run.metrics.stages.update({"collect": "OK", "collect.arxiv": "FAILED"})
    record(decide(run, DAY - timedelta(days=1), published_count=0))

    state = status()
    assert state["interrupted_dates"] == [str(DAY)]
    assert state["unpublished_dates"] == [str(DAY - timedelta(days=1))]


# --------------------------------------------------------------------------
# The window follows the date it was asked for (phase 0N, P1)
# --------------------------------------------------------------------------


def test_an_explicit_date_collects_that_dates_window(repo, wiring, monkeypatch):
    """`uc daily --date X` must collect X's window, not today's.

    It was collecting the current seven days and filing them under X, which is
    what left `2026-08-11_2026-08-17` raw files inside `runs/run_2026-07-02/`.
    The failure alert tells YJUN to run exactly this command to retry a date.
    """
    from pipeline.daily import target_window

    seen: dict = {}

    def collect(run, d, backfill_from=None, **kw):
        seen["covers_to"] = d
        seen["covers_from"] = backfill_from
        run.metrics.stages.update(GOOD)
        run.metrics.counts.arxiv_fetched = 10
        run.metrics.counts.openalex_fetched = 10
        return []

    monkeypatch.setattr(run_stages, "stage_collect", collect)
    asked = date(2026, 7, 2)
    daily_mod.run_daily(d=asked, dry_run=True, use_llm=False)

    expected_from, expected_to = target_window(asked)
    assert (seen["covers_from"], seen["covers_to"]) == (expected_from, expected_to)
    # And that window sits around the date asked for, not around today.
    assert seen["covers_to"] < date(2026, 7, 3)


def test_catch_up_retries_a_date_with_its_own_window(repo, monkeypatch):
    """Catch-up runs every day once the schedule is on.

    Passing the real `today` made every recovered date carry papers from the
    current week, and nothing in the archive would have said so.
    """
    from pipeline import daily as dm

    missed = date(2026, 8, 12)
    record_interrupted(missed, "cancelled")
    from pipeline.outcome import NOT_PUBLISHED, Outcome, record

    record(Outcome(date=missed, status=NOT_PUBLISHED, reasons=["collect.arxiv did not run"]))

    seen: list = []
    monkeypatch.setattr(
        dm, "run_daily",
        lambda d=None, today=None, **kw: (seen.append((d, today)), {"status": "published"})[1],
    )
    dm.catch_up(today=date(2026, 8, 18))

    assert seen, "nothing was retried"
    for asked_date, as_of in seen:
        assert as_of == asked_date, "the window must follow the date being retried"
