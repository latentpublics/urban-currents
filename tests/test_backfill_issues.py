"""Filling the archive (phase 0L, N1).

The archive is five days long, which is why coupling anchors came out at n=3,
canon lines at n=2, and `tag shift` with no row on any day. These tests pin the
four rules that make a backfill worth trusting: it never rewrites what exists,
it goes oldest first so a truncated run leaves consecutive days, it stops at the
spend ceiling instead of going over, and it resumes where it stopped.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pipeline import paths
from pipeline.backfill_issues import (
    LIVE_MARGIN_DAYS,
    backfill,
    load_checkpoint,
    target_dates,
)

TODAY = date(2026, 8, 18)


def _fake_runner(calls: list, cost_each: float = 0.0, statuses=None):
    """A runner that records what it was asked for and spends a little."""
    from pipeline.llm import UsageState

    def run(d=None, window=None, backfilled=False, today=None, **kw):
        calls.append({"date": d, "window": window, "backfilled": backfilled})
        if cost_each:
            usage = UsageState.load()
            usage.cost_usd = round(usage.cost_usd + cost_each, 6)
            usage.calls += 1
            usage.save()
        status = (statuses or {}).get(str(d), "published")
        return {"status": status, "published": 12, "candidates": 40}

    return run


# --------------------------------------------------------------------------
# Which days, in which order
# --------------------------------------------------------------------------


def test_days_are_oldest_first():
    """A truncated run must leave a consecutive run of days.

    Every window-based measurement — coupling, tag shift, first internal
    citation — needs consecutive days, so sixty days with holes is worth less
    than thirty-five without.
    """
    days = target_dates(5, today=TODAY)

    assert days == sorted(days)
    assert days[0] < days[-1]


def test_the_backfill_never_reaches_into_the_live_window():
    """CI owns the recent days. Two machines writing one date is a conflict this
    project has already had once."""
    days = target_dates(60, today=TODAY)

    assert max(days) == TODAY - timedelta(days=LIVE_MARGIN_DAYS)
    assert TODAY not in days


def test_an_existing_issue_is_skipped_not_rewritten(repo):
    """The five original issues and anything CI published are immutable."""
    issues = paths.CONTENT / "issues"
    issues.mkdir(parents=True, exist_ok=True)
    taken = TODAY - timedelta(days=LIVE_MARGIN_DAYS)
    (issues / f"{taken}.json").write_text('{"date": "x"}', encoding="utf-8")
    before = (issues / f"{taken}.json").read_bytes()

    calls: list = []
    result = backfill(days=3, today=TODAY, runner=_fake_runner(calls), on_checkpoint=None)

    assert taken not in [c["date"] for c in calls]
    assert result.skipped == 1
    assert (issues / f"{taken}.json").read_bytes() == before


def test_each_day_gets_a_one_day_window_and_the_backfilled_flag(repo):
    """A historical issue says what appeared that day.

    Dating it by a seven-day sweep would make it claim its neighbours' papers.
    """
    calls: list = []
    backfill(days=3, today=TODAY, runner=_fake_runner(calls), on_checkpoint=None)

    assert len(calls) == 3
    for c in calls:
        assert c["window"] == (c["date"], c["date"])
        assert c["backfilled"] is True


# --------------------------------------------------------------------------
# The budget
# --------------------------------------------------------------------------


def test_it_stops_at_the_spend_ceiling_and_says_where(repo):
    calls: list = []
    result = backfill(
        days=20,
        budget_usd=0.30,
        today=TODAY,
        runner=_fake_runner(calls, cost_each=0.10),
        on_checkpoint=None,
    )

    assert result.attempted == 3          # 0.00, 0.10, 0.20 spent; 0.30 stops it
    assert result.stopped_on is not None
    assert "spend budget reached" in result.stopped_on
    assert result.spend_usd == pytest.approx(0.30, abs=1e-6)


def test_a_zero_budget_attempts_nothing(repo):
    calls: list = []
    result = backfill(
        days=5, budget_usd=0.0, today=TODAY, runner=_fake_runner(calls), on_checkpoint=None
    )

    assert result.attempted == 0
    assert calls == []


# --------------------------------------------------------------------------
# Stopping and resuming
# --------------------------------------------------------------------------


def test_a_checkpoint_is_written_after_every_day(repo):
    calls: list = []
    backfill(days=3, today=TODAY, runner=_fake_runner(calls), on_checkpoint=None)

    state = load_checkpoint()
    assert len(state["done"]) == 3


def test_a_second_run_resumes_rather_than_starting_over(repo):
    """A run that dies on day 41 must not begin again at day 1."""
    first: list = []
    backfill(
        days=6,
        budget_usd=0.25,
        today=TODAY,
        runner=_fake_runner(first, cost_each=0.10),
        on_checkpoint=None,
    )
    assert len(first) == 3

    second: list = []
    backfill(
        days=6,
        budget_usd=10.0,
        today=TODAY,
        runner=_fake_runner(second),
        on_checkpoint=None,
    )

    # The three already done are not repeated, and the rest are picked up.
    assert set(d["date"] for d in first).isdisjoint(d["date"] for d in second)
    assert len(second) == 3
    assert len(load_checkpoint()["done"]) == 6


def test_commit_blocks_fire_on_the_interval(repo):
    """Sixty days in one commit is a revert that is all or nothing."""
    calls: list = []
    blocks: list = []
    backfill(
        days=6,
        today=TODAY,
        runner=_fake_runner(calls),
        on_checkpoint=lambda n, state: blocks.append(n),
        commit_every=2,
    )

    assert blocks == [2, 4, 6]


def test_outcomes_are_counted_apart(repo):
    calls: list = []
    days = target_dates(3, today=TODAY)
    statuses = {
        str(days[0]): "published",
        str(days[1]): "quiet",
        str(days[2]): "not_published",
    }
    result = backfill(
        days=3,
        today=TODAY,
        runner=_fake_runner(calls, statuses=statuses),
        on_checkpoint=None,
    )

    assert (result.published, result.quiet, result.not_published) == (1, 1, 1)


def test_the_budget_spans_passes_rather_than_resetting(repo):
    """A sixty-day run outlives one process, so it is driven in passes.

    Measuring spend from "cumulative cost when this pass started" gives every
    pass a fresh ceiling — fifteen passes, fifteen budgets. The symptom was a
    checkpoint reporting *less* spend after a day had been added than before it.
    """

    first: list = []
    backfill(
        days=6,
        budget_usd=0.25,
        today=TODAY,
        runner=_fake_runner(first, cost_each=0.10),
        on_checkpoint=None,
    )
    assert len(first) == 3
    spent_after_first = load_checkpoint()["spend_usd"]
    assert spent_after_first == pytest.approx(0.30, abs=1e-6)

    # A second pass under the same ceiling must have nothing left to spend.
    second: list = []
    result = backfill(
        days=6,
        budget_usd=0.25,
        today=TODAY,
        runner=_fake_runner(second, cost_each=0.10),
        on_checkpoint=None,
    )

    assert second == []
    assert result.attempted == 0
    assert result.stopped_on is not None
    # And the reported total never goes backwards.
    assert load_checkpoint()["spend_usd"] >= spent_after_first


def test_the_baseline_survives_a_checkpoint_rewrite(repo):
    calls: list = []
    backfill(days=2, today=TODAY, runner=_fake_runner(calls, cost_each=0.01), on_checkpoint=None)
    state = load_checkpoint()

    assert "baseline_cost_usd" in state
    assert state["spend_usd"] == pytest.approx(0.02, abs=1e-6)
