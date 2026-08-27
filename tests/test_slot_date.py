"""The day a scheduled run belongs to, and the gap a wrong answer leaves.

2026-08-26 has no issue and no run-log row, and nothing failed to produce it.
The 21:00 UTC slot for that day started at about 00:35 UTC on 08-27 — GitHub's
scheduler is best-effort and ran roughly three and a half hours late. The
pipeline dated itself with `date.today()`, so the run published a **08-27**
issue and wrote an 08-27 row saying `attempts: 1`: it did not know it was late.

Then three separate safety nets each looked straight past the hole, all for the
same reason — every one of them asked the wall clock what day it was:

* the workflow's interrupted-row step asked about "today", found the 08-27 row
  that had just been written, and reported that the day was already recorded;
* `catch_up` built its queue from the rows in `content/runs_log/`, and 08-26
  had no row to be built from;
* the deadman checked how old the newest row was, and the newest row was
  *newer* than it should have been.

So the tests here are in three groups, one per net, plus the one that keeps the
duplicated hour honest. The property they defend together: **a day that was
never attempted is a fact the system can see**, which it was not.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from pipeline.daily import slot_date
from pipeline.outcome import (
    INTERRUPTED,
    NOT_PUBLISHED,
    PUBLISHED,
    QUIET,
    Outcome,
    missing_dates,
    record,
)

ROOT = Path(__file__).resolve().parent.parent
DAY = date(2026, 8, 26)


def _log(d: date, status: str) -> None:
    record(Outcome(date=d, status=status, candidates=1, published=1))


def _utc(y: int, m: int, dd: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, dd, hh, mm, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# The slot rule
# --------------------------------------------------------------------------


def test_a_run_on_time_is_dated_the_day_its_slot_fell_on():
    assert slot_date(_utc(2026, 8, 26, 21, 0), slot_hour=21) == DAY
    assert slot_date(_utc(2026, 8, 26, 21, 5), slot_hour=21) == DAY


def test_the_run_that_lost_08_26_would_now_be_dated_08_26():
    """The measured case. 00:35 UTC on 08-27, and `date.today()` said 08-27."""
    late = _utc(2026, 8, 27, 0, 35)

    assert late.date() == date(2026, 8, 27)  # what the old code used
    assert slot_date(late, slot_hour=21) == DAY


def test_lateness_costs_nothing_until_it_reaches_the_next_slot():
    """Three hours used to cost a day. Twenty now cost nothing.

    The boundary is the next slot, not midnight: a run is misfiled only once
    another slot has come due, and a run that late has a bigger problem than
    its date.
    """
    for hours in (1, 3, 6, 12, 20, 23):
        started = _utc(2026, 8, 26, 21, 0) + timedelta(hours=hours)
        assert slot_date(started, slot_hour=21) == DAY, hours

    just_past_the_next_slot = _utc(2026, 8, 26, 21, 0) + timedelta(hours=24)
    assert slot_date(just_past_the_next_slot, slot_hour=21) == DAY + timedelta(days=1)


def test_a_run_before_the_slot_belongs_to_yesterdays():
    """The 09:00 UTC deadman asks this question and must get 'yesterday'."""
    assert slot_date(_utc(2026, 8, 27, 9, 0), slot_hour=21) == DAY


def test_a_naive_datetime_is_read_as_utc():
    """The runner is UTC and so is the cron. Nothing here is local time."""
    assert slot_date(datetime(2026, 8, 27, 0, 35), slot_hour=21) == DAY


def test_the_hour_comes_from_config_when_it_is_not_passed(repo):
    assert slot_date(_utc(2026, 8, 27, 0, 35)) == DAY


# --------------------------------------------------------------------------
# The hour is written down three times. It must be the same number.
# --------------------------------------------------------------------------


def _daily_yml() -> str:
    return (ROOT / ".github/workflows/daily.yml").read_text(encoding="utf-8")


def test_the_cron_the_config_and_the_workflow_env_agree():
    """Three copies of one fact, pinned.

    `config/pipeline.yaml` holds it for the pipeline, `daily.yml`'s `cron:` is
    the thing itself, and `SLOT_HOUR_UTC` in the same file exists for the
    `always()` step that has to work when the install did not. Nothing stops
    someone moving the cron and leaving the other two behind — except this.
    """
    workflow = yaml.safe_load(_daily_yml())
    # `on` is parsed as the boolean True by YAML 1.1, which is why it is fetched
    # this way rather than by name.
    triggers = workflow.get("on", workflow.get(True))
    crons = [entry["cron"] for entry in triggers["schedule"]]
    assert len(crons) == 1, crons
    cron_hour = int(crons[0].split()[1])

    env_hour = int(workflow["jobs"]["daily"]["env"]["SLOT_HOUR_UTC"])

    config = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text(encoding="utf-8"))
    config_hour = int(config["daily"]["slot_hour_utc"])

    assert cron_hour == env_hour == config_hour


def test_the_workflow_passes_a_date_rather_than_letting_the_clock_decide():
    """`run_daily`'s `date.today()` fallback must not be reachable on a runner.

    It is kept for a person typing `uc daily`, who does mean the day they are
    having. A scheduled run means the slot it belongs to, and the workflow has
    to say so out loud.
    """
    text = _daily_yml()
    assert "uc slot-date" in text
    assert "--date \"$ISSUE_DATE\"" in text or "--date \"$ISSUE_DATE\"" in text.replace("'", '"')


def test_the_interrupted_net_asks_about_the_slot_day():
    """It used to ask `date -u +%Y-%m-%d`, and on 08-26 that was the bug.

    The job had crossed midnight; `date -u` said 08-27; the 08-27 row had just
    been written by the run itself; the step said "already written" and passed.
    """
    text = _daily_yml()
    step = text.split("- name: Record an interrupted run", 1)[1]
    step = step.split("- name: Report", 1)[0]
    assert 'day="$SLOT_DATE"' in step
    assert "uc record-interrupted --date" in step


# --------------------------------------------------------------------------
# Net 2 — the catch-up queue is a calendar
# --------------------------------------------------------------------------


def test_catch_up_retries_a_day_that_left_no_row_at_all(repo, monkeypatch):
    """The 08-26 case, end to end.

    Every day around it published; 08-26 has nothing. The old queue read the
    rows that existed and so could not name a day that had never written one.
    """
    from pipeline import daily as daily_mod

    today = date(2026, 8, 27)
    for i in range(1, 8):
        d = today - timedelta(days=i)
        if d != DAY:
            _log(d, PUBLISHED)

    attempted: list[date] = []
    monkeypatch.setattr(
        daily_mod,
        "run_daily",
        lambda d=None, **kw: (attempted.append(d), {"date": str(d), "status": QUIET})[1],
    )
    daily_mod.catch_up(today=today)

    assert attempted == [DAY]


def test_catch_up_retries_an_interrupted_day(repo, monkeypatch):
    """A row saying "the runner died" is a day worth asking about again.

    Nothing retried these. The workflow writes the row precisely so the day is
    not silent, and then the day sat there.
    """
    from pipeline import daily as daily_mod

    today = date(2026, 8, 27)
    for i in range(1, 8):
        _log(today - timedelta(days=i), PUBLISHED)
    _log(DAY, INTERRUPTED)

    attempted: list[date] = []
    monkeypatch.setattr(
        daily_mod,
        "run_daily",
        lambda d=None, **kw: (attempted.append(d), {"date": str(d), "status": QUIET})[1],
    )
    daily_mod.catch_up(today=today)

    assert attempted == [DAY]


def test_catch_up_leaves_published_and_quiet_days_alone(repo, monkeypatch):
    """A calendar queue must not mean re-running the whole week every night."""
    from pipeline import daily as daily_mod

    today = date(2026, 8, 27)
    for i in range(1, 8):
        _log(today - timedelta(days=i), PUBLISHED if i % 2 else QUIET)

    attempted: list[date] = []
    monkeypatch.setattr(
        daily_mod,
        "run_daily",
        lambda d=None, **kw: (attempted.append(d), {"date": str(d), "status": QUIET})[1],
    )
    daily_mod.catch_up(today=today)

    assert attempted == []


def test_a_gap_past_the_horizon_stays_a_gap(repo, monkeypatch):
    """Bounded, for the same reason it always was: the sources have moved on.

    A day with no row is not more recoverable than a day with a failed one, so
    widening the queue must not widen the horizon.
    """
    from pipeline import daily as daily_mod

    today = date(2026, 8, 27)
    for i in range(1, 8):
        _log(today - timedelta(days=i), PUBLISHED)
    # 08-15 is twelve days back. It has no row and it is not coming back.

    attempted: list[date] = []
    monkeypatch.setattr(
        daily_mod,
        "run_daily",
        lambda d=None, **kw: (attempted.append(d), {"date": str(d), "status": QUIET})[1],
    )
    daily_mod.catch_up(today=today)

    assert attempted == []


def test_catch_up_still_goes_oldest_first(repo, monkeypatch):
    from pipeline import daily as daily_mod

    today = date(2026, 8, 27)
    for i in (1, 2, 3):
        _log(today - timedelta(days=i), NOT_PUBLISHED)
    for i in (4, 5, 6, 7):
        _log(today - timedelta(days=i), PUBLISHED)

    attempted: list[date] = []
    monkeypatch.setattr(
        daily_mod,
        "run_daily",
        lambda d=None, **kw: (attempted.append(d), {"date": str(d), "status": QUIET})[1],
    )
    daily_mod.catch_up(today=today)

    assert attempted == [
        today - timedelta(days=3),
        today - timedelta(days=2),
        today - timedelta(days=1),
    ]


# --------------------------------------------------------------------------
# Net 3 — the deadman looks for holes, not just for staleness
# --------------------------------------------------------------------------


def test_missing_dates_names_the_day_with_no_row(repo):
    for i in range(1, 8):
        d = date(2026, 8, 27) - timedelta(days=i)
        if d != DAY:
            _log(d, PUBLISHED)

    assert missing_dates(date(2026, 8, 20), date(2026, 8, 26)) == [DAY]


def test_a_gap_is_invisible_to_the_freshness_check(repo):
    """The two questions are different, and this is the proof.

    08-26 missing, 08-27 present and fresh. Any check that asks how old the
    newest row is reports perfect health here — which is what happened.
    """
    for i in range(1, 8):
        d = date(2026, 8, 27) - timedelta(days=i)
        if d != DAY:
            _log(d, PUBLISHED)
    _log(date(2026, 8, 27), PUBLISHED)

    from pipeline.outcome import all_logs

    newest = max(r["date"] for r in all_logs())
    assert newest == "2026-08-27"  # the freshness check is satisfied
    assert missing_dates(date(2026, 8, 20), date(2026, 8, 26)) == [DAY]  # and wrong


def test_missing_dates_is_empty_on_a_complete_week(repo):
    for i in range(0, 8):
        _log(date(2026, 8, 27) - timedelta(days=i), PUBLISHED)

    assert missing_dates(date(2026, 8, 20), date(2026, 8, 26)) == []


def test_a_day_that_failed_is_not_reported_as_missing(repo):
    """It has a row. It is loud in `uc status` and in the alert already.

    The gap check is about days nothing ever spoke for; conflating the two would
    make the alarm fire on ordinary bad weather and get itself ignored.
    """
    for i in range(1, 8):
        d = date(2026, 8, 27) - timedelta(days=i)
        _log(d, NOT_PUBLISHED if d == DAY else PUBLISHED)

    assert missing_dates(date(2026, 8, 20), date(2026, 8, 26)) == []


@pytest.mark.parametrize("grace", [1, 2])
def test_the_grace_window_keeps_the_alarm_from_crying_wolf(repo, grace):
    """A day inside the grace window may simply be running late.

    The deadman fires at 09:00 UTC and the slot is 21:00, so with `grace: 1` the
    newest day it will call missing is one whose run was due about 36 hours ago
    — deliberately the same tolerance the freshness threshold was tuned to.
    """
    for i in range(1, 10):
        _log(date(2026, 8, 27) - timedelta(days=i), PUBLISHED)

    as_of = slot_date(_utc(2026, 8, 27, 9, 0), slot_hour=21)  # 2026-08-26
    end = as_of - timedelta(days=grace)
    assert missing_dates(end - timedelta(days=6), end) == []
    assert end < as_of


def test_the_deadman_would_have_gone_red_the_morning_after(repo):
    """09:00 UTC on 08-28, with the archive exactly as it stands today."""
    for d in ("2026-08-20", "2026-08-21", "2026-08-22", "2026-08-23",
              "2026-08-24", "2026-08-25", "2026-08-27"):
        _log(date.fromisoformat(d), PUBLISHED)

    as_of = slot_date(_utc(2026, 8, 28, 9, 0), slot_hour=21)  # 2026-08-27
    end = as_of - timedelta(days=1)                            # 2026-08-26
    assert missing_dates(end - timedelta(days=6), end) == [DAY]


# --------------------------------------------------------------------------
# The deadman's own numbers
# --------------------------------------------------------------------------


def test_the_deadman_horizon_matches_the_catch_up_horizon(repo):
    """The gap alarm must not name days catch-up is not allowed to fix.

    A wider deadman horizon reports gaps nothing can repair, which trains an
    operator to close the alarm; a narrower one hides repairable ones.
    """
    from pipeline.config import cfg

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/deadman.yml").read_text(encoding="utf-8")
    )
    horizon = int(workflow["jobs"]["deadman"]["env"]["HORIZON_DAYS"])
    assert horizon == int(cfg("daily.catch_up_days", 7))


def test_the_deadman_runs_the_command_operations_tells_a_person_to_run():
    text = (ROOT / ".github/workflows/deadman.yml").read_text(encoding="utf-8")
    assert "uc missing-days" in text
    assert re.search(r"uc missing-days .*--grace", text)


# --------------------------------------------------------------------------
# `uc status` — the first command after being away
# --------------------------------------------------------------------------


def test_status_reports_a_day_nothing_ever_spoke_for(repo, monkeypatch):
    """The morning-after command has to be able to say it.

    Every other field `status` reports is read off a run-log row, so on the
    morning of 08-27 it had nothing to say about 08-26 — not "missing", not
    "failed", nothing. The archive and the status output agreed, and both were
    wrong in the same direction.
    """
    from pipeline import daily as daily_mod
    from pipeline import notify

    # `status()` imports `slot_date` at call time, so the module it lives in is
    # the thing to patch.
    monkeypatch.setattr(daily_mod, "slot_date", lambda *a, **k: DAY)
    for i in range(1, 7):
        _log(DAY - timedelta(days=i), PUBLISHED)
    _log(DAY + timedelta(days=1), PUBLISHED)  # the run that took 08-26's place

    state = notify.status()

    assert state["missing_dates"] == [DAY.isoformat()]
    # And it is not smuggled into the field that means something else.
    assert state["unpublished_dates"] == []


def test_status_separates_a_gap_from_a_failed_day(repo, monkeypatch):
    from pipeline import daily as daily_mod
    from pipeline import notify

    # `status()` imports `slot_date` at call time, so the module it lives in is
    # the thing to patch.
    monkeypatch.setattr(daily_mod, "slot_date", lambda *a, **k: DAY)
    for i in range(0, 7):
        d = DAY - timedelta(days=i)
        if d != DAY - timedelta(days=2):
            _log(d, NOT_PUBLISHED if i == 4 else PUBLISHED)

    state = notify.status()

    assert state["missing_dates"] == [(DAY - timedelta(days=2)).isoformat()]
    assert state["unpublished_dates"] == [(DAY - timedelta(days=4)).isoformat()]
