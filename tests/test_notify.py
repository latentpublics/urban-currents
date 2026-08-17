"""Telling someone when nobody is watching (phase 0k, X7).

Three properties: the subject is enough on its own, the fifth alert does not
read like the first, and a notification that cannot be sent costs a notification
rather than a run.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pipeline.deliver import DeliveryError, Message
from pipeline.notify import (
    consecutive_failures,
    failure_body,
    failure_subject,
    notify_failure,
    notify_weekly,
    status,
    weekly_body,
    weekly_summary,
)
from pipeline.outcome import NOT_PUBLISHED, PUBLISHED, QUIET, Outcome, record

DAY = date(2026, 8, 11)


def _log(d: date, status_: str, reasons=None, published: int = 0) -> None:
    record(
        Outcome(
            date=d,
            status=status_,
            reasons=reasons or [],
            published=published,
            candidates=100 if status_ != NOT_PUBLISHED else None,
        )
    )


class _Recorder:
    """A backend that remembers instead of sending."""

    name = "recorder"

    def __init__(self):
        self.sent: list[Message] = []

    def send(self, message: Message):
        self.sent.append(message)
        return {"backend": self.name, "recipients": len(message.recipients)}


class _Broken:
    name = "broken"

    def send(self, message: Message):
        raise DeliveryError("the mail server is not answering")


# --------------------------------------------------------------------------
# The subject line has to be enough
# --------------------------------------------------------------------------


def test_the_subject_carries_the_date_and_the_reason(repo):
    """Judged from a notification list, without opening anything."""
    subject = failure_subject(DAY, ["collect.openalex finished FAILED"], streak=1)

    assert "2026-08-11" in subject
    assert "collect.openalex" in subject
    assert len(subject) < 120  # not truncated by a mail client before the reason


def test_an_alert_with_no_reason_still_says_so(repo):
    subject = failure_subject(DAY, [], streak=1)
    assert "2026-08-11" in subject
    assert "unknown" in subject


# --------------------------------------------------------------------------
# Five identical alerts carry less than one
# --------------------------------------------------------------------------


def test_a_run_of_failures_is_counted(repo):
    for i in range(3):
        _log(DAY - timedelta(days=i), NOT_PUBLISHED, ["collect.arxiv did not run"])

    assert consecutive_failures(DAY) == 3


def test_a_success_breaks_the_run(repo):
    _log(DAY, NOT_PUBLISHED, ["x"])
    _log(DAY - timedelta(days=1), PUBLISHED, published=12)
    _log(DAY - timedelta(days=2), NOT_PUBLISHED, ["x"])

    assert consecutive_failures(DAY) == 1


def test_a_quiet_day_breaks_the_run_too(repo):
    """A quiet day is a day we looked. It is not an outage."""
    _log(DAY, NOT_PUBLISHED, ["x"])
    _log(DAY - timedelta(days=1), QUIET)

    assert consecutive_failures(DAY) == 1


def test_the_fourth_alert_does_not_read_like_the_first(repo):
    first = failure_subject(DAY, ["openalex 500"], streak=1)
    fourth = failure_subject(DAY, ["openalex 500"], streak=4)

    assert first != fourth
    assert "4 days" in fourth
    # And the body stops repeating the same sentence.
    assert "day 1 of" not in failure_body(DAY, ["openalex 500"], streak=1)
    assert "in a row" in failure_body(DAY, ["openalex 500"], streak=4)


def test_the_body_says_no_issue_was_written_not_that_it_was_quiet(repo):
    body = failure_body(DAY, ["collect.arxiv finished FAILED"], streak=1)

    assert "quiet" in body  # it explains the distinction
    assert "day we could not see" in body
    assert "Nothing was sent to readers." in body
    assert "uc status" in body


# --------------------------------------------------------------------------
# An alert is never worth a run
# --------------------------------------------------------------------------


def test_a_failed_alert_is_recorded_and_not_raised(repo, monkeypatch):
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")
    from pipeline.metrics import Run

    run = Run.for_date(DAY)
    result = notify_failure(DAY, ["collect.arxiv did not run"], backend=_Broken(), run=run)

    assert result["status"] == "alert_failed"
    assert "DeliveryError" in result["error"]
    assert any("notify" in e for e in run.metrics.errors)


def test_no_alert_recipient_is_not_an_error(repo, monkeypatch):
    monkeypatch.delenv("UC_ALERT_RECIPIENT", raising=False)
    assert notify_failure(DAY, ["x"])["status"] == "no_alert_recipient"


def test_the_alert_goes_to_the_operator_not_the_readers(repo, monkeypatch):
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")
    monkeypatch.setenv("UC_PREVIEW_RECIPIENT", "reader@example.org")

    backend = _Recorder()
    notify_failure(DAY, ["collect.openalex finished FAILED"], backend=backend)

    assert [m.recipients for m in backend.sent] == [["yjun@example.org"]]


def test_an_alert_writes_no_delivery_ledger_entry(repo, monkeypatch):
    """Operational mail is not an issue. It must not make the day look sent."""
    from pipeline.deliver import already_delivered

    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")
    notify_failure(DAY, ["x"], backend=_Recorder())

    assert already_delivered(DAY) is None


# --------------------------------------------------------------------------
# The weekly summary
# --------------------------------------------------------------------------


def test_the_weekly_summary_distinguishes_three_outcomes_and_a_gap(repo):
    _log(DAY, PUBLISHED, published=12)
    _log(DAY - timedelta(days=1), QUIET)
    _log(DAY - timedelta(days=2), NOT_PUBLISHED, ["collect.arxiv finished FAILED"])
    # DAY-3 .. DAY-6 have no record at all — the pipeline never ran.

    summary = weekly_summary(end=DAY)

    assert summary["outcomes"] == {PUBLISHED: 1, QUIET: 1, NOT_PUBLISHED: 1}
    assert summary["days_without_a_record"] == 4
    assert summary["items_published"] == 12
    assert len(summary["days"]) == 7


def test_a_day_that_never_ran_is_not_counted_as_quiet(repo):
    """The same distinction as the outcome model, one level up."""
    summary = weekly_summary(end=DAY)

    assert summary["outcomes"][QUIET] == 0
    assert summary["days_without_a_record"] == 7
    assert all(row["status"] == "no record" for row in summary["days"])


def test_the_weekly_body_names_the_reason_for_a_missed_day(repo):
    _log(DAY, NOT_PUBLISHED, ["collect.openalex finished FAILED"])
    body = weekly_body(weekly_summary(end=DAY))

    assert "collect.openalex finished FAILED" in body
    assert "no record" in body


def test_the_weekly_summary_is_produced_even_with_nowhere_to_send_it(repo, monkeypatch):
    monkeypatch.delenv("UC_ALERT_RECIPIENT", raising=False)
    result = notify_weekly(end=DAY)

    assert result["status"] == "no_alert_recipient"
    assert result["summary"]["from"] == "2026-08-05"


# --------------------------------------------------------------------------
# uc status
# --------------------------------------------------------------------------


def test_status_answers_is_anything_wrong_first(repo):
    _log(DAY - timedelta(days=1), PUBLISHED, published=10)
    _log(DAY, NOT_PUBLISHED, ["collect.arxiv did not run"])

    state = status()

    assert state["last_success"] == "2026-08-10"
    assert state["last_success_status"] == PUBLISHED
    assert state["unpublished_dates"] == ["2026-08-11"]
    assert state["next_window"]["from"] <= state["next_window"]["to"]


def test_status_separates_the_archive_from_the_run_log(repo):
    """Every issue published before X3 exists without a log row.

    Reporting only `last_success` would tell someone looking at a working site
    that nothing has ever run. Both numbers, so the gap shows as a gap.
    """
    from pipeline import paths

    issues = paths.CONTENT / "issues"
    issues.mkdir(parents=True, exist_ok=True)
    (issues / "2026-08-09.json").write_text("{}", encoding="utf-8")
    (issues / "2026-08-10.json").write_text("{}", encoding="utf-8")

    state = status()

    assert state["last_issue"] == "2026-08-10"
    assert state["issues_published"] == 2
    assert state["last_success"] is None  # no run log for either


def test_status_on_an_empty_repo_says_nothing_has_run(repo):
    state = status()

    assert state["last_success"] is None
    assert state["last_issue"] is None
    assert state["unpublished_dates"] == []
    assert state["delivery_backend"] == "file"
    assert state["lock"] is None


def test_status_reports_a_held_lock(repo):
    from pipeline.daily import acquire_lock

    lock = acquire_lock()
    try:
        assert status()["lock"]["pid"] > 0
    finally:
        lock.release()
    assert status()["lock"] is None


def test_status_counts_recipients_without_printing_them(repo, monkeypatch):
    """An address is personal data. The count answers the question."""
    monkeypatch.setenv("UC_PREVIEW_RECIPIENT", "reader@example.org")
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")

    state = status()
    blob = repr(state)

    assert state["reader_recipients"] == 1
    assert state["alert_recipients"] == 1
    assert "reader@example.org" not in blob
    assert "yjun@example.org" not in blob


# --------------------------------------------------------------------------
# Catch-up
# --------------------------------------------------------------------------


def test_catch_up_retries_the_missed_days_oldest_first(repo, monkeypatch):
    from pipeline import daily as daily_mod

    for i in range(3):
        _log(DAY - timedelta(days=i), NOT_PUBLISHED, ["collect.arxiv did not run"])

    attempted: list[date] = []
    monkeypatch.setattr(
        daily_mod,
        "run_daily",
        lambda d=None, **kw: (attempted.append(d), {"date": str(d), "status": QUIET})[1],
    )

    daily_mod.catch_up(today=DAY + timedelta(days=1))

    assert attempted == [DAY - timedelta(days=2), DAY - timedelta(days=1), DAY]


def test_catch_up_stops_at_the_horizon(repo, monkeypatch):
    """Past the horizon a missed day stays missed rather than being chased."""
    from pipeline import daily as daily_mod

    old = DAY - timedelta(days=30)
    _log(old, NOT_PUBLISHED, ["collect.arxiv did not run"])
    _log(DAY, NOT_PUBLISHED, ["collect.arxiv did not run"])

    attempted: list[date] = []
    monkeypatch.setattr(
        daily_mod,
        "run_daily",
        lambda d=None, **kw: (attempted.append(d), {"date": str(d), "status": QUIET})[1],
    )

    daily_mod.catch_up(today=DAY + timedelta(days=1))

    assert attempted == [DAY]
    # And it is still on the record, not quietly forgotten.
    assert old.isoformat() in status()["unpublished_dates"]


def test_one_unrecoverable_day_does_not_stop_the_others(repo, monkeypatch):
    from pipeline import daily as daily_mod

    _log(DAY - timedelta(days=1), NOT_PUBLISHED, ["x"])
    _log(DAY, NOT_PUBLISHED, ["x"])

    def flaky(d=None, **kw):
        if d == DAY - timedelta(days=1):
            raise RuntimeError("arXiv returned malformed XML")
        return {"date": str(d), "status": PUBLISHED}

    monkeypatch.setattr(daily_mod, "run_daily", flaky)
    results = daily_mod.catch_up(today=DAY + timedelta(days=1))

    assert [r["status"] for r in results] == ["retry_failed", PUBLISHED]
    assert "RuntimeError" in results[0]["error"]
