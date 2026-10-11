"""The alert counts days we did not see a source, not days it was quiet (#1).

`silent_sources` and `failed_sources` are two facts about the same absence: on
a silent day the source answered and had nothing, on a failed day every request
to it died. `notify_silent_sources` counted only the first, so three days of
HTTP 406 reached the one alert route anyone has confirmed reading — not at all.

The only thing that fired instead was the daily job going red at
`FAILED_SOURCE_RED_DAYS`, which GitHub mails to whichever account last edited
the workflow's `cron:`. That account's notification settings have never been
checked. The Resend route this tier uses was confirmed end to end in 1P. These
tests hold the widening that keeps the confirmed route in play.

## Reaching the line

This repository has three times shipped a test that passed without reaching
what it claimed to hold — most relevantly 1L's
`test_failure_is_not_written_down_as_silence`, which called `silent_sources`
with `window_days=1`, a width at which the function returns `[]` before reading
any input. It passed for three weeks while the defect it named sat there.

So every test below asserts the **streak count itself** next to the behaviour
it is really about, and the mixed-run tests additionally assert that
`silent_streak` and `failed_streak` are both short. If the union were not what
decided, those numbers could not add up — the assertion fails rather than the
test quietly agreeing with a line it never ran.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from pipeline.deliver import Message
from pipeline.notify import SILENT_ALERT_DAYS, notify_silent_sources
from pipeline.outcome import (
    UNSEEN_BOTH,
    UNSEEN_FAILED,
    UNSEEN_SILENT,
    failed_streak,
    log_dir,
    silent_streak,
    unseen_days,
    unseen_streak,
)

DAY = date(2026, 10, 10)
ARXIV = "collect.arxiv"

# One letter per day, oldest first, the way the issue writes them.
#   S  silent: the stage finished OK and the source returned nothing
#   F  failed: every request to the source died
#   B  both, which is what the log actually records today — see `_row`
#   O  seen: a row exists and names the source in neither list
#   -  no row at all
SILENT, FAILED, BOTH, SEEN, GAP = "S", "F", "B", "O", "-"


class _Recorder:
    name = "recorder"

    def __init__(self):
        self.sent: list[Message] = []

    def send(self, message: Message):
        self.sent.append(message)
        return {"backend": self.name, "recipients": len(message.recipients)}


def _row(d: date, mark: str, source: str = ARXIV) -> None:
    """Write one run-log row in the shape `uc daily` writes it.

    Same two fields, same spelling, same `published` verdict: a source dying
    never changes the day's status (L0), so a row that said `not_published`
    would be testing a day this alert deliberately never fires on.
    """
    silent = [source] if mark in (SILENT, BOTH) else []
    failed = [source] if mark in (FAILED, BOTH) else []
    path = log_dir() / f"{d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "date": str(d),
            "status": "published",
            "published": 3,
            "candidates": 200,
            "silent_sources": silent,
            "failed_sources": failed,
        }),
        encoding="utf-8",
    )


def _write(pattern: str, end: date = DAY) -> None:
    """Lay a pattern down ending at `end`, newest last."""
    for i, mark in enumerate(pattern):
        day = end - timedelta(days=len(pattern) - 1 - i)
        if mark != GAP:
            _row(day, mark)


def _run_days(pattern: str, end: date = DAY, source: str = ARXIV) -> list[Message]:
    """Run the whole pattern through the alert, one day at a time.

    Day by day, because that is how `uc daily` calls it: the row is recorded
    and the mail is sent on the same morning, so an alert that only ever fires
    when asked about the last day of a finished outage is not the thing being
    shipped.
    """
    backend = _Recorder()
    for i, mark in enumerate(pattern):
        day = end - timedelta(days=len(pattern) - 1 - i)
        if mark == GAP:
            continue
        # Row first, then the mail — `daily.py`'s order, and it matters: the
        # streak is counted out of `content/runs_log/` and today's row is part
        # of it, so an alert sent before `record()` would be counting yesterday.
        _row(day, mark, source)
        silent = [source] if mark in (SILENT, BOTH) else []
        failed = [source] if mark in (FAILED, BOTH) else []
        if not (silent or failed):
            continue  # the guard in `daily.py`: nothing absent, nothing to say
        notify_silent_sources(day, silent, failed, backend=backend)
    return backend.sent


# --------------------------------------------------------------------------
# The streak itself
# --------------------------------------------------------------------------


def test_a_run_of_failures_is_a_streak(repo):
    """The 406 case. Eleven days of it happened in September and this tier
    said nothing on any of them, because every one was a failure."""
    _write("FFF")

    assert unseen_streak(ARXIV, DAY) == 3
    # ...and it is the union doing it. Neither half can reach three.
    assert silent_streak(ARXIV, DAY) == 0
    assert failed_streak(ARXIV, DAY) == 3


def test_a_mixed_run_is_one_streak_not_two_short_ones(repo):
    """`S F S`: invisible for three days, and both old counts say one.

    This is the assertion that cannot pass without the union being read. If
    anything below still counted a single field, the streak would be 1 or 2
    and the first line would fail.
    """
    for pattern in ("SFS", "FSS", "SSF"):
        _write(pattern)
        assert unseen_streak(ARXIV, DAY) == 3, pattern
        assert silent_streak(ARXIV, DAY) < 3, pattern
        assert failed_streak(ARXIV, DAY) < 3, pattern


def test_a_day_that_answered_ends_the_streak(repo):
    """`F O F O F` is three incidents, not one state. The streak counts
    consecutive days, and a day the source worked is evidence it works."""
    _write("FOFOF")

    assert unseen_streak(ARXIV, DAY) == 1


def test_a_day_with_no_row_ends_the_streak(repo):
    """"We did not run" is not evidence either way — the distinction
    `outcome.py` exists for, kept where it decides whether a person is
    mailed."""
    _write("FF-FF")

    assert unseen_streak(ARXIV, DAY) == 2


def test_the_other_source_has_its_own_streak(repo):
    """A row names which source was absent, so arXiv failing for a week says
    nothing about OpenAlex."""
    _write("FFF")

    assert unseen_streak("collect.openalex", DAY) == 0


def test_each_day_keeps_its_own_reason(repo):
    """The count is one number; the days are not one fact. `unseen_days`
    returns both so the mail cannot describe a failure as silence."""
    _write("SFB")

    assert unseen_days(ARXIV, DAY) == [
        (DAY - timedelta(days=2), UNSEEN_SILENT),
        (DAY - timedelta(days=1), UNSEEN_FAILED),
        (DAY, UNSEEN_BOTH),
    ]


def test_a_day_on_both_lists_is_one_day(repo):
    """`content/runs_log/2026-10-06.json` is such a row: `collect.arxiv` in
    `silent_sources` and in `failed_sources` at once, because
    `silent_sources()` checks the stage and L0 froze the stage at OK. The
    streak must count that day once, not twice."""
    _write("BBB")

    assert unseen_streak(ARXIV, DAY) == 3
    assert [r for _, r in unseen_days(ARXIV, DAY)] == [UNSEEN_BOTH] * 3


# --------------------------------------------------------------------------
# What gets mailed
# --------------------------------------------------------------------------


def test_three_days_of_failure_mails_on_the_third(repo, monkeypatch):
    """🔴 The hole this issue is about. Before the change this sent nothing at
    all, and the only alert left was the job turning red."""
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")

    sent = _run_days("FFF")

    assert len(sent) == 1
    assert str(DAY) in sent[0].subject
    assert "arXiv" in sent[0].subject
    assert f"{SILENT_ALERT_DAYS} days" in sent[0].subject


def test_a_mixed_three_days_mails(repo, monkeypatch):
    """Every ordering of the mixed run, each from a clean log."""
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")

    for pattern in ("SFS", "FSS", "SSF", "FSF", "SFF", "FFS"):
        for path in log_dir().glob("*.json"):
            path.unlink()
        sent = _run_days(pattern)
        assert len(sent) == 1, f"{pattern} sent {len(sent)}"


def test_three_genuinely_silent_days_still_mail(repo, monkeypatch):
    """🔴 The widening guard. The point of this change is that more reaches a
    person, not that the definition moved — every streak that mailed before
    mails now."""
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")

    sent = _run_days("SSS")

    assert len(sent) == 1
    assert unseen_streak(ARXIV, DAY) == 3
    assert silent_streak(ARXIV, DAY) == 3, "the old count agrees on this one"


def test_a_broken_run_never_mails(repo, monkeypatch):
    """`F O F O F` again, now at the thing it protects: five bad days and no
    mail, because none of them is a three-day state."""
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")

    assert _run_days("FOFOF") == []


def test_it_mails_once_and_not_again(repo, monkeypatch):
    """The streak is the news; its continuing is not. Unchanged by the
    widening, and worth re-holding because the widening makes more streaks."""
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")

    # Six consecutive failures, the third of which lands on DAY.
    sent = _run_days("FFFFFF", end=DAY + timedelta(days=3))

    assert len(sent) == 1
    assert str(DAY) in sent[0].subject, "the day the streak reached three"


def test_the_mail_says_which_day_was_which(repo, monkeypatch):
    """🔴 A streak that spans two reasons must not be described as one.

    "arXiv has returned nothing for three days" is false of a day arXiv refused
    the connection, and it is false in the direction this project keeps
    getting wrong: it reports blindness as an observation.
    """
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")

    body = _run_days("SFS")[0].text

    assert "every request failed" in body
    assert "finished OK and returned nothing" in body
    for i in range(3):
        assert str(DAY - timedelta(days=i)) in body
    # The failed day's line and the silent days' lines are different lines.
    lines = [ln for ln in body.splitlines() if str(DAY - timedelta(days=1)) in ln]
    assert len(lines) == 1 and "every request failed" in lines[0]


def test_an_all_failed_mail_points_at_the_recorded_cause(repo, monkeypatch):
    """A failed day has an error recorded against it; a silent one has nothing
    to look at. The mail sends the reader to the right place."""
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")

    body = _run_days("FFF")[0].text

    assert "source_failures" in body
    assert "runs_log" in body
    assert "indexing lag" not in body, "no silent day in this streak to guess about"
    assert "uc status" in body, "but the standing instructions are still there"


def test_a_stage_name_never_reaches_the_reader(repo, monkeypatch):
    """1E's rule, re-held on the new body: `collect.arxiv` is a log name."""
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")

    sent = _run_days("SFB")[0]

    assert "collect.arxiv" not in sent.subject
    assert "collect.arxiv" not in sent.text


def test_no_recipient_is_reported_not_raised(repo, monkeypatch):
    """Unchanged guarantee: an alert with nowhere to go costs an alert."""
    monkeypatch.delenv("UC_ALERT_RECIPIENT", raising=False)
    _write("FFF")

    result = notify_silent_sources(DAY, [], [ARXIV])

    assert result["status"] == "no_alert_recipient"
    assert result["sources"] == [ARXIV]


# --------------------------------------------------------------------------
# The weekly summary: two counts, never one
# --------------------------------------------------------------------------


def test_the_weekly_counts_failed_days_apart_from_silent_ones(repo):
    """🔴 Not a bigger number — a second number.

    A week of `S S F F O O O` has two silent days' worth of "the source had
    nothing" and two days' worth of "we were blind". Summed, the mail would
    report four of something that happened twice.
    """
    from pipeline.notify import weekly_body, weekly_summary

    start = DAY
    _write("SSFFOOO", end=start + timedelta(days=6))

    summary = weekly_summary(end=start + timedelta(days=6))

    assert summary["silent_source_days"] == {ARXIV: 2}
    assert summary["failed_source_days"] == {ARXIV: 2}

    body = weekly_body(summary)
    assert "arXiv on 2 of 7 days" in body
    assert "returned nothing" in body
    assert "failed every request" in body


def test_a_day_on_both_lists_is_counted_once_in_each(repo):
    """The 10-06 shape again. Both fields name the day, so both counts see it
    — and neither inflates, because each counts days, not marks."""
    from pipeline.notify import weekly_summary

    _write("BBOOOOO", end=DAY + timedelta(days=6))

    summary = weekly_summary(end=DAY + timedelta(days=6))

    assert summary["silent_source_days"] == {ARXIV: 2}
    assert summary["failed_source_days"] == {ARXIV: 2}


def test_a_clean_week_says_nothing_about_either(repo):
    """The regression guard that 1E set for the silent heading, extended to
    the new one. A heading that appears every week reading zero is how a
    channel stops being read."""
    from pipeline.notify import weekly_body, weekly_summary

    _write("OOOOOOO", end=DAY + timedelta(days=6))

    summary = weekly_summary(end=DAY + timedelta(days=6))

    assert summary["silent_source_days"] == {}
    assert summary["failed_source_days"] == {}
    body = weekly_body(summary)
    assert "returned nothing" not in body
    assert "failed every request" not in body


def test_the_day_rows_name_the_reason(repo):
    """Per day, in the rows at the bottom of the mail, the two marks are
    different words — a reader scanning the week sees which was which."""
    from pipeline.notify import weekly_body, weekly_summary

    _write("SFOOOOO", end=DAY + timedelta(days=6))

    body = weekly_body(weekly_summary(end=DAY + timedelta(days=6)))

    def row(d: date) -> str:
        # The day rows, not the header line, which carries the range's dates.
        return next(
            ln for ln in body.splitlines() if ln.strip().startswith(str(d))
        )

    silent_row, failed_row = row(DAY), row(DAY + timedelta(days=1))
    assert "no arXiv" in silent_row and "failed" not in silent_row
    assert "arXiv failed" in failed_row
