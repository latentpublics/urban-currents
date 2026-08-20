"""A failed day is not a quiet day (phase 0k, X3).

The five tests the directive names, plus the lock behaviour `uc daily` depends
on. Every one of them is about the same sentence: when this service says a day
was quiet, it looked.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from pipeline import paths, store
from pipeline.metrics import Run
from pipeline.models import Bibliography, Headline, Issue, Item, ScanMeta
from pipeline.outcome import (
    NOT_PUBLISHED,
    PUBLISHED,
    QUIET,
    decide,
    load_log,
    looked,
    record,
    unpublished_dates,
)

DAY = date(2026, 8, 20)


def _run(stages: dict[str, str], **counts) -> Run:
    run = Run.for_date(DAY)
    run.metrics.stages.update(stages)
    for key, value in counts.items():
        setattr(run.metrics.counts, key, value)
    return run


# A run that observed the day **and produced an issue**. The four stages after
# the sources are `REQUIRED_STAGES` (0U, U1): a run that never summarised is not
# a good day, it is a day whose cards all read "Summary pending review."
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
# 1. a day whose collection raised writes no issue
# --------------------------------------------------------------------------


def test_a_failed_collection_writes_no_issue(repo):
    run = _run({"collect": "FAILED", "collect.arxiv": "FAILED", "collect.openalex": "OK"})
    outcome = decide(run, DAY, published_count=0)

    assert outcome.status == NOT_PUBLISHED
    assert outcome.writes_issue is False
    assert outcome.sends_email is False
    assert any("collect.arxiv" in r for r in outcome.reasons)


# --------------------------------------------------------------------------
# 2. a day that returned zero *successfully* is quiet
# --------------------------------------------------------------------------


def test_an_honest_zero_is_a_quiet_day(repo):
    """Counting zero and failing to count are different facts."""
    run = _run({**GOOD, "collect": "EMPTY"}, arxiv_candidates=0, journal_candidates=0)
    outcome = decide(run, DAY, published_count=0)

    assert outcome.status == QUIET
    assert outcome.writes_issue is True
    assert outcome.candidates == 0


def test_a_zero_with_a_failed_source_is_not_quiet(repo):
    """The case phase 0h shipped: nothing came back, and nobody checked why."""
    run = _run({"collect": "EMPTY", "collect.arxiv": "OK", "collect.openalex": "FAILED"})
    outcome = decide(run, DAY, published_count=0)

    assert outcome.status == NOT_PUBLISHED
    assert any("openalex" in r for r in outcome.reasons)


# --------------------------------------------------------------------------
# 3. budget exhaustion is a failure, not a short issue
# --------------------------------------------------------------------------


def test_budget_exhaustion_publishes_nothing_rather_than_a_truncated_issue(repo):
    run = _run(GOOD, arxiv_candidates=40, journal_candidates=30)
    outcome = decide(run, DAY, published_count=12, budget_exceeded=True)

    assert outcome.status == NOT_PUBLISHED
    assert outcome.published == 0
    assert any("budget" in r for r in outcome.reasons)


# --------------------------------------------------------------------------
# 4. half the scope is not the scope
# --------------------------------------------------------------------------


def test_a_partial_failure_is_not_published(repo):
    """arXiv answered, OpenAlex did not. The issue claims a scope; half of one
    is a different claim."""
    run = _run(
        {"collect": "OK", "collect.arxiv": "OK", "collect.openalex": "SKIPPED"},
        arxiv_candidates=55,
    )
    outcome = decide(run, DAY, published_count=9)

    assert outcome.status == NOT_PUBLISHED
    assert outcome.published == 0


def test_a_later_stage_failing_also_blocks_the_issue(repo):
    run = _run({**GOOD, "summarize": "FAILED"}, arxiv_candidates=40)
    outcome = decide(run, DAY, published_count=10)
    assert outcome.status == NOT_PUBLISHED
    assert "summarize" in outcome.failed_stages


def test_a_good_day_publishes(repo):
    run = _run(GOOD, arxiv_candidates=40, journal_candidates=30)
    outcome = decide(run, DAY, published_count=18)
    assert outcome.status == PUBLISHED
    assert outcome.writes_issue and outcome.sends_email


# --------------------------------------------------------------------------
# 5. the archive draws them differently
# --------------------------------------------------------------------------


def test_the_archive_distinguishes_a_quiet_day_from_a_blind_one(repo):
    """Same grey chip for both would undo everything above."""
    from pipeline.outcome import Outcome
    from pipeline.render.site import build_archive

    item = Item(
        work_key="arxiv:2608.00001",
        first_published=date(2026, 8, 19),
        bibliography=Bibliography(title="A Paper"),
    )
    store.save_item(item, today=date(2026, 8, 19))
    store.save_issue(Issue(
        date=date(2026, 8, 19),
        items=[item.work_key],
        headline=Headline(present=True, work_key=item.work_key, line="A line."),
        scan_meta=ScanMeta(items_published=1),
    ))
    store.save_issue(Issue(date=date(2026, 8, 18), items=[], quiet_day=True))
    record(Outcome(
        date=DAY, status=NOT_PUBLISHED,
        reasons=["collect.openalex finished FAILED"],
    ))

    # `build_archive` returns every page it wrote (one per month, 0R T3);
    # the landing page is first.
    html = build_archive()[0].read_text(encoding="utf-8")
    assert "a quiet day" in html
    assert "no issue" in html
    assert 'uc-row--missing' in html
    # And the missing day is not counted as an issue.
    assert 'data-date="2026-08-20"' in html


# --------------------------------------------------------------------------
# The log
# --------------------------------------------------------------------------


def test_every_outcome_is_logged_including_the_good_ones(repo):
    """A log of failures only would make its own silence ambiguous."""
    run = _run(GOOD, arxiv_candidates=10)
    record(decide(run, DAY, published_count=5))

    logged = load_log(DAY)
    assert logged["status"] == PUBLISHED
    assert logged["attempts"] == 1


def test_a_retry_increments_the_attempt_count(repo):
    from pipeline.outcome import Outcome

    record(Outcome(date=DAY, status=NOT_PUBLISHED, reasons=["first"]))
    record(Outcome(date=DAY, status=NOT_PUBLISHED, reasons=["second"]))
    assert load_log(DAY)["attempts"] == 2

    record(Outcome(date=DAY, status=PUBLISHED))
    assert load_log(DAY)["status"] == PUBLISHED


def test_unpublished_dates_are_the_catch_up_queue(repo):
    from pipeline.outcome import Outcome

    record(Outcome(date=date(2026, 8, 18), status=PUBLISHED))
    record(Outcome(date=date(2026, 8, 19), status=NOT_PUBLISHED, reasons=["x"]))
    record(Outcome(date=DAY, status=NOT_PUBLISHED, reasons=["y"]))

    queue = [r["date"] for r in unpublished_dates()]
    assert queue == ["2026-08-20", "2026-08-19"]


def test_a_missed_day_has_a_row_but_is_not_an_issue(repo):
    """It appears, and it is not counted.

    A gap in the list would read as "nothing happened", which is the claim we
    could not make. So the day gets a row carrying its reason — and no issue
    file, no entry in any published total.
    """
    from pipeline.outcome import Outcome
    from pipeline.render.site import archive_rows, archive_stats

    record(Outcome(date=DAY, status=NOT_PUBLISHED, reasons=["collect.openalex failed"]))

    rows = {r["date"]: r for r in archive_rows()}
    assert DAY.isoformat() in rows
    row = rows[DAY.isoformat()]
    assert row["missing"] is True
    assert row["quiet"] is False           # not the same claim
    assert "openalex" in row["reason"]

    assert not (paths.CONTENT / "issues" / f"{DAY}.json").exists()
    stats = archive_stats(list(rows.values()), {})
    assert stats["published_days"] == 0


# --------------------------------------------------------------------------
# The lock
# --------------------------------------------------------------------------


def test_a_lock_held_by_a_live_process_refuses(repo):
    import os

    from pipeline.daily import DailyLocked, acquire_lock, lock_path

    lock_path().parent.mkdir(parents=True, exist_ok=True)
    lock_path().write_text(
        json.dumps({"pid": os.getpid(), "started_at": __import__("time").time()}),
        encoding="utf-8",
    )
    with pytest.raises(DailyLocked):
        acquire_lock()


def test_a_lock_from_a_dead_process_is_reclaimed(repo):
    """Two stale lock files have already cost this project time."""
    import time as _time

    from pipeline.daily import acquire_lock, lock_path

    lock_path().parent.mkdir(parents=True, exist_ok=True)
    lock_path().write_text(
        json.dumps({"pid": 999999, "started_at": _time.time() - 60}), encoding="utf-8"
    )
    lock = acquire_lock()
    assert lock.acquired
    assert lock.reclaimed_from["pid"] == 999999
    lock.release()
    assert not lock_path().exists()


# --------------------------------------------------------------------------
# 6. the failure that reports success (X6)
# --------------------------------------------------------------------------


def test_a_source_that_returns_nothing_over_a_window_is_recorded(repo):
    """Found by running it: a 3-day window fetched 0 arXiv items, every stage OK.

    The window sat inside arXiv's indexing lag, so half our declared scope was
    missing and nothing in the data said so.
    """
    run = _run(GOOD, arxiv_fetched=0, openalex_fetched=11, journal_candidates=4)
    outcome = decide(run, DAY, published_count=4, window_days=3)

    assert outcome.silent_sources == ["collect.arxiv"]
    assert any("returned nothing" in e for e in run.metrics.errors)


def test_a_silent_source_does_not_veto_the_issue(repo):
    """The journal papers are real. Withholding them would trade partial for none."""
    run = _run(GOOD, arxiv_fetched=0, openalex_fetched=11, journal_candidates=4)
    outcome = decide(run, DAY, published_count=4, window_days=3)

    assert outcome.status == PUBLISHED
    assert outcome.writes_issue is True


def test_zero_over_a_single_day_is_not_called_silence(repo):
    """One empty day from one source is ordinary. A week of them is not."""
    run = _run(GOOD, arxiv_fetched=0, openalex_fetched=11)
    outcome = decide(run, DAY, published_count=4, window_days=1)

    assert outcome.silent_sources == []


def test_a_failed_source_is_counted_as_failed_not_as_silent(repo):
    """Two different faults must not be reported as one."""
    run = _run(
        {"collect": "OK", "collect.arxiv": "FAILED", "collect.openalex": "OK"},
        arxiv_fetched=0,
        openalex_fetched=11,
    )
    outcome = decide(run, DAY, published_count=0, window_days=7)

    assert outcome.status == NOT_PUBLISHED
    assert outcome.silent_sources == []
    assert any("collect.arxiv" in r for r in outcome.reasons)


def test_the_silence_survives_into_the_log(repo):
    run = _run(GOOD, arxiv_fetched=0, openalex_fetched=11)
    record(decide(run, DAY, published_count=4, window_days=7))

    assert load_log(DAY)["silent_sources"] == ["collect.arxiv"]


# --------------------------------------------------------------------------
# 7. a rehearsal does not enter the record (X6)
# --------------------------------------------------------------------------


def test_a_dry_run_leaves_no_row_in_the_run_log(repo, monkeypatch):
    """`status: published` for a date with no issue would make `uc status` lie.

    The run log answers "did this day get covered". A dry run's answer is no,
    however much work it did, so it stays in `runs/` where rehearsals belong.
    """
    from pipeline import daily as daily_mod, run_stages
    from pipeline.outcome import log_dir

    def fake_collect(run, d, backfill_from=None, **kw):
        run.metrics.stages.update(GOOD)
        run.metrics.counts.arxiv_fetched = 300
        run.metrics.counts.openalex_fetched = 40
        return []

    monkeypatch.setattr(run_stages, "stage_collect", fake_collect)
    monkeypatch.setattr(daily_mod, "STAGES", ())
    monkeypatch.setattr(run_stages, "read_stage", lambda run, name: [1, 2, 3])

    result = daily_mod.run_daily(d=DAY, dry_run=True, use_llm=False)

    assert result["dry_run"] is True
    assert result["published"] == 3
    assert load_log(DAY) is None
    assert not log_dir().exists() or not list(log_dir().glob("*.json"))


def test_a_dry_run_sends_no_alert_even_when_the_day_failed(repo, monkeypatch):
    from pipeline import daily as daily_mod, run_stages

    def broken_collect(run, d, backfill_from=None, **kw):
        run.metrics.stages.update({"collect": "FAILED", "collect.arxiv": "FAILED"})
        raise RuntimeError("arXiv is unreachable")

    monkeypatch.setattr(run_stages, "stage_collect", broken_collect)
    monkeypatch.setattr(daily_mod, "STAGES", ())
    monkeypatch.setattr(run_stages, "read_stage", lambda run, name: [])
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")

    result = daily_mod.run_daily(d=DAY, dry_run=True, use_llm=False)

    assert result["status"] == NOT_PUBLISHED
    assert "alert" not in result
