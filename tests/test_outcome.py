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


GOOD = {"collect": "OK", "collect.arxiv": "OK", "collect.openalex": "OK"}


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

    html = build_archive().read_text(encoding="utf-8")
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
