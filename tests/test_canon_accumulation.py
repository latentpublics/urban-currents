"""Filling the citation base must never cost a day (phase 0T, V1).

`uc accumulate-canon` was written in 0k and called from nowhere. The pending
queue and the resolved store were both last written on 2026-08-13 while the
pipeline ran every day since, and the OpenAlex day budget went almost entirely
unspent — $0.0004 of $1.00 on 2026-08-18.

Wiring it into `uc daily` puts a network chore after the issue has published,
which is the dangerous shape: a background task that can rewrite a good day as a
failure. **That is what these tests are for.**
"""

from __future__ import annotations

from datetime import date

import pytest

from pipeline import paths
from pipeline.graph import daily_canon


# --------------------------------------------------------------------------
# V1 — the day's verdict is untouchable
# --------------------------------------------------------------------------


def test_a_canon_failure_is_recorded_and_not_raised(repo, monkeypatch):
    """Any exception at all. The `except` in `_accumulate_canon` is wide on
    purpose: `TimeBudgetExceeded` and `Interrupted` are `RuntimeError`s, and
    letting either escape would reach `run_daily`'s handler and rewrite a
    published day as `not_published`."""
    from pipeline.daily import _accumulate_canon
    from pipeline.metrics import Run

    run = Run.for_date(date(2026, 8, 20))

    def boom(*a, **k):
        raise RuntimeError("OpenAlex fell over")

    monkeypatch.setattr(daily_canon, "accumulate_day", boom)
    out = _accumulate_canon(run, date(2026, 8, 20), deadline=None)

    assert out["status"] == "FAILED"
    assert "OpenAlex fell over" in out["error"]
    assert run.metrics.stages["canon"] == "FAILED"


@pytest.mark.parametrize("exc", ["TimeBudgetExceeded", "Interrupted"])
def test_running_out_of_time_in_the_chore_does_not_unwind_the_day(repo, monkeypatch, exc):
    """The specific pair that would otherwise be caught by `run_daily`."""
    import pipeline.daily as daily_mod
    from pipeline.metrics import Run

    error = getattr(daily_mod, exc)

    def boom(*a, **k):
        raise error("out of time")

    monkeypatch.setattr(daily_canon, "accumulate_day", boom)
    out = daily_mod._accumulate_canon(Run.for_date(date(2026, 8, 20)),
                                      date(2026, 8, 20), deadline=None)

    assert out["status"] == "FAILED"


def test_a_dry_run_does_not_spend_anything(repo):
    from pipeline.daily import _accumulate_canon
    from pipeline.metrics import Run

    run = Run.for_date(date(2026, 8, 20))

    assert _accumulate_canon(run, date(2026, 8, 20), None, dry_run=True)["status"] == "SKIPPED"
    assert _accumulate_canon(run, date(2026, 8, 20), None, smoke=True)["status"] == "SKIPPED"


def test_it_can_be_switched_off_entirely(repo, monkeypatch):
    from pipeline.daily import _accumulate_canon
    from pipeline.metrics import Run

    import pipeline.daily as daily_mod

    monkeypatch.setattr(daily_mod, "cfg", lambda k, d=None: False if k == "canon.accumulate_daily" else d)
    out = _accumulate_canon(Run.for_date(date(2026, 8, 20)), date(2026, 8, 20), None)

    assert out["status"] == "SKIPPED"


# --------------------------------------------------------------------------
# Stopping is a normal ending
# --------------------------------------------------------------------------


def test_the_deadline_is_read_from_a_property_not_a_method(repo):
    """`Deadline.remaining` is a property. Reading it through a `callable()`
    test silently returned infinity, which would have let the chore run past the
    deadline it exists to respect."""
    from pipeline.daily import Deadline

    d = Deadline()

    assert daily_canon._seconds_left(d) > 0
    assert daily_canon._seconds_left(None) == float("inf")


def test_a_sigterm_means_no_time_left(repo):
    from pipeline.daily import Deadline

    d = Deadline()
    d.signalled = True

    assert daily_canon._seconds_left(d) == 0.0


def test_resolution_stops_before_the_deadline_and_says_so(repo, monkeypatch):
    class NoTime:
        remaining = daily_canon.SAFETY_MARGIN_S - 1
        signalled = False

    rows, cost, stopped = daily_canon._resolve(["openalex:W1"], deadline=NoTime())

    assert rows == []
    # Either reason is a correct stop and neither is an alert: with no key the
    # client is absent, and with one the deadline check fires first.
    assert stopped in ("deadline", "no OpenAlex client")
    assert cost == 0.0


def test_the_budget_is_what_is_left_not_the_whole_share(repo):
    """Collection runs first and its spend is already on the clock. Taking the
    fraction of the gross budget would let the chore exceed it on a day when
    collection was expensive."""
    from pipeline.metrics import Run

    run = Run.for_date(date(2026, 8, 20))
    run.add_cost("openalex_usd", 0.4)

    assert daily_canon._openalex_spent(run) == pytest.approx(0.4)


# --------------------------------------------------------------------------
# V2 — the queue has to be able to drain
# --------------------------------------------------------------------------


def test_ids_that_never_answer_are_parked_not_retried_forever(repo):
    """Measured: a batch of 3,000 returned 2,548 works. The other 452 are
    merged, withdrawn, or never existed — and the queue is ordered by how often
    our corpus cites them, so a much-cited dead id sorts to the front and is
    asked for again every single day."""
    assert daily_canon.MAX_ATTEMPTS == 3

    paths.STATE.mkdir(parents=True, exist_ok=True)
    daily_canon._park({"openalex:W404"}, {"openalex:W404": 3})

    assert "openalex:W404" in daily_canon.load_unresolvable()


def test_a_single_miss_does_not_park_an_id(repo):
    """An empty response can also be a bad minute at the API. Parking on one
    miss would quietly shrink the canon."""
    assert daily_canon.MAX_ATTEMPTS > 1


def test_attempts_survive_a_rewrite_of_the_queue(repo):
    paths.STATE.mkdir(parents=True, exist_ok=True)
    daily_canon._write_pending(["openalex:W1", "openalex:W2"], {"openalex:W1": 2})

    assert daily_canon.load_attempts()["openalex:W1"] == 2
    assert daily_canon.load_pending() == ["openalex:W1", "openalex:W2"]


def test_the_pending_filter_is_not_quadratic(repo):
    """It was: the resolved set was built inside the comprehension condition, so
    Python rebuilt it per element. 116,630 ids against a 17,000-entry set is
    about two billion operations, and the first live run sat in it long enough
    to look hung."""
    import inspect

    source = inspect.getsource(daily_canon.accumulate_day)

    assert "done = {" in source
    assert 'if i not in {r["openalex_id"]' not in source


# --------------------------------------------------------------------------
# V3 — resolving is daily, republishing is not
# --------------------------------------------------------------------------


def test_accumulation_never_rebuilds_the_published_canon(repo):
    """`Still cited` publishes from `candidates.json` every day. If a chore
    rebuilt it, the card would move without anybody deciding that it should."""
    import inspect

    source = inspect.getsource(daily_canon)

    assert "build_candidates" not in source


def test_a_rebuild_reports_what_moved_in_the_top_30(repo):
    from pipeline.graph.canon import top_diff

    before = {"candidates": [
        {"openalex_id": "openalex:W1", "title": "Stays", "subfield": "Transportation"},
        {"openalex_id": "openalex:W2", "title": "Leaves", "subfield": "Ecology"},
    ]}
    after = {"candidates": [
        {"openalex_id": "openalex:W1", "title": "Stays", "subfield": "Transportation"},
        {"openalex_id": "openalex:W3", "title": "Enters", "subfield": "Ecology"},
    ]}

    d = top_diff(before, after)

    assert d["entered"] == ["Enters"]
    assert d["left"] == ["Leaves"]
    assert d["unchanged"] == 1
    assert d["transport_share_before"] == 0.5
    assert d["transport_share_after"] == 0.5


# --------------------------------------------------------------------------
# V4 — an old run must not stop the base being built
# --------------------------------------------------------------------------


def test_a_stage_file_from_an_older_schema_is_skipped_not_fatal(repo):
    """60 run directories still carry the `lens` field 0k removed, and one of
    them raised a `ValidationError` that killed the whole reference base."""
    from pipeline.stages import read_stage

    class _R:
        dir = paths.RUNS / "run_2026-08-05"

    (_R.dir / "stages").mkdir(parents=True, exist_ok=True)
    (_R.dir / "stages" / "classify.jsonl").write_text(
        '{"work_key": "arxiv:2608.00001", "bibliography": {"title": "x"}, "lens": null}\n',
        encoding="utf-8",
    )

    # Strict by default: inside a run this must still be an error.
    with pytest.raises(Exception):
        read_stage(_R, "classify")

    # And **recovered**, not dropped. Measured across the archive, simply
    # skipping these cost 866 items carrying 7,001 references — a hole in the
    # base this batch exists to finish filling.
    items = read_stage(_R, "classify", old_schema=True)
    assert len(items) == 1
    assert items[0].work_key == "arxiv:2608.00001"
