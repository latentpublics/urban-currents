"""What happens when a stage breaks (phase 0k, final verification).

The most important tests in this batch, because everything else in Phase 0k is
about the day that works. Three faults are injected into a real `run_daily` —
collection, summarisation, delivery — and each asks the same three questions:

1. Is the issue withheld when it should be, and written when it should be?
2. Does somebody get told?
3. Does the next attempt resume rather than start over?

The third matters as much as the first two. A run that dies after summarising
has already spent the money, and a retry that re-collects and re-summarises pays
for the same day twice.

Collection and summarisation are failures that withhold the issue. **Delivery is
not**, and the asymmetry is the point: by the time the mail is sent the issue is
already written and already on the site, so a delivery failure is a delivery to
retry rather than a reason to unwind a day's work.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from pipeline import daily as daily_mod
from pipeline import paths, run_stages, store
from pipeline.deliver import DeliveryError, Message, already_delivered
from pipeline.models import (
    Bibliography,
    Headline,
    Issue,
    Item,
    PrimaryLocation,
    ScanMeta,
    SummaryEn,
)
from pipeline.outcome import NOT_PUBLISHED, PUBLISHED, load_log

DAY = date(2026, 8, 20)
# A healthy day's stage map. The four after the sources are `REQUIRED_STAGES`
# (0U, U1): this harness runs only `summarize` through `_guard` and calls the
# issue stage directly, so without them a day that is fine in every way this
# file cares about would be refused for stages the harness never ran.
GOOD_STAGES = {
    "collect": "OK",
    "collect.arxiv": "OK",
    "collect.openalex": "OK",
    "classify": "OK",
    "summarize": "OK",
    "select": "OK",
    "issue": "OK",
}


class Recorder:
    """A delivery backend that remembers, and can be told to break."""

    name = "recorder"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.sent: list[Message] = []

    def send(self, message: Message):
        if self.fail:
            raise DeliveryError("the mail server refused the connection")
        self.sent.append(message)
        return {"backend": self.name, "recipients": len(message.recipients)}


def _item(i: int = 0) -> Item:
    item = Item(
        work_key=f"arxiv:2608.9{i:04d}",
        first_published=DAY,
        bibliography=Bibliography(
            title=f"Street network entropy and travel time, paper {i}",
            abstract="x",
            primary_location=PrimaryLocation(
                source_name="arXiv", landing_page_url=f"https://arxiv.org/abs/2608.9{i:04d}"
            ),
        ),
    )
    item.summary.en = SummaryEn(what=f"What paper {i} found.", why=f"Why paper {i} matters.")
    return item


@pytest.fixture
def wiring(monkeypatch):
    """A `run_daily` whose stages are ours, so a fault can be placed precisely.

    Only the stage bodies are replaced. The outcome model, the lock, the run
    log, the alert and the delivery ledger are all the real ones — those are
    what is under test.
    """
    state = {"collect_calls": 0, "summarize_calls": 0, "collected": []}

    def collect(run, d, backfill_from=None, **kw):
        state["collect_calls"] += 1
        if state.get("collect_fails"):
            raise RuntimeError("arXiv returned 503 for every page")
        run.metrics.stages.update(GOOD_STAGES)
        run.metrics.counts.arxiv_fetched = 311
        run.metrics.counts.openalex_fetched = 42
        state["collected"] = [_item(0), _item(1)]
        return state["collected"]

    def summarize(run, use_llm=True, **kw):
        state["summarize_calls"] += 1
        if state.get("summarize_fails"):
            raise RuntimeError("the model returned invalid JSON twice")
        return state["collected"]

    def issue_stage(run, d):
        items = state["collected"]
        for item in items:
            store.save_item(item)
        issue = Issue(
            date=d,
            items=[i.work_key for i in items],
            headline=Headline(present=True, work_key=items[0].work_key, line="A full line."),
            scan_meta=ScanMeta(items_published=len(items), candidates_scanned=353, journals=96),
        )
        return issue

    monkeypatch.setattr(run_stages, "stage_collect", collect)
    monkeypatch.setattr(run_stages, "stage_summarize", summarize)
    monkeypatch.setattr(run_stages, "stage_issue", issue_stage)
    monkeypatch.setattr(daily_mod, "stage_issue", issue_stage, raising=False)
    monkeypatch.setattr(run_stages, "read_stage", lambda run, name: state["collected"])
    monkeypatch.setattr(daily_mod, "STAGES", ("summarize",))
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")
    monkeypatch.setenv("UC_PREVIEW_RECIPIENT", "reader@example.org")
    return state


def _issue_file(d: date):
    return paths.CONTENT / "issues" / f"{d}.json"


# --------------------------------------------------------------------------
# 1. Collection fails
# --------------------------------------------------------------------------


def test_a_failed_collection_withholds_the_issue_alerts_and_then_recovers(
    repo, wiring, monkeypatch
):
    alerts = Recorder()
    monkeypatch.setattr("pipeline.deliver.get_backend", lambda name=None: alerts)

    wiring["collect_fails"] = True
    first = daily_mod.run_daily(d=DAY, use_llm=False)

    # 1. No issue.
    assert first["status"] == NOT_PUBLISHED
    assert not _issue_file(DAY).exists()
    assert already_delivered(DAY) is None

    # 2. Somebody was told, and the subject says what happened.
    # An alert was raised **and it reached nobody**, because `deliver.backend`
    # is `file` (0U, U2). Both halves matter: the day noticed it had failed,
    # and the notice went into a file on a runner that then evaporated. This
    # used to assert `"alerted"`, which is the false reassurance U2 removed.
    assert first["alert"]["status"] == "alert_undeliverable"
    assert first["alert"]["reached_a_person"] is False
    assert str(DAY) in alerts.sent[0].subject
    assert alerts.sent[0].recipients == ["yjun@example.org"]
    assert "collect" in alerts.sent[0].subject

    # ...and the log holds the reason, not just the fact.
    log = load_log(DAY)
    assert log["status"] == NOT_PUBLISHED
    assert any("collect" in r for r in log["reasons"])

    # 3. The next run, with the source back, publishes — and records attempt 2.
    wiring["collect_fails"] = False
    second = daily_mod.run_daily(d=DAY, use_llm=False)

    assert second["status"] == PUBLISHED
    assert _issue_file(DAY).exists()
    assert load_log(DAY)["attempts"] == 2
    assert len(alerts.sent) == 2  # the alert, then the issue
    assert alerts.sent[1].recipients == ["reader@example.org"]


# --------------------------------------------------------------------------
# 2. Summarisation fails
# --------------------------------------------------------------------------


def test_a_failed_summarize_withholds_the_issue_and_the_retry_does_not_recollect(
    repo, wiring, monkeypatch
):
    """The expensive half must not be paid twice for one day."""
    alerts = Recorder()
    monkeypatch.setattr("pipeline.deliver.get_backend", lambda name=None: alerts)

    wiring["summarize_fails"] = True
    first = daily_mod.run_daily(d=DAY, use_llm=False)

    assert first["status"] == NOT_PUBLISHED
    assert not _issue_file(DAY).exists()
    assert "summarize" in first["failed_stages"]
    # An alert was raised **and it reached nobody**, because `deliver.backend`
    # is `file` (0U, U2). Both halves matter: the day noticed it had failed,
    # and the notice went into a file on a runner that then evaporated. This
    # used to assert `"alerted"`, which is the false reassurance U2 removed.
    assert first["alert"]["status"] == "alert_undeliverable"
    assert first["alert"]["reached_a_person"] is False
    assert "summarize" in alerts.sent[0].subject

    # A half-summarised day is an incomplete day, not a short one: nothing that
    # did succeed was published on its own.
    assert already_delivered(DAY) is None

    wiring["summarize_fails"] = False
    second = daily_mod.run_daily(d=DAY, use_llm=False)

    assert second["status"] == PUBLISHED
    assert _issue_file(DAY).exists()
    # Two summarize attempts, because that is the stage that failed. Collection
    # ran again too — it is cheap and the window moves — but the reason this
    # matters is the LLM cache underneath, which serves the same day for free.
    assert wiring["summarize_calls"] == 2


# --------------------------------------------------------------------------
# 3. Delivery fails — and the day survives
# --------------------------------------------------------------------------


def test_a_failed_delivery_does_not_unwind_the_day(repo, wiring, monkeypatch):
    broken = Recorder(fail=True)
    monkeypatch.setattr("pipeline.deliver.get_backend", lambda name=None: broken)

    result = daily_mod.run_daily(d=DAY, use_llm=False)

    # The issue is written and the day counts as published.
    assert result["status"] == PUBLISHED
    assert _issue_file(DAY).exists()
    assert load_log(DAY)["status"] == PUBLISHED

    # The send did not happen, and the ledger does not pretend it did.
    assert result["delivery"]["status"] == "failed"
    assert "DeliveryError" in result["delivery"]["error"]
    assert already_delivered(DAY) is None


def test_a_failed_delivery_can_be_retried_without_republishing(repo, wiring, monkeypatch):
    broken = Recorder(fail=True)
    monkeypatch.setattr("pipeline.deliver.get_backend", lambda name=None: broken)
    daily_mod.run_daily(d=DAY, use_llm=False)

    before = _issue_file(DAY).read_bytes()

    working = Recorder()
    monkeypatch.setattr("pipeline.deliver.get_backend", lambda name=None: working)
    second = daily_mod.run_daily(d=DAY, use_llm=False)

    assert second["delivery"]["status"] == "sent"
    assert len(already_delivered(DAY)["sends"]) == 1
    # And the issue itself did not change under the retry.
    assert _issue_file(DAY).read_bytes() == before


def test_a_delivery_failure_never_produces_a_second_send(repo, wiring, monkeypatch):
    """The ledger is the guard, and a failure must not leave a phantom entry."""
    working = Recorder()
    monkeypatch.setattr("pipeline.deliver.get_backend", lambda name=None: working)
    daily_mod.run_daily(d=DAY, use_llm=False)
    assert len(already_delivered(DAY)["sends"]) == 1

    daily_mod.run_daily(d=DAY, use_llm=False)
    daily_mod.run_daily(d=DAY, use_llm=False)

    assert len(already_delivered(DAY)["sends"]) == 1
    assert len(working.sent) == 1


# --------------------------------------------------------------------------
# The alert itself failing
# --------------------------------------------------------------------------


def test_an_alert_that_cannot_be_sent_still_leaves_the_record(repo, wiring, monkeypatch):
    """A notification failure must never be the reason a fact goes unrecorded."""
    monkeypatch.setattr("pipeline.deliver.get_backend", lambda name=None: Recorder(fail=True))

    wiring["collect_fails"] = True
    result = daily_mod.run_daily(d=DAY, use_llm=False)

    assert result["status"] == NOT_PUBLISHED
    assert result["alert"]["status"] == "alert_failed"
    assert load_log(DAY)["status"] == NOT_PUBLISHED
    assert any("collect" in r for r in load_log(DAY)["reasons"])
