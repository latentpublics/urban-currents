"""Three failures that used to look like successes (phase 0U).

`tests/test_failure_injection.py` covers the failures that already announced
themselves: a stage raises, `_guard` records FAILED, the day is withheld. These
three did not raise. Each one ran to completion, wrote something, and reported a
result that a person reading `uc status` would have called fine:

1. **The LLM fails on every call.** `summarize_items` returns SKIPPED, which
   `looked()` never inspected, so the verdict was `published` and the issue went
   out with every card reading *"Summary pending review."* — a digest of papers
   nobody read, mailed to a reader as though it were the product.
2. **`select` fails.** `read_input` walks backwards for the most recent stage
   that produced data, cannot tell *skipped* from *failed*, and hands `issue`
   the whole classified candidate pool. `uc run` published it unranked and
   unchosen; `uc daily` survived only by accident, because `looked()` refuses
   the day afterwards for an unrelated reason.
3. **arXiv returns nothing but 429.** The rate-limit branch raised its own
   retry ceiling on every 429, so the loop could not terminate. The run sat in
   `collect` sleeping 90 seconds at a time until GitHub killed the job.

The unattended case is the whole point. All three are survivable when a person
is watching the terminal; none of them is when the next thing that happens is a
mail going out at 06:00.

No network, no keys, and nothing here sleeps for real.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from pipeline import daily as daily_mod
from pipeline import paths, run_stages, store
from pipeline.llm import LLMResponse, LLMUnavailable
from pipeline.outcome import NOT_PUBLISHED, decide, looked

PENDING_CARD = "Summary pending review."


def _issue_file(d: date):
    return paths.CONTENT / "issues" / f"{d}.json"


@pytest.fixture
def offline_daily(monkeypatch):
    """`uc daily`, with the fixture collector standing in for the network.

    Everything after `collect` is the real pipeline: the same stage order, the
    same `read_input`, the same `decide()`. Only the source of the items is
    replaced, because the failures under test are downstream of it.
    """

    def collect(run, d, backfill_from=None, **kw):
        return run_stages.stage_collect(run, d, fixture=True)

    monkeypatch.setattr(run_stages, "stage_collect", collect)
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")
    monkeypatch.setenv("UC_PREVIEW_RECIPIENT", "reader@example.org")


# --------------------------------------------------------------------------
# 1. The LLM fails on every call
# --------------------------------------------------------------------------


@pytest.fixture
def dead_llm(monkeypatch):
    """A configured provider that fails every call — not a missing key.

    The distinction matters. A missing key is the state `skips.py` was written
    for and is visible in `uc status`; this is the provider accepting the
    request and refusing it, which produces the same SKIPPED status by a route
    nobody is watching.
    """
    calls = {"n": 0}

    def boom(system: str, user: str) -> LLMResponse:
        calls["n"] += 1
        raise LLMUnavailable("gemini: 503 backend overloaded")

    from pipeline.summarize import run as summarize_run

    real = summarize_run.LLMClient

    def client(*a, **kw):
        return real(*a, caller=boom, **kw)

    monkeypatch.setattr(summarize_run, "LLMClient", client)
    return calls


def test_a_total_llm_failure_is_refused_rather_than_published(repo, sample_date, dead_llm):
    run = run_stages.run_all(sample_date, fixture=True, use_llm=True)

    # The stage did the honest thing: it tried, got nothing, and said SKIPPED.
    assert dead_llm["n"] >= 1, "the client was never called, so nothing was proven"
    assert run.metrics.stages["summarize"] == "SKIPPED"

    # And the verdict layer now reads that. Before U1 this returned True.
    ok, reasons = looked(run)
    assert ok is False
    assert any("summarize" in r for r in reasons)

    outcome = decide(run, sample_date, 3)
    assert outcome.status == NOT_PUBLISHED
    assert "summarize" in outcome.skipped_stages

    # And the items really are the ones that would have rendered as
    # placeholders — the refusal is not passing for some unrelated reason.
    items = run_stages.read_stage(run, "summarize") or []
    assert items, "the fixture produced no items, so nothing was summarised or not"
    assert all(not (it.summary.en and it.summary.en.what) for it in items)


def test_a_total_llm_failure_publishes_no_pending_cards(repo, sample_date, dead_llm, offline_daily):
    """The symptom, stated as the reader would have met it."""
    result = daily_mod.run_daily(d=sample_date, use_llm=True)

    assert result["status"] == NOT_PUBLISHED
    assert not _issue_file(sample_date).exists()

    # Nothing was mailed, and nothing on disk carries the placeholder. The
    # search is over the whole of `content/`, not just today's issue: the
    # failure being pinned is a card reaching a reader, and any file under
    # `content/` is a file the site publishes.
    published = [
        p for p in paths.CONTENT.rglob("*")
        if p.is_file() and p.suffix in (".json", ".html")
        and PENDING_CARD in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert published == []
    assert store.load_issue(sample_date) is None


def test_the_placeholder_string_still_exists():
    """Guard against the search above passing for the wrong reason.

    If the template stopped saying *"Summary pending review."* the `rglob`
    would find nothing and the test would go green while the bug returned
    under a new wording. Read from the real repository, not the temporary
    root: the template is source, not output.
    """
    template = (
        Path(__file__).resolve().parent.parent
        / "pipeline" / "render" / "templates" / "item_card.html.j2"
    ).read_text(encoding="utf-8")

    assert PENDING_CARD in template


# --------------------------------------------------------------------------
# 2. `select` fails
# --------------------------------------------------------------------------


@pytest.fixture
def broken_select(monkeypatch):
    def explode(run, *a, **kw):
        raise RuntimeError("the ranker raised on a malformed score")

    monkeypatch.setattr(run_stages, "stage_select", explode)


def _candidate_pool(run) -> int:
    return len(run_stages.read_stage(run, "classify") or [])


def test_uc_run_does_not_publish_the_candidate_pool(repo, sample_date, broken_select):
    run = run_stages.run_all(sample_date, fixture=True, use_llm=False)

    assert run.metrics.stages["select"] == "FAILED"
    # `issue` refuses rather than reaching past it. That is the fix: without it
    # the stage read `classify.jsonl` and published all of it.
    assert run.metrics.stages["issue"] == "FAILED"
    assert _candidate_pool(run) > 0, "no pool means the test proves nothing"
    assert store.load_issue(sample_date) is None
    assert not _issue_file(sample_date).exists()

    ok, reasons = looked(run)
    assert ok is False
    assert any("select" in r for r in reasons)


def test_uc_daily_does_not_publish_the_candidate_pool(
    repo, sample_date, broken_select, offline_daily
):
    result = daily_mod.run_daily(d=sample_date, use_llm=False)

    assert result["status"] == NOT_PUBLISHED
    assert not _issue_file(sample_date).exists()
    assert "select" in result["failed_stages"]


def test_a_skipped_upstream_stage_is_still_walked_past(repo, sample_date):
    """The other half of U4, and the reason it is a list rather than a rule.

    `enrich` is SKIPPED on every run without a Springer key. If refusing to
    walk back applied to skipping too, no day would ever publish.
    """
    run = run_stages.run_all(sample_date, fixture=True, use_llm=False)

    assert run.metrics.stages.get("enrich") in ("SKIPPED", "PARTIAL", "OK")
    assert run.metrics.stages["select"] == "OK"
    assert run.metrics.stages["issue"] == "OK"


# --------------------------------------------------------------------------
# 3. arXiv returns nothing but 429
# --------------------------------------------------------------------------


class _Always429:
    """An arXiv that is throttling and never stops."""

    class _Response:
        status_code = 429
        headers: dict[str, str] = {}
        text = ""

        def raise_for_status(self):  # pragma: no cover - never reached on 429
            raise AssertionError("raise_for_status must not be called for a 429")

    def __init__(self, retry_after: str | None = None):
        self.calls = 0
        if retry_after is not None:
            self._Response.headers = {"Retry-After": retry_after}

    def get(self, url: str, params: Any = None):
        self.calls += 1
        return self._Response()


@pytest.fixture
def no_real_sleep(monkeypatch):
    """Count the waiting instead of doing it."""
    slept: list[float] = []

    monkeypatch.setattr(
        "pipeline.collectors.arxiv.time.sleep", lambda s: slept.append(float(s))
    )
    return slept


def test_an_arxiv_throttling_forever_stops_by_itself(repo, no_real_sleep):
    from pipeline.collectors.arxiv import ArxivCollector
    from pipeline.metrics import Run

    server = _Always429()
    collector = ArxivCollector(Run.for_date(date(2026, 8, 20)), client=server)

    with pytest.raises(RuntimeError) as exc:
        collector._fetch({"search_query": "cat:cs.CY"})

    # It gave up on purpose, and says so. The old loop never got here at all.
    assert "giving up rather than" in str(exc.value)

    # Bounded three ways, because a hang is a hang whichever one runs out.
    assert server.calls <= ArxivCollector.MAX_RATE_LIMIT_RETRIES + 2
    cooldowns = [s for s in no_real_sleep if s >= ArxivCollector.RATE_LIMIT_COOLDOWN_S]
    assert len(cooldowns) <= ArxivCollector.MAX_RATE_LIMIT_RETRIES
    assert sum(cooldowns) <= ArxivCollector.MAX_RATE_LIMIT_SLEEP_S


def test_a_long_retry_after_exhausts_the_clock_not_the_count(repo, no_real_sleep):
    """The two ceilings fail differently and both have to hold.

    A server answering instantly runs out of attempts; one asking for ten
    minutes runs out of budget on the first or second wait. Without the second
    ceiling, six honoured `Retry-After: 900`s would be an hour and a half.
    """
    from pipeline.collectors.arxiv import ArxivCollector
    from pipeline.metrics import Run

    server = _Always429(retry_after="900")
    collector = ArxivCollector(Run.for_date(date(2026, 8, 20)), client=server)

    with pytest.raises(RuntimeError) as exc:
        collector._fetch({"search_query": "cat:cs.CY"})

    assert "giving up rather than" in str(exc.value)
    assert sum(no_real_sleep) <= ArxivCollector.MAX_RATE_LIMIT_SLEEP_S + 60
    assert server.calls <= 2


def test_a_throttled_collection_fails_the_day_rather_than_hanging(repo, no_real_sleep):
    """What the ceiling buys: a verdict instead of a killed job.

    A 429 storm now ends `collect` as FAILED, which `looked()` already refuses.
    Before, the day ended as `interrupted` — killed by the platform's
    `timeout-minutes` with nothing said about why.
    """
    from pipeline.collectors.arxiv import ArxivCollector
    from pipeline.metrics import Run

    run = Run.for_date(date(2026, 8, 20))
    server = _Always429()

    def collect(_run, d, **kw):
        collector = ArxivCollector(_run, client=server)
        return collector._fetch({"search_query": "cat:cs.CY"})

    from pipeline.run_stages import _guard

    _guard(run, "collect", lambda: collect(run, date(2026, 8, 20)))

    assert run.metrics.stages["collect"] == "FAILED"
    ok, reasons = looked(run)
    assert ok is False
    assert any("collect" in r for r in reasons)
