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

    # Bound before the patch: `run_stages.stage_collect` inside the stub would
    # be the stub itself. 0U had that recursion, and because a RecursionError
    # in collect reads as `collect: FAILED` — which produces the
    # `not_published` these tests assert — they passed without ever collecting
    # anything (0V, V6-4).
    real_collect = run_stages.stage_collect

    def collect(run, d, backfill_from=None, **kw):
        items = real_collect(run, d, fixture=True)
        run.metrics.stages["collect.arxiv"] = "OK"
        run.metrics.stages["collect.openalex"] = "OK"
        run.count("openalex_fetched", 0)
        return items

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
    """The symptom, stated as the reader would have met it.

    Looks where the rendered output actually is (0V, V6-3). 0U searched
    `content/` for the placeholder, and the placeholder is produced at render
    time into `runs/{run_id}/preview.html` and `email.html` — there is no
    `.html` under `content/` at all, so that search could not have failed
    however broken the pipeline was.
    """
    result = daily_mod.run_daily(d=sample_date, use_llm=True)

    assert result["status"] == NOT_PUBLISHED
    assert not _issue_file(sample_date).exists()
    assert store.load_issue(sample_date) is None

    rendered = [
        f for f in paths.RUNS.rglob("*.html")
        if PENDING_CARD in f.read_text(encoding="utf-8", errors="ignore")
    ]
    assert rendered == [], f"a placeholder card was rendered into {rendered}"

    # And nothing under `content/` either — belt and braces, and the assertion
    # 0U meant to make.
    archived = [
        f for f in paths.CONTENT.rglob("*")
        if f.is_file()
        and PENDING_CARD in f.read_text(encoding="utf-8", errors="ignore")
    ]
    assert archived == []


def test_an_unsummarised_item_really_does_render_the_placeholder(repo, sample_date):
    """The positive control the test above needs in order to mean anything.

    0U's version read the template and asserted the string was in the file,
    which proves the string exists and not that anything reaches it. This
    renders an issue containing a real unsummarised item and finds the
    placeholder in the output, so a reworded template or an unreachable
    fallback branch fails here instead of quietly making the search above
    unfalsifiable.
    """
    from pipeline.models import Bibliography, Headline, Issue, Item, ScanMeta
    from pipeline.render.preview import render_issue

    item = Item(
        work_key="arxiv:2608.99999",
        first_published=sample_date,
        bibliography=Bibliography(title="An unsummarised paper"),
    )
    issue = Issue(
        date=sample_date,
        items=[item.work_key],
        headline=Headline(present=False),
        scan_meta=ScanMeta(items_published=1, candidates_scanned=1, journals=1),
    )

    html = render_issue(issue, [item])

    assert PENDING_CARD in html


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
    """The other half of U4, exercised on a stage the rule actually names.

    Rewritten in 0V (V6-1). The first version asserted
    `stages.get("enrich") in ("SKIPPED", "PARTIAL", "OK")` — which every
    possible outcome satisfies — and `enrich` is not in `UPSTREAM_REQUIRED` at
    all, so the code path it described and the code path it touched were
    different ones.

    `summarize` **is** listed, as an upstream of `issue`. Skipped, it has to be
    walked past: that is the whole reason the rule is a list of failures rather
    than a rule about order. Failed, it must stop the walk.
    """
    from pipeline.metrics import Run
    from pipeline.run_stages import write_stage
    from pipeline.stages import UpstreamFailed, read_input

    run = Run.for_date(sample_date)
    selected = run_stages.stage_collect(run, sample_date, fixture=True)
    write_stage(run, "select", selected)
    run.stage("select", "OK")

    # No model, so the stage could not run. `issue` must still see the
    # selection rather than lose the day.
    run.stage("summarize", "SKIPPED")
    walked = read_input(run, "issue")
    assert [it.work_key for it in walked] == [it.work_key for it in selected]

    # The same stage, failed, is a different fact.
    run.stage("summarize", "FAILED")
    with pytest.raises(UpstreamFailed):
        read_input(run, "issue")


# --------------------------------------------------------------------------
# 3. arXiv returns nothing but 429
# --------------------------------------------------------------------------


class _Response:
    """One 429, carrying whatever headers this server was built with."""

    status_code = 429
    text = ""

    def __init__(self, headers: dict[str, str]):
        self.headers = headers

    def raise_for_status(self):  # pragma: no cover - never reached on 429
        raise AssertionError("raise_for_status must not be called for a 429")


class _Always429:
    """An arXiv that is throttling and never stops.

    State per instance (0V, V6-2). 0U wrote `self._Response.headers = {...}`
    in `__init__`, which assigns to the **class** — so the server built with
    `Retry-After: 900` left that header on every instance created afterwards
    in the same session. Running the module in order hid it; running the
    count-ceiling test alone with `-k` gave it a 900-second first wait, which
    breaches the sleep ceiling immediately and passes in zero iterations.
    """

    def __init__(self, retry_after: str | None = None):
        self.calls = 0
        self.headers = {"Retry-After": retry_after} if retry_after is not None else {}

    def get(self, url: str, params: Any = None):
        self.calls += 1
        return _Response(dict(self.headers))


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
    # The lower bounds matter as much: with the class-attribute leak 0V fixed
    # (V6-2) this test could finish in **zero** iterations and still pass every
    # ceiling, which is a test that proves the ceilings by never reaching them.
    assert server.calls > 1, "the collector gave up without retrying at all"
    assert 1 < server.calls <= ArxivCollector.MAX_RATE_LIMIT_RETRIES + 2
    cooldowns = [s for s in no_real_sleep if s >= ArxivCollector.RATE_LIMIT_COOLDOWN_S]
    assert 1 <= len(cooldowns) <= ArxivCollector.MAX_RATE_LIMIT_RETRIES
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

class _Always500:
    """A server that is up, unhappy, and consistent about it."""

    class _Response:
        status_code = 500
        headers: dict = {}
        text = ""

        def raise_for_status(self):
            raise RuntimeError("500 Server Error")

    def __init__(self):
        self.calls = 0

    def get(self, url: str, params=None):
        self.calls += 1
        return self._Response()


def test_the_ordinary_backoff_has_a_ceiling_too(repo, no_real_sleep):
    """V7: the 429 budget did not cover the 5xx path.

    `slept` was accumulated only in the rate-limit branch, so the exponential
    backoff for empty bodies and 5xx spent whatever it liked. The two paths
    interact: a 429 storm raises the attempt ceiling to nine *and* pushes the
    request interval to its 12-second cap, and 12x(64+128+256) is about 5,400
    seconds — no infinite loop, and well past `timeout-minutes: 45`.
    """
    from pipeline.collectors.arxiv import ArxivCollector
    from pipeline.metrics import Run

    server = _Always500()
    collector = ArxivCollector(Run.for_date(date(2026, 8, 20)), client=server)
    # The state a 429 storm leaves behind, which is what makes this expensive.
    collector.interval = 12.0
    collector.max_retries = 9

    with pytest.raises(RuntimeError):
        collector._fetch({"search_query": "cat:cs.CY"})

    assert sum(no_real_sleep) <= ArxivCollector.MAX_TOTAL_SLEEP_S
    assert max(no_real_sleep) <= ArxivCollector.MAX_BACKOFF_SLEEP_S
    assert server.calls >= 2, "it gave up without retrying"

