"""How the arXiv collector answers a refusal, and who hears about it (phase 1N).

2026-09-17..27 arXiv answered every request with `406 Not Acceptable`, eleven
days running, and stopped on 09-28 with nothing changed on our side. The cause
is unknown and nothing here claims otherwise. What is pinned is our response:

- a 4xx other than 429 fails at once instead of riding the 5xx backoff (N1);
- 429 still takes its own cooldown branch, and 5xx is still retried;
- the request names the type it parses and goes to https directly (N2);
- a required source that fails outright N days running is countable, and the
  count is of *consecutive* days (N3);
- 🔴 and through all of it, the day still publishes on its journal papers.

No network, no keys, and nothing here sleeps for real.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest

from pipeline.collectors.arxiv import ACCEPT, ArxivCollector, ArxivRejected
from pipeline.metrics import Run
from pipeline.outcome import PUBLISHED, decide

DAY = date(2026, 9, 19)


@pytest.fixture
def no_real_sleep(monkeypatch):
    """A clock that only moves when the collector sleeps, so the throttle sees
    the backoff it has just waited through — as it does on a real runner."""
    slept: list[float] = []
    clock = [1000.0]

    def sleep(s):
        slept.append(float(s))
        clock[0] += float(s)

    monkeypatch.setattr("pipeline.collectors.arxiv.time.sleep", sleep)
    monkeypatch.setattr("pipeline.collectors.arxiv.time.monotonic", lambda: clock[0])
    return slept


def _server(status: int, seen: list[httpx.Request], headers: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, headers=headers or {}, text="")

    return handler


def _collector(handler, run: Run | None = None) -> ArxivCollector:
    collector = ArxivCollector(run or Run.for_date(DAY))
    # The collector's own client, with only the transport swapped, so the
    # headers under test are the ones `_http()` really builds.
    real = collector._http()
    collector._client = httpx.Client(
        headers=real.headers,
        timeout=real.timeout,
        follow_redirects=True,
        transport=httpx.MockTransport(handler),
    )
    return collector


# --------------------------------------------------------------------------
# N1 — three branches, pinned separately
# --------------------------------------------------------------------------


def test_a_406_fails_at_once_and_is_not_retried(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    collector = _collector(_server(406, seen))

    with pytest.raises(ArxivRejected) as exc:
        collector._fetch({"search_query": "cat:cs.CY"})

    # ★ 1Q changed one thing here: a 406 now earns one *different* request (the
    # pre-1N shape, `test_arxiv_406_probe_1q.py`). The same request is still
    # never sent twice, and nothing is backed off — the only sleep is the
    # ordinary between-requests throttle.
    assert len(seen) == 2
    assert seen[0].headers["accept"] != seen[1].headers["accept"], "a 406 was asked again"
    assert no_real_sleep == [collector.interval], "a 406 was backed off as if it were transient"
    message = str(exc.value)
    assert "406" in message and "rejected the request immediately" in message
    # The sentence the 11 days recorded, and could not be told apart from a
    # flaky server by. It must not be what a refusal says now.
    assert "attempts" not in message
    assert collector.last_status == 406


@pytest.mark.parametrize("status", [400, 403, 404])
def test_every_other_4xx_is_a_refusal_too(repo, no_real_sleep, status):
    seen: list[httpx.Request] = []
    collector = _collector(_server(status, seen))
    with pytest.raises(ArxivRejected):
        collector._fetch({"search_query": "cat:cs.CY"})
    assert len(seen) == 1 and no_real_sleep == []


def test_a_5xx_is_still_retried_with_backoff(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    collector = _collector(_server(503, seen))

    with pytest.raises(RuntimeError) as exc:
        collector._fetch({"search_query": "cat:cs.CY"})

    assert not isinstance(exc.value, ArxivRejected)
    assert len(seen) == collector.max_retries
    assert len(no_real_sleep) == collector.max_retries
    assert "failed after" in str(exc.value)


def test_a_429_still_takes_the_cooldown_branch(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    collector = _collector(_server(429, seen))

    with pytest.raises(RuntimeError) as exc:
        collector._fetch({"search_query": "cat:cs.CY"})

    assert not isinstance(exc.value, ArxivRejected)
    assert "giving up rather than" in str(exc.value)
    assert len(seen) > 1, "429 stopped being retried"
    assert max(no_real_sleep) >= ArxivCollector.RATE_LIMIT_COOLDOWN_S


def test_the_406_eleven_days_cost(repo, no_real_sleep, monkeypatch):
    """N4-1, as arithmetic. With `request_interval_s: 5` the old path slept
    5 + 10 + 20 = 35 s a morning before giving up, which is the 35.3-35.8 s the
    Actions logs show for `collect.arxiv` on every 406 day. Pinned here so the
    number in the report is the code's, not a guess."""
    collector = _collector(_server(503, []))
    collector.interval = 5.0
    collector.max_retries = 3
    with pytest.raises(RuntimeError):
        collector._fetch({})
    assert sum(no_real_sleep) == 35.0


# --------------------------------------------------------------------------
# N2 — variables removed, not a fix
# --------------------------------------------------------------------------


def test_the_accept_header_actually_goes_out(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    collector = _collector(_server(406, seen))
    with pytest.raises(ArxivRejected):
        collector._fetch({})
    sent = seen[0].headers["accept"]
    assert sent == ACCEPT
    assert sent.startswith("application/atom+xml")
    # The fallback that keeps this header from causing a 406 the httpx default
    # (`*/*`) would not have.
    assert "*/*" in sent


def test_the_api_is_asked_over_https_directly(repo):
    from pipeline.config import cfg

    assert cfg("arxiv.api_url").startswith("https://")
    assert ArxivCollector(Run.for_date(DAY)).api_url.startswith("https://")


# --------------------------------------------------------------------------
# 🔴 N0-1 — arXiv dead, issue still published
# --------------------------------------------------------------------------


def test_a_refused_arxiv_still_publishes_the_day(repo, no_real_sleep):
    """The 1L regression guard, end to end through the collector this time.

    `collect()` still swallows the window's exception and returns `[]`; the
    stage verdict is still OK; `decide()` still publishes on what the journals
    gave. Only the record is better: it says *rejected*, not *3 attempts*.
    """
    run = Run.for_date(DAY)
    collector = _collector(_server(406, []), run=run)

    assert collector.collect(DAY) == []

    run.metrics.stages.update({
        "collect.arxiv": "OK", "collect.openalex": "OK", "collect": "OK",
        "summarize": "OK", "classify": "OK", "select": "OK", "issue": "OK",
    })
    outcome = decide(run, DAY, published_count=5)
    assert outcome.status == PUBLISHED
    assert outcome.published == 5

    obs = run.metrics.source_observations["collect.arxiv"]
    assert obs["failed_all"] is True and obs["http_status"] == 406
    (reason,) = run.metrics.source_failures["collect.arxiv"]
    assert "rejected the request immediately" in reason


# --------------------------------------------------------------------------
# N3 — the count is consecutive
# --------------------------------------------------------------------------


def _row(repo, d: date, failed: list[str]) -> None:
    from pipeline.outcome import log_dir

    path = log_dir() / f"{d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"date": str(d), "status": "published", "failed_sources": failed}),
        encoding="utf-8",
    )


def test_the_streak_counts_consecutive_days(repo):
    from pipeline.outcome import failed_streak

    a = "collect.arxiv"
    for i in range(5):
        _row(repo, DAY - timedelta(days=i), [a])
    assert failed_streak(a, DAY) == 5


def test_a_day_that_worked_resets_the_streak(repo):
    from pipeline.outcome import failed_streak

    a = "collect.arxiv"
    _row(repo, DAY, [a])
    _row(repo, DAY - timedelta(days=1), [a])
    _row(repo, DAY - timedelta(days=2), [])  # arXiv answered that day
    _row(repo, DAY - timedelta(days=3), [a])
    _row(repo, DAY - timedelta(days=4), [a])
    assert failed_streak(a, DAY) == 2
    assert failed_streak(a, DAY - timedelta(days=2)) == 0


def test_a_day_with_no_row_breaks_the_streak(repo):
    from pipeline.outcome import failed_streak

    a = "collect.arxiv"
    _row(repo, DAY, [a])
    _row(repo, DAY - timedelta(days=2), [a])
    assert failed_streak(a, DAY) == 1


def test_dead_sources_turns_red_at_n_and_not_before(repo):
    from typer.testing import CliRunner

    from pipeline.cli import app
    from pipeline.notify import FAILED_SOURCE_RED_DAYS, dead_required_sources

    a = "collect.arxiv"
    for i in range(FAILED_SOURCE_RED_DAYS - 1):
        _row(repo, DAY - timedelta(days=i), [a])
    assert dead_required_sources(DAY) == {}
    ok = CliRunner().invoke(app, ["dead-sources", "--date", str(DAY)])
    assert ok.exit_code == 0, ok.output

    _row(repo, DAY - timedelta(days=FAILED_SOURCE_RED_DAYS - 1), [a])
    assert dead_required_sources(DAY) == {a: FAILED_SOURCE_RED_DAYS}
    red = CliRunner().invoke(app, ["dead-sources", "--date", str(DAY)])
    assert red.exit_code == 1
    assert "[DEAD]" in red.output


def test_the_eleven_days_would_have_gone_red_on_the_second(repo):
    """The 09-17..27 rows, shaped as they were committed."""
    from pipeline.notify import dead_required_sources

    start = date(2026, 9, 17)
    _row(repo, start - timedelta(days=1), [])
    for i in range(11):
        _row(repo, start + timedelta(days=i), ["collect.arxiv"])
    _row(repo, start + timedelta(days=11), [])

    red = [
        str(start + timedelta(days=i))
        for i in range(12)
        if dead_required_sources(start + timedelta(days=i))
    ]
    assert red[0] == "2026-09-18"
    assert red[-1] == "2026-09-27"
    assert len(red) == 10


# --------------------------------------------------------------------------
# N3 — the workflow
# --------------------------------------------------------------------------


def _daily_yml() -> str:
    return (Path(__file__).resolve().parents[1] / ".github/workflows/daily.yml").read_text(
        encoding="utf-8"
    )


def test_the_dead_source_check_runs_after_the_commit():
    """🔴 The issue is published and pushed before the source is judged."""
    text = _daily_yml()
    commit = text.index("- name: Commit content")
    dead = text.index("- name: Fail the job if a required source is dead")
    assert commit < dead
    block = text[dead:]
    assert "uc dead-sources" in block and "exit 1" in block
    assert "always()" in block.splitlines()[1]
