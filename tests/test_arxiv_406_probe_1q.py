"""A 406 gets one different request, and the answer is written down (phase 1Q).

Q1/Q2. 1N stopped retrying refusals and added an explicit `Accept`. On
2026-10-06 the 406 came back on 1N's code, the day after a 200 on the same
code — so the static request is not the cause, but whether its shape is the
trigger is still open. A 406 is now answered **once per window** with the
pre-1N request, and the answer is written down:

- 200 → the `Accept` 1N added is what arXiv refuses, and the window recovers;
- 406 → the request's shape is not the trigger.

What must not move: 400/403/404 get no fallback, 5xx and timeouts keep their
backoff, 429 keeps its cooldown, a second 406 ends the window, and arXiv dying
completely still publishes the day (L0).

No network, no keys, and nothing here sleeps for real.
"""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from pipeline.collectors.arxiv import (
    ACCEPT,
    CURRENT,
    PRE_1N,
    PRE_1N_ACCEPT,
    USER_AGENT_TEMPLATE,
    ArxivCollector,
    ArxivRejected,
)
from pipeline.metrics import Run
from pipeline.outcome import PUBLISHED, decide

DAY = date(2026, 10, 6)

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <entry>
    <id>http://arxiv.org/abs/2610.01234v1</id>
    <published>2026-10-05T17:31:00Z</published>
    <title>Street-View Imagery and Pedestrian Volume</title>
    <summary>We train a model.</summary>
    <author><name>Rui Alvarez</name></author>
    <arxiv:primary_category term="cs.CV"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2610.02345v1</id>
    <published>2026-10-05T04:00:00Z</published>
    <title>Transit Accessibility</title>
    <summary>Panel of 43 metropolitan areas.</summary>
    <author><name>Hana Oyelaran</name></author>
    <arxiv:primary_category term="cs.CY"/>
  </entry>
</feed>
"""


@pytest.fixture
def no_real_sleep(monkeypatch):
    """A clock that only moves when the collector sleeps (as in the 1N tests)."""
    slept: list[float] = []
    clock = [1000.0]

    def sleep(s):
        slept.append(float(s))
        clock[0] += float(s)

    monkeypatch.setattr("pipeline.collectors.arxiv.time.sleep", sleep)
    monkeypatch.setattr("pipeline.collectors.arxiv.time.monotonic", lambda: clock[0])
    return slept


def _collector(handler, run: Run | None = None) -> ArxivCollector:
    collector = ArxivCollector(run or Run.for_date(DAY))
    real = collector._http()
    collector._client = httpx.Client(
        headers=real.headers,
        timeout=real.timeout,
        follow_redirects=True,
        transport=httpx.MockTransport(handler),
    )
    return collector


def _respond(seen: list[httpx.Request], *statuses: int, headers: dict | None = None):
    """Answer the n-th request with the n-th status (the last one repeats)."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status = statuses[min(len(seen) - 1, len(statuses) - 1)]
        body = ATOM if status == 200 else ""
        return httpx.Response(status, headers=headers or {}, text=body)

    return handler


def _refuse_the_1n_shape(seen: list[httpx.Request]):
    """An arXiv that refuses exactly 1N's Accept and serves anything else."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("accept") == ACCEPT:
            return httpx.Response(406, text="")
        return httpx.Response(200, text=ATOM)

    return handler


def _obs(collector: ArxivCollector) -> dict:
    return collector.run.metrics.source_observations.get("collect.arxiv", {})


# --------------------------------------------------------------------------
# Q1 — the fallback is the pre-1N request, byte for byte
# --------------------------------------------------------------------------


def test_an_explicit_star_is_not_the_pre_1n_request():
    """The measurement behind building the fallback from a pre-1N client.

    Same value, different order: httpx puts its own `Accept` first and appends
    one the caller sets. If this ever stops being true the fallback is still
    right — it is built the pre-1N way — but the reason for it is gone.
    """
    ua = USER_AGENT_TEMPLATE.format(email="x@example.com")
    url, params = "https://export.arxiv.org/api/query", {"start": 0}
    pre = httpx.Client(headers={"User-Agent": ua}).build_request("GET", url, params=params)
    explicit = httpx.Client(headers={"User-Agent": ua, "Accept": ACCEPT}).build_request(
        "GET", url, params=params, headers={"Accept": "*/*"}
    )
    assert pre.headers["accept"] == explicit.headers["accept"] == "*/*"
    assert pre.headers.raw != explicit.headers.raw


def test_the_fallback_is_byte_identical_to_the_pre_1n_request(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    collector = _collector(_respond(seen, 406, 406))
    params = {"search_query": "cat:cs.CY", "start": 0}
    with pytest.raises(ArxivRejected):
        collector._fetch(params)

    # How `_http()` looked before 1N: a User-Agent and nothing else. Built
    # here independently of `pre_1n_headers()`, so this is not the code
    # checking itself.
    with httpx.Client(
        headers={"User-Agent": seen[1].headers["user-agent"]}
    ) as pre_1n_client:
        expected = pre_1n_client.build_request("GET", collector.api_url, params=params)

    assert seen[1].headers.raw == expected.headers.raw
    assert seen[1].url == expected.url
    assert seen[1].headers["accept"] == PRE_1N_ACCEPT
    assert "cookie" not in seen[1].headers
    # The first request was 1N's, unchanged.
    assert seen[0].headers["accept"] == ACCEPT


def test_the_fallback_keeps_the_client_timeout(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    collector = _collector(_respond(seen, 406, 406))
    with pytest.raises(ArxivRejected):
        collector._fetch({})
    assert seen[1].extensions["timeout"] == collector._http().timeout.as_dict()


# --------------------------------------------------------------------------
# Q1 — both results are recorded
# --------------------------------------------------------------------------


def test_406_then_200_recovers_the_window_and_says_how(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    run = Run.for_date(DAY)
    collector = _collector(_refuse_the_1n_shape(seen), run=run)

    items = collector.collect(DAY)

    assert len(items) == 2, "the fallback's 200 should have recovered the day"
    assert len(seen) == 2
    obs = _obs(collector)
    (entry,) = obs["accept_fallback"]
    assert entry["recovered"] is True
    assert entry["trigger_status"] == 406 and entry["fallback_status"] == 200
    assert entry["trigger_accept"] == ACCEPT and entry["fallback_accept"] == "*/*"
    assert entry["fallback_shape"] == PRE_1N
    assert entry["window"] == f"{DAY}..{DAY}"
    assert obs["accept_shape"] == PRE_1N
    assert obs["failed_all"] is False and obs["http_status"] == 200
    assert "collect.arxiv" not in run.metrics.source_failures


def test_406_then_406_ends_the_window_and_says_so(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    run = Run.for_date(DAY)
    collector = _collector(_respond(seen, 406, 406, headers={"Server": "edge"}), run=run)

    assert collector.collect(DAY) == []

    assert len(seen) == 2, "a refused fallback must not be followed by another"
    obs = _obs(collector)
    (entry,) = obs["accept_fallback"]
    assert entry["recovered"] is False and entry["fallback_status"] == 406
    assert entry["fallback_headers"]["server"] == "edge"
    assert obs["failed_all"] is True and obs["http_status"] == 406
    assert obs["accept_shape"] is None
    (reason,) = run.metrics.source_failures["collect.arxiv"]
    assert "rejected the request immediately" in reason
    assert "pre-1N shape" in reason and "got HTTP 406" in reason


def test_a_normal_day_records_the_current_shape_and_no_fallback(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    collector = _collector(_respond(seen, 200))
    assert len(collector.collect(DAY)) == 2
    obs = _obs(collector)
    assert obs["accept_shape"] == CURRENT
    assert "accept_fallback" not in obs and "rejected_headers" not in obs


def test_a_fallback_that_never_answers_is_recorded_not_retried(repo, no_real_sleep):
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(406, text="")
        raise httpx.ConnectTimeout("timed out", request=request)

    collector = _collector(handler)
    with pytest.raises(ArxivRejected) as exc:
        collector._fetch({})
    assert len(seen) == 2
    (entry,) = _obs(collector)["accept_fallback"]
    assert entry["fallback_status"] is None and entry["recovered"] is False
    assert "ConnectTimeout" in entry["fallback_error"]
    assert "did not complete" in str(exc.value)
    assert no_real_sleep == [collector.interval], "the probe was backed off"


# --------------------------------------------------------------------------
# Q1 — the boundary: 406 only, once per window, interval kept
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_no_fallback_for_any_other_4xx(repo, no_real_sleep, status):
    seen: list[httpx.Request] = []
    collector = _collector(_respond(seen, status))
    with pytest.raises(ArxivRejected):
        collector._fetch({})
    assert len(seen) == 1 and no_real_sleep == []
    obs = _obs(collector)
    assert "accept_fallback" not in obs
    assert obs["rejected_status"] == status  # Q2 still records who said no


def test_5xx_keeps_its_backoff_and_gets_no_fallback(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    collector = _collector(_respond(seen, 503))
    with pytest.raises(RuntimeError) as exc:
        collector._fetch({})
    assert not isinstance(exc.value, ArxivRejected)
    assert len(seen) == collector.max_retries
    assert all(r.headers["accept"] == ACCEPT for r in seen)
    assert "accept_fallback" not in _obs(collector)


def test_a_timeout_keeps_its_backoff_and_gets_no_fallback(repo, no_real_sleep):
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        raise httpx.ReadTimeout("slow", request=request)

    collector = _collector(handler)
    with pytest.raises(RuntimeError) as exc:
        collector._fetch({})
    assert "failed after" in str(exc.value)
    assert len(seen) == collector.max_retries
    assert "accept_fallback" not in _obs(collector)


def test_429_keeps_its_cooldown_and_gets_no_fallback(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    collector = _collector(_respond(seen, 429))
    with pytest.raises(RuntimeError) as exc:
        collector._fetch({})
    assert "giving up rather than" in str(exc.value)
    assert max(no_real_sleep) >= ArxivCollector.RATE_LIMIT_COOLDOWN_S
    assert all(r.headers["accept"] == ACCEPT for r in seen)
    assert "accept_fallback" not in _obs(collector)


def test_the_fallback_keeps_the_request_interval(repo, no_real_sleep):
    collector = _collector(_respond([], 406, 406))
    collector.interval = 5.0
    with pytest.raises(ArxivRejected):
        collector._fetch({})
    # First request: nothing to wait for. Fallback: the interval, and only it.
    assert no_real_sleep == [5.0]


def test_once_per_window_even_when_the_recovered_window_is_refused_later(
    repo, no_real_sleep
):
    """Page 0: 406, fallback 200. Page 1 goes out in the shape that worked and
    is refused — the window's fallback is spent, so the window ends there."""
    seen: list[httpx.Request] = []
    run = Run.for_date(DAY)
    collector = _collector(_respond(seen, 406, 200, 406), run=run)
    collector.page_size = 1  # two pages for a two-result feed

    assert collector.collect(DAY) == []

    assert len(seen) == 3
    assert [r.headers["accept"] for r in seen] == [ACCEPT, "*/*", "*/*"]
    assert len(_obs(collector)["accept_fallback"]) == 1
    (reason,) = run.metrics.source_failures["collect.arxiv"]
    assert "fallback already spent" in reason


def test_each_backfill_window_gets_its_own_one_fallback(repo, no_real_sleep):
    seen: list[httpx.Request] = []
    collector = _collector(_respond(seen, 406))
    start = date(2026, 9, 1)
    collector.collect(date(2026, 9, 14), backfill_from=start)

    # Two windows, two requests each, never three.
    assert len(seen) == 4
    entries = _obs(collector)["accept_fallback"]
    assert [e["window"] for e in entries] == ["2026-09-01..2026-09-07", "2026-09-08..2026-09-14"]
    assert all(e["fallback_status"] == 406 for e in entries)


# --------------------------------------------------------------------------
# Q2 — headers are kept, secrets are not
# --------------------------------------------------------------------------


def test_refusal_headers_are_kept_and_scrubbed(repo, no_real_sleep, monkeypatch):
    contact = "operator-mailbox@example.org"
    monkeypatch.setenv("CONTACT_EMAIL", contact)
    hostile = {
        "Server": "cloudflare",
        "Via": "1.1 varnish",
        "CF-Ray": "8c0ffee-ICN",
        "X-Served-By": "cache-icn",
        "X-Echo-UA": f"urban-currents/0.2 (mailto:{contact})",
        "X-Debug": "api_key=sk-live-SHOULDNOTSURVIVE",
        "Set-Cookie": "__cf_bm=COOKIESECRETVALUE; path=/",
        "Authorization": "Bearer AUTHSECRETVALUE123",
        "X-Api-Key": "HEADERKEYSECRET",
        "X-Session-Token": "SESSIONSECRET42",
        "Proxy-Authenticate": "Basic realm=PROXYSECRET",
    }
    run = Run.for_date(DAY)
    collector = _collector(_respond([], 406, headers=hostile), run=run)
    collector.collect(DAY)

    obs = _obs(collector)
    kept = obs["rejected_headers"]
    assert kept["server"] == "cloudflare" and kept["cf-ray"] == "8c0ffee-ICN"
    assert kept["via"] == "1.1 varnish" and kept["x-served-by"] == "cache-icn"
    assert kept["set_cookie_present"] is True
    for name in ("set-cookie", "authorization", "x-api-key", "x-session-token",
                 "proxy-authenticate"):
        assert name not in kept

    # Every path the observation leaves by: the runs_log row, the result dict
    # that is printed to the Actions log, and the run's own metrics.
    run.metrics.stages.update({
        "collect.arxiv": "OK", "collect.openalex": "OK", "collect": "OK",
        "summarize": "OK", "classify": "OK", "select": "OK", "issue": "OK",
    })
    run.count("openalex_fetched", 9)
    outcome = decide(run, DAY, published_count=4, window_days=7)
    from pipeline.outcome import record

    written = record(outcome).read_text(encoding="utf-8")
    everything = "\n".join([
        written,
        json.dumps(outcome.to_dict() if hasattr(outcome, "to_dict") else outcome.__dict__,
                   default=str),
        run.metrics.model_dump_json(),
    ])
    for secret in (contact, "SHOULDNOTSURVIVE", "COOKIESECRETVALUE", "AUTHSECRETVALUE123",
                   "HEADERKEYSECRET", "SESSIONSECRET42", "PROXYSECRET"):
        assert secret not in everything, secret
    assert "cloudflare" in written


def test_header_volume_is_bounded(repo, no_real_sleep):
    noisy = {f"X-Noise-{i}": "v" * 1000 for i in range(60)}
    collector = _collector(_respond([], 406, headers=noisy))
    with pytest.raises(ArxivRejected):
        collector._fetch({})
    kept = _obs(collector)["rejected_headers"]
    assert len(kept) <= 21  # twenty headers and the set-cookie flag
    assert all(len(v) <= 200 for v in kept.values() if isinstance(v, str))


# --------------------------------------------------------------------------
# 🔴 L0 — arXiv dead, the day still publishes
# --------------------------------------------------------------------------


def _all_ok(run: Run) -> None:
    run.metrics.stages.update({
        "collect.arxiv": "OK", "collect.openalex": "OK", "collect": "OK",
        "summarize": "OK", "classify": "OK", "select": "OK", "issue": "OK",
    })


def test_arxiv_dead_through_the_fallback_still_publishes(repo, no_real_sleep):
    run = Run.for_date(DAY)
    collector = _collector(_respond([], 406, 406), run=run)
    assert collector.collect(DAY) == []
    _all_ok(run)
    run.count("openalex_fetched", 9)

    outcome = decide(run, DAY, published_count=5, window_days=7)

    assert outcome.status == PUBLISHED and outcome.published == 5
    assert outcome.failed_sources == ["collect.arxiv"]
