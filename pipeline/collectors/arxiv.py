"""arXiv collector (PRD §5.1).

The arXiv API asks for at least 3 seconds between requests and a contact address
in the User-Agent; both are honoured here, along with three backoff retries
for the failures a retry can change (5xx, timeouts, empty bodies). A 4xx other
than 429 is not retried at all — see `ArxivRejected`. A 406 alone gets one
*different* request, in the pre-1N shape, per window — see `_fallback`.

An Item must stand up on arXiv metadata alone — the matching OpenAlex Work often
does not exist yet on the day a preprint appears, and waiting for it would mean
publishing nothing.

Raw Atom responses are written to ``runs/{run_id}/raw/arxiv/`` before parsing.
Re-parsing after a parser fix must not cost another round of API calls, and
yesterday's run has to stay reproducible.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator, Optional
from xml.etree import ElementTree as ET

import httpx

from ..config import cfg, contact_email
from ..metrics import Run
from ..models import Author, Bibliography, Ids, Item, PrimaryLocation, Provenance
from ..redact import redact
from .base import ARXIV_SOURCE_ID, arxiv_doi, clean_text, normalize_arxiv_id

# The stage name this collector reports under. Used for the 1L facts recorded
# beside the stage verdict, never for the verdict itself.
SOURCE = "collect.arxiv"

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

USER_AGENT_TEMPLATE = "urban-currents/0.2 (Phase 0 research scan; mailto:{email})"

# ★ 1N, N2. Said out loud rather than left to httpx's `*/*`.
#
# 2026-09-17..27 every request came back `406 Not Acceptable`, eleven days in a
# row, and then stopped on 09-28 with nothing changed on our side. The cause is
# **not known** and this does not claim to be it. 406 is literally a
# content-negotiation refusal, though, so it is the one candidate the status
# code itself points at — and naming the type we parse costs nothing. The
# `*/*;q=0.1` tail keeps any representation acceptable, so this header cannot
# itself produce a 406 that the old default would not have.
#
# A removed variable, not a fix: if 406 comes back, this is already ruled out.
ACCEPT = "application/atom+xml, application/xml;q=0.9, */*;q=0.1"

# ★ 1Q, Q1. The request shape before 1N, kept so a 406 can be answered with it.
#
# Before 1N this collector sent no `Accept` at all and httpx filled in `*/*`.
# An explicit `*/*` is the same value but **not the same request**: httpx puts
# its own default first in the header list and appends one the caller sets,
# so the bytes on the wire differ in order. Measured on httpx 0.28.1 — and a
# WAF that fingerprints a request is exactly the kind of thing that reads
# order. So the fallback is not "1N's request with another Accept"; it is the
# request a client configured the pre-1N way builds, header for header (see
# `pre_1n_headers`). The one deliberate difference is the URL: pre-1N went to
# `http://` and followed a redirect, and the request it actually completed —
# the one after the redirect — is this one, over https. `api_url` stays put.
PRE_1N = "pre-1N"
CURRENT = "1N"
PRE_1N_ACCEPT = "*/*"

# ★ 1Q, Q2. Which response headers a refusal leaves behind.
#
# Enough to say who answered (`Server`, `Via`, a CDN's `CF-*`, a WAF's `X-*`)
# and nothing a credential could ride in on. Names are matched first, then
# every kept value still goes through `redact()` — an `X-` header can echo
# whatever we sent, and the User-Agent we send carries the contact address.
# `Set-Cookie` is never kept; that one was set is recorded as a yes/no.
_DIAGNOSTIC_HEADERS = frozenset({
    "server", "via", "date", "content-type", "content-length", "retry-after",
    "vary", "age", "cache-control", "x-cache",
})
_DIAGNOSTIC_PREFIXES = ("cf-", "x-")
_NEVER_KEEP = ("cookie", "auth", "token", "key", "secret", "session", "password")
MAX_KEPT_HEADERS = 20
MAX_HEADER_VALUE = 200


def pre_1n_headers() -> httpx.Headers:
    """The headers a pre-1N client sent: its User-Agent over httpx's defaults.

    Built by configuring a client exactly as `_http()` did before 1N and
    reading back what it holds, rather than by writing the list out, so a
    change in httpx's defaults changes both sides of the comparison together.
    """
    with httpx.Client(
        headers={"User-Agent": USER_AGENT_TEMPLATE.format(email=contact_email())}
    ) as client:
        return httpx.Headers(client.headers)


def diagnostic_headers(response) -> dict[str, Any]:
    """The part of a refusal's headers worth keeping, scrubbed. Never raises."""
    try:
        kept: dict[str, Any] = {}
        set_cookie = False
        for name, value in response.headers.items():
            low = name.lower()
            if low == "set-cookie":
                set_cookie = True
                continue
            if any(word in low for word in _NEVER_KEEP):
                continue
            if low not in _DIAGNOSTIC_HEADERS and not low.startswith(_DIAGNOSTIC_PREFIXES):
                continue
            if len(kept) >= MAX_KEPT_HEADERS:
                break
            kept[low] = redact(str(value))[:MAX_HEADER_VALUE]
        kept["set_cookie_present"] = set_cookie
        return kept
    except Exception:  # noqa: BLE001 - a diagnostic must not become the failure
        return {"unreadable": True}


class ArxivRejected(RuntimeError):
    """arXiv refused the request outright (a 4xx other than 429). Not retried.

    ★ 1N, N1. The 406 of 2026-09-17..27 went through the 5xx backoff — three
    attempts and 35 seconds of sleep a morning, eleven mornings — and was
    written down as *"failed after 3 attempts"*, which reads exactly like a
    flaky server. Nobody could tell from the record that it was a refusal that
    no retry would ever change. This type is that distinction: its message says
    "rejected immediately" and carries the status code.
    """

    def __init__(self, status: int, reason: str, url: str, note: str = ""):
        self.status = status
        super().__init__(
            f"arXiv rejected the request immediately: HTTP {status} {reason} "
            f"(not retried — a {status} does not change on retry){note} "
            f"for url '{url}'"
        )


@dataclass
class ArxivPage:
    entries: list[dict]
    total_results: int
    start: int


class ArxivCollector:
    def __init__(self, run: Run, client: Optional[httpx.Client] = None):
        self.run = run
        self.interval = float(cfg("arxiv.request_interval_s", 3.0))
        self.max_retries = int(cfg("arxiv.max_retries", 3))
        self.page_size = int(cfg("arxiv.page_size", 200))
        self.api_url = cfg("arxiv.api_url", "https://export.arxiv.org/api/query")
        self.categories = list(cfg("arxiv.categories", []) or [])
        self._client = client
        self._last_request = 0.0
        # ★ 1L, L2. The last HTTP status seen, so a run that never got a usable
        # response still records *what* it got. None means no request completed.
        self.last_status: Optional[int] = None
        # ★ 1Q, Q1. Which request shape this window is using, whether its one
        # fallback has been spent, and every fallback taken this run. Reset per
        # window by `_collect_window`; a bare `_fetch()` starts fresh.
        self._shape = CURRENT
        self._fallback_spent = False
        self._window: Optional[str] = None
        self._fallbacks: list[dict[str, Any]] = []
        self._last_ok_shape: Optional[str] = None

    # -- HTTP ------------------------------------------------------------

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={
                    "User-Agent": USER_AGENT_TEMPLATE.format(email=contact_email()),
                    "Accept": ACCEPT,
                },
                timeout=60.0,
                follow_redirects=True,
            )
        return self._client

    def _get(self, params: dict) -> httpx.Response:
        """One request, in whichever shape this window is using."""
        if self._shape == PRE_1N:
            return self._send_pre_1n(params)
        return self._http().get(self.api_url, params=params)

    def _send_pre_1n(self, params: dict) -> httpx.Response:
        """The pre-1N request, sent through the same client and transport.

        Built as a bare `httpx.Request` so the client's 1N headers are not
        merged into it, and without the client's cookie jar: pre-1N's request
        each morning went out on a fresh client with no cookies, and a 406 that
        sets one (a WAF challenge, say) must not make the fallback a request
        pre-1N never sent. The timeout is carried over by hand because
        `build_request` is what normally attaches it.
        """
        client = self._http()
        request = httpx.Request(
            "GET",
            self.api_url,
            params=params,
            headers=pre_1n_headers(),
            extensions={"timeout": client.timeout.as_dict()},
        )
        return client.send(request)

    def _fallback(self, params: dict, refused) -> str:
        """★ 1Q, Q1. A 406 is answered once with the pre-1N request. Not a retry.

        1N's rule stands — a refusal is not asked again, because the same
        request gets the same answer. This sends a **different** request, the
        one shape whose result separates two explanations:

        - it returns 200: the `Accept` header 1N added is what arXiv refuses,
          and the window is recovered rather than lost;
        - it returns 406 too: the request's shape is not the trigger, and the
          question narrows to IP, rate or a WAF on arXiv's side.

        Once per window, 406 only, no loop: a second refusal ends the window
        exactly as a first one did before this existed. The throttle is
        honoured; the 429 and 5xx branches are not entered from here.
        """
        self._fallback_spent = True
        entry: dict[str, Any] = {
            "window": self._window,
            "trigger_status": refused.status_code,
            "trigger_accept": ACCEPT,
            "trigger_headers": diagnostic_headers(refused),
            "fallback_shape": PRE_1N,
            "fallback_accept": PRE_1N_ACCEPT,
            "fallback_status": None,
            "recovered": False,
        }
        self._fallbacks.append(entry)
        note = ""
        try:
            self._throttle()
            r = self._send_pre_1n(params)
            self.last_status = r.status_code
            entry["fallback_status"] = r.status_code
            if 200 <= r.status_code < 300:
                entry["recovered"] = True
                # The rest of this window's pages go out the way that worked.
                self._shape = PRE_1N
                return r.text
            entry["fallback_headers"] = diagnostic_headers(r)
            note = (
                f"; asked once more in the pre-1N shape (Accept: {PRE_1N_ACCEPT}) "
                f"and got HTTP {r.status_code}"
            )
        except Exception as e:  # noqa: BLE001 - the probe's failure is recorded, not raised
            entry["fallback_error"] = redact(f"{type(e).__name__}: {e}")[:300]
            note = (
                f"; asked once more in the pre-1N shape (Accept: {PRE_1N_ACCEPT}) "
                f"and that request did not complete"
            )
        finally:
            self.run.observe_source(SOURCE, accept_fallback=list(self._fallbacks))
        raise ArxivRejected(
            refused.status_code,
            getattr(refused, "reason_phrase", "") or "",
            str(getattr(refused, "url", self.api_url)),
            note=note,
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_request = time.monotonic()

    # A 429 from arXiv means "you have been asking too fast for a while"; the
    # ordinary 3/6/12s backoff is far too short and just earns another 429.
    RATE_LIMIT_COOLDOWN_S = 90.0

    # ★ Ceilings on that patience (phase 0U, U3).
    #
    # The extra patience was implemented as `attempts = max(attempts, attempt + 3)`,
    # which **never terminates**: the ceiling is re-raised to three past the
    # counter on every 429, so `attempt < attempts` stays true forever while each
    # pass sleeps at least 90 seconds. `Deadline.check()` runs *between* stages,
    # so nothing inside the collector could stop it — the run would sit there
    # until GitHub's `timeout-minutes: 45` killed the job and the day was
    # recorded `interrupted`.
    #
    # The intent was right and is kept: a 429 should not consume an ordinary
    # attempt. It now buys extra attempts **from a fixed allowance** instead of
    # from an unbounded one. Two ceilings because they fail differently — a
    # server answering instantly with 429 exhausts the count, and one honouring a
    # long `Retry-After` exhausts the clock.
    MAX_RATE_LIMIT_RETRIES = 6
    MAX_RATE_LIMIT_SLEEP_S = 600.0

    # The ordinary backoff needs a ceiling of its own (phase 0V, V7).
    #
    # `MAX_RATE_LIMIT_SLEEP_S` only counted sleeps taken in the 429 branch, and
    # the 5xx/empty-body path sleeps `interval * 2**attempt` without counting
    # anything. Those two interact: a 429 storm raises `attempts` toward its
    # ceiling of nine *and* pushes `interval` to its 12-second cap, so a run of
    # 5xx after that sleeps 12x(64+128+256) ~ 5,376s — inside no infinite loop
    # and outside `timeout-minutes: 45`.
    #
    # Two numbers rather than one: a per-sleep cap so no single wait is absurd,
    # and a total that spans **both** branches so the collector cannot spend
    # the day's budget in small pieces.
    MAX_BACKOFF_SLEEP_S = 60.0
    MAX_TOTAL_SLEEP_S = 900.0

    def _fetch(self, params: dict) -> str:
        last: Optional[Exception] = None
        attempts = self.max_retries
        attempt = 0
        rate_limited = 0
        slept = 0.0
        while attempt < attempts:
            self._throttle()
            try:
                r = self._get(params)
                if r.status_code == 429:
                    retry_after = float(r.headers.get("Retry-After") or 0)
                    wait = max(self.RATE_LIMIT_COOLDOWN_S, retry_after)
                    last = RuntimeError("429 rate limited")
                    rate_limited += 1
                    if (
                        rate_limited > self.MAX_RATE_LIMIT_RETRIES
                        or slept + wait > self.MAX_RATE_LIMIT_SLEEP_S
                    ):
                        raise RuntimeError(
                            f"arXiv rate limited {rate_limited} time(s) and "
                            f"{slept:.0f}s slept; giving up rather than waiting "
                            f"past the day's budget"
                        )
                    # Rate limiting is recoverable and worth more patience than a
                    # transient error, so it does not consume a normal attempt —
                    # but the allowance it draws on is finite.
                    attempts = min(
                        self.max_retries + self.MAX_RATE_LIMIT_RETRIES, attempt + 3
                    )
                    attempts = max(attempts, attempt + 1)
                    # Slow down permanently for the rest of the run. Returning to
                    # the old cadence after a 429 just earns the next one.
                    self.interval = min(self.interval * 1.5, 12.0)
                    time.sleep(wait)
                    slept += wait
                    attempt += 1
                    continue
                # ★ 1L, L2. The last status this source actually returned.
                # A 200 with nothing in it and a request that never completed
                # are the same empty list once `collect()` is done.
                self.last_status = r.status_code
                # ★ 1N, N1. Any other 4xx is a refusal, and asking again gets
                # the same refusal. 429 never reaches here (its branch is above
                # and stays exactly as it was).
                if 400 <= r.status_code < 500:
                    # ★ 1Q, Q2. Who said no, before the refusal is raised.
                    self.run.observe_source(
                        SOURCE,
                        rejected_status=r.status_code,
                        rejected_headers=diagnostic_headers(r),
                    )
                    # ★ 1Q, Q1. 406 only, and once per window. 400, 403 and
                    # 404 are not about the request's shape; a fallback there
                    # would add a request and prove nothing.
                    if r.status_code == 406 and not self._fallback_spent:
                        text = self._fallback(params, r)
                        self._last_ok_shape = self._shape
                        return text
                    raise ArxivRejected(
                        r.status_code,
                        getattr(r, "reason_phrase", "") or "",
                        str(getattr(r, "url", self.api_url)),
                        note=(
                            f"; sent in the {self._shape} shape, fallback "
                            f"already spent this window"
                            if r.status_code == 406 else ""
                        ),
                    )
                r.raise_for_status()
                self._last_ok_shape = self._shape
                return r.text
            except ArxivRejected:
                raise
            except Exception as e:  # noqa: BLE001
                last = e
                # A ceiling breach is a decision, not a transient error: it must
                # leave the loop rather than be retried like a flaky 5xx.
                if isinstance(e, RuntimeError) and "giving up rather than" in str(e):
                    raise
                # arXiv returns empty bodies or 5xx under load; back off — but
                # inside the same budget the 429 branch draws on (0V, V7).
                wait = min(self.interval * (2**attempt), self.MAX_BACKOFF_SLEEP_S)
                if slept + wait > self.MAX_TOTAL_SLEEP_S:
                    raise RuntimeError(
                        f"arXiv backoff would pass {self.MAX_TOTAL_SLEEP_S:.0f}s "
                        f"of waiting in one request ({slept:.0f}s already slept); "
                        f"giving up rather than waiting past the day's budget"
                    ) from e
                time.sleep(wait)
                slept += wait
                attempt += 1
        raise RuntimeError(f"arXiv request failed after {attempt} attempts: {last}")

    # -- Query -----------------------------------------------------------

    @staticmethod
    def date_range_query(categories: list[str], start: date, end: date) -> str:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        # arXiv's submittedDate range is inclusive of both endpoints, in UTC.
        lo = start.strftime("%Y%m%d") + "0000"
        hi = end.strftime("%Y%m%d") + "2359"
        return f"({cats}) AND submittedDate:[{lo} TO {hi}]"

    # The legacy arXiv API refuses `start` beyond 10,000 with a 500, so a long
    # range has to be split into windows that each stay under that ceiling.
    # Seven days runs ~2,200 items across our seven categories — a wide margin.
    PAGING_LIMIT = 10000
    WINDOW_DAYS = 7

    def collect(
        self,
        d: date,
        backfill_from: Optional[date] = None,
        max_pages: Optional[int] = None,
    ) -> list[Item]:
        start = backfill_from or d
        items: dict[str, Item] = {}
        pages_used = 0
        attempted = 0
        failed = 0

        for w_start, w_end in self._windows(start, d):
            attempted += 1
            try:
                got, pages = self._collect_window(
                    w_start, w_end, None if max_pages is None else max_pages - pages_used
                )
            except Exception as e:  # noqa: BLE001
                # One bad window must not discard the rest of a 90-day backfill.
                # The gap is recorded so the report can say which days are thin.
                failed += 1
                reason = f"window {w_start}..{w_end}: {type(e).__name__}: {e}"
                self.run.error(f"collect.arxiv: {reason} failed")
                # ★ 1L, L3. The same sentence, but somewhere that survives the
                # runner. `run.error()` reaches `runs/{run_id}/metrics.json` and
                # `runs/` is thrown away; this reaches `content/runs_log/`.
                # D261 recorded exactly this gap ("the reason disappears with
                # the runner") and proposed only printing to stdout, which
                # Actions keeps for 90 days. This keeps it for good.
                self.run.source_failure(SOURCE, reason)
                continue
            pages_used += pages
            for item in got:
                items.setdefault(item.work_key, item)
            if max_pages is not None and pages_used >= max_pages:
                break

        # ★ 1L, L1/L2. Two facts the stage verdict cannot carry.
        #
        # L1: a daily run has exactly one window, so one dead window means the
        # collector returns `[]` while every stage stays OK — and
        # `silent_sources` then writes down "reported OK and returned nothing",
        # which is the *wrong fact*. Silence and failure are different, and this
        # is the line that tells them apart. `failed_all` is deliberately "every
        # window we tried died", not "any window died": a backfill that loses
        # one window of thirteen is thin, not blind.
        #
        # L2: an empty result with no exception at all — a query arXiv rejected
        # or silently corrected — is invisible to L1 because nothing raised.
        # `total_results` straight from the feed is what separates it.
        #
        # 🔴 Neither touches `stages["collect.arxiv"]`. See L0: the verdict is
        # read by `outcome.looked()`, and a failed arXiv collection must not
        # cost the day its journal papers.
        self.run.observe_source(
            SOURCE,
            windows=attempted,
            windows_failed=failed,
            failed_all=bool(attempted and failed == attempted),
            items=len(items),
            # Recorded here as well as per-page, because a window that never
            # returned a page leaves no other trace of what the wire said.
            http_status=self.last_status,
            # ★ 1Q, Q1. Which request shape the last successful response came
            # back to: "1N" normally, "pre-1N" if a 406 was recovered by the
            # fallback, None if nothing succeeded.
            accept_shape=self._last_ok_shape,
        )

        return sorted(items.values(), key=lambda it: it.work_key)

    def _windows(self, start: date, end: date) -> Iterator[tuple[date, date]]:
        if (end - start).days < self.WINDOW_DAYS:
            yield start, end
            return
        cursor = start
        while cursor <= end:
            stop = min(cursor + timedelta(days=self.WINDOW_DAYS - 1), end)
            yield cursor, stop
            cursor = stop + timedelta(days=1)

    def _collect_window(
        self, start: date, end: date, max_pages: Optional[int]
    ) -> tuple[list[Item], int]:
        query = self.date_range_query(self.categories, start, end)
        # ★ 1Q, Q1. Every window starts in the current shape with its one
        # fallback unspent, so a backfill tests the question again per window
        # instead of carrying one morning's answer through thirteen.
        self._shape = CURRENT
        self._fallback_spent = False
        self._window = f"{start}..{end}"
        items: list[Item] = []
        offset = 0
        page_no = 0

        while True:
            params = {
                "search_query": query,
                "start": offset,
                "max_results": self.page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            xml = self._fetch(params)
            self.run.write_raw(f"arxiv/{start}_{end}_p{page_no:03d}.xml", xml)
            page = parse_atom(xml)
            if page_no == 0:
                # ★ 1L, L2. `total_results` is the feed's own count for the
                # query, so a rejected or silently corrected query shows up as
                # 0 here with an HTTP 200 beside it — the case L1 cannot see
                # because nothing raised.
                self.run.observe_source(
                    SOURCE, http_status=self.last_status,
                    total_results=page.total_results,
                )

            for entry in page.entries:
                item = entry_to_item(entry)
                if item is not None:
                    items.append(item)

            page_no += 1
            offset += self.page_size
            if not page.entries or offset >= page.total_results:
                break
            if offset >= self.PAGING_LIMIT:
                self.run.error(
                    f"collect.arxiv: window {start}..{end} has {page.total_results} "
                    f"results, above the {self.PAGING_LIMIT} paging limit; truncated"
                )
                break
            if max_pages is not None and page_no >= max_pages:
                break

        return items, page_no


# --------------------------------------------------------------------------
# Parsing — kept as free functions so raw fixtures can be re-parsed in tests
# --------------------------------------------------------------------------


def parse_atom(xml: str) -> ArxivPage:
    root = ET.fromstring(xml)
    total = root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults") or "0"
    start = root.findtext("{http://a9.com/-/spec/opensearch/1.1/}startIndex") or "0"
    entries = [_entry_dict(e) for e in root.findall(f"{ATOM}entry")]
    return ArxivPage(
        entries=[e for e in entries if e], total_results=int(total), start=int(start)
    )


def _entry_dict(entry: ET.Element) -> dict:
    def text(tag: str) -> Optional[str]:
        return clean_text(entry.findtext(f"{ATOM}{tag}"))

    authors = []
    for a in entry.findall(f"{ATOM}author"):
        name = clean_text(a.findtext(f"{ATOM}name"))
        affil = clean_text(a.findtext(f"{ARXIV_NS}affiliation"))
        if name:
            authors.append({"name": name, "affiliation": affil})

    categories = [
        c.get("term")
        for c in entry.findall(f"{ATOM}category")
        if c.get("term")
    ]
    primary = entry.find(f"{ARXIV_NS}primary_category")
    if primary is not None and primary.get("term"):
        term = primary.get("term")
        categories = [term] + [c for c in categories if c != term]

    links = {}
    for link in entry.findall(f"{ATOM}link"):
        rel, href = link.get("rel"), link.get("href")
        title = link.get("title")
        if title == "pdf":
            links["pdf"] = href
        elif rel == "alternate":
            links["abs"] = href
        elif title == "doi":
            links["doi"] = href

    return {
        "id": text("id"),
        "title": text("title"),
        "summary": text("summary"),
        "published": text("published"),
        "updated": text("updated"),
        "authors": authors,
        "categories": categories,
        "links": links,
        "doi": clean_text(entry.findtext(f"{ARXIV_NS}doi")),
        "comment": clean_text(entry.findtext(f"{ARXIV_NS}comment")),
        "journal_ref": clean_text(entry.findtext(f"{ARXIV_NS}journal_ref")),
    }


_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def _to_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    m = _ISO.match(value)
    return date.fromisoformat(m.group(1)) if m else None


def entry_to_item(entry: dict) -> Optional[Item]:
    arxiv_id = normalize_arxiv_id(entry.get("id"))
    title = entry.get("title")
    if not arxiv_id or not title:
        return None

    published = _to_date(entry.get("published"))
    return Item(
        work_key=f"arxiv:{arxiv_id}",
        first_published=published,
        updated=_to_date(entry.get("updated")) or published,
        ids=Ids(
            arxiv=arxiv_id,
            # A journal DOI in the arXiv record wins; otherwise the DataCite DOI.
            doi=(entry.get("doi") or arxiv_doi(arxiv_id)).lower(),
        ),
        bibliography=Bibliography(
            title=title,
            authors=[
                Author(
                    name=a["name"],
                    institutions=(
                        [{"name": a["affiliation"]}] if a.get("affiliation") else []
                    ),
                )
                for a in entry.get("authors", [])
            ],
            publication_date=published,
            primary_location=PrimaryLocation(
                source_id=ARXIV_SOURCE_ID,
                source_name="arXiv",
                type="repository",
                version="submittedVersion",
                landing_page_url=entry.get("links", {}).get("abs")
                or f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=entry.get("links", {}).get("pdf")
                or f"https://arxiv.org/pdf/{arxiv_id}",
            ),
            abstract=entry.get("summary"),
            # Parsed since phase 0 and discarded until phase 0k. This is where
            # authors put "Code available at github.com/…".
            comment=entry.get("comment"),
            categories=[c for c in entry.get("categories", []) if c],
        ),
        provenance=Provenance(collected_at=_utcnow(), collectors=["arxiv"]),
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

