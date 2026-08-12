"""arXiv collector (PRD §5.1).

The arXiv API asks for at least 3 seconds between requests and a contact address
in the User-Agent; both are honoured here, along with three backoff retries.

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
from typing import Iterator, Optional
from xml.etree import ElementTree as ET

import httpx

from ..config import cfg, contact_email
from ..metrics import Run
from ..models import Author, Bibliography, Ids, Item, PrimaryLocation, Provenance
from .base import ARXIV_SOURCE_ID, arxiv_doi, clean_text, normalize_arxiv_id

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

USER_AGENT_TEMPLATE = "urban-currents/0.2 (Phase 0 research scan; mailto:{email})"


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
        self.api_url = cfg("arxiv.api_url", "http://export.arxiv.org/api/query")
        self.categories = list(cfg("arxiv.categories", []) or [])
        self._client = client
        self._last_request = 0.0

    # -- HTTP ------------------------------------------------------------

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={"User-Agent": USER_AGENT_TEMPLATE.format(email=contact_email())},
                timeout=60.0,
                follow_redirects=True,
            )
        return self._client

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_request = time.monotonic()

    # A 429 from arXiv means "you have been asking too fast for a while"; the
    # ordinary 3/6/12s backoff is far too short and just earns another 429.
    RATE_LIMIT_COOLDOWN_S = 90.0

    def _fetch(self, params: dict) -> str:
        last: Optional[Exception] = None
        attempts = self.max_retries
        attempt = 0
        while attempt < attempts:
            self._throttle()
            try:
                r = self._http().get(self.api_url, params=params)
                if r.status_code == 429:
                    retry_after = float(r.headers.get("Retry-After") or 0)
                    wait = max(self.RATE_LIMIT_COOLDOWN_S, retry_after)
                    last = RuntimeError("429 rate limited")
                    # Rate limiting is recoverable and worth more patience than a
                    # transient error, so it does not consume a normal attempt.
                    attempts = max(attempts, attempt + 3)
                    # Slow down permanently for the rest of the run. Returning to
                    # the old cadence after a 429 just earns the next one.
                    self.interval = min(self.interval * 1.5, 12.0)
                    time.sleep(wait)
                    attempt += 1
                    continue
                r.raise_for_status()
                return r.text
            except Exception as e:  # noqa: BLE001
                last = e
                # arXiv returns empty bodies or 5xx under load; back off.
                time.sleep(self.interval * (2**attempt))
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

        for w_start, w_end in self._windows(start, d):
            try:
                got, pages = self._collect_window(
                    w_start, w_end, None if max_pages is None else max_pages - pages_used
                )
            except Exception as e:  # noqa: BLE001
                # One bad window must not discard the rest of a 90-day backfill.
                # The gap is recorded so the report can say which days are thin.
                self.run.error(
                    f"collect.arxiv: window {w_start}..{w_end} failed: "
                    f"{type(e).__name__}: {e}"
                )
                continue
            pages_used += pages
            for item in got:
                items.setdefault(item.work_key, item)
            if max_pages is not None and pages_used >= max_pages:
                break

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
            categories=[c for c in entry.get("categories", []) if c],
        ),
        provenance=Provenance(collected_at=_utcnow(), collectors=["arxiv"]),
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def daterange(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
