"""Abstract enrichment — Crossref and Springer Nature (PRD §5.1, phase 0c).

These are **enrichers, not collectors**. They never create an Item. They take
Items that already exist and have no abstract, and try to fill that one field.

Why they exist: publishers withdrew abstracts from OpenAlex — Springer Nature in
2022, Elsevier in November 2024, about 1.1 million closed-access records. Measured
on the 2026-08-11 issue, half the journal path is permanently dark and a quarter
is recoverable from the publisher's own free API. So "OpenAlex has no abstract"
stopped meaning "there is no abstract" and started meaning "ask somewhere else".

The order is **openalex → crossref → springer_api**, cheapest and most permissive
first, and it stops at the first hit — `provenance.abstract_source` records which
one paid off. What survives all three is genuinely unreadable, and that is a
measurement rather than a failure (see `stage_issue`'s `unreadable` list).

Not attempted, and why — measured in 0c and recorded in the working notes:

- **Elsevier**: abstracts need an institutional entitlement, the terms forbid
  public display, and they forbid processing content through public AI tools.
  Three independent blocks, so no key would help.
- **Taylor & Francis**: no API exists.
- **IEEE**: terms forbid machine-harvestable redistribution and LLM processing.
- **Semantic Scholar**: keys are not issued to free-domain addresses, the
  unauthenticated pool returns 429, and the licence may be non-commercial.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

import httpx

from ..config import cfg, contact_email, secret
from ..metrics import Run
from ..models import Item
from .base import strip_markup

CROSSREF_API = "https://api.crossref.org/works/"
SPRINGER_API = "https://api.springernature.com/meta/v2/json"

# DOI prefixes Springer Nature owns. Used to decide whether asking is worth a
# request at all — the API answers for its own corpus only.
SPRINGER_PREFIXES = ("10.1007", "10.1186", "10.1038", "10.1057", "10.1140", "10.1245")

# Publisher name by DOI prefix, for the unreadable-by-publisher tally. Only the
# prefixes the whitelist actually contains; anything else is reported by its
# bare prefix rather than guessed at.
PUBLISHER_BY_PREFIX = {
    "10.1016": "Elsevier",
    "10.1080": "Taylor & Francis",
    # Routledge book chapters, which reach the whitelist through their series.
    "10.4324": "Taylor & Francis (Routledge)",
    "10.1007": "Springer",
    "10.1186": "Springer (BMC)",
    "10.1038": "Springer Nature",
    "10.1177": "Sage",
    "10.1111": "Wiley",
    "10.1002": "Wiley",
    "10.3390": "MDPI",
    "10.1061": "ASCE",
    "10.1109": "IEEE",
    "10.3389": "Frontiers",
    "10.5194": "Copernicus",
    "10.1017": "Cambridge",
    "10.1093": "Oxford",
}


# Which route can reach a publisher's abstracts, keyed by the publisher name
# OpenAlex reports. Percentages are Crossref deposit rates for current articles,
# measured 2026-08-13 (0c, source survey).
#
# This is a **routing** field, not an exclusion rule. A journal whose abstracts
# we cannot get stays on the whitelist and its articles still publish, in
# `Also published today`. Reviewing the whitelist now has two axes — is this
# urban research, and can we read it — and they are answered separately.
_ABSTRACT_SOURCE_BY_PUBLISHER = {
    "elsevier": "none",             # 4.84%, and the terms forbid the rest
    "taylor & francis": "none",     # 0.065%, and no API exists
    "informa": "none",              # Taylor & Francis' parent
    "ieee": "none",                 # 1.07%, terms forbid LLM processing
    "american society of civil engineers": "none",   # 3.09%
    "springer": "springer_api",     # 24.6% via Crossref, full corpus via its own API
    "nature": "springer_api",
    "biomed central": "springer_api",
    "wiley": "crossref",            # 66.5%
    "sage": "crossref",             # 76.5%
    "frontiers": "crossref",        # 90.6%
    "multidisciplinary digital publishing institute": "crossref",  # MDPI, 98.2%
    "mdpi": "crossref",
}


def abstract_source_for_publisher(publisher: Optional[str]) -> str:
    """Best route to this publisher's abstracts, or `openalex` if it already has them.

    `openalex` is the default because the publishers that withdrew are the
    exception; assuming a withdrawal we have not measured would understate
    coverage and send needless requests.
    """
    if not publisher:
        return "openalex"
    name = publisher.lower()
    for needle, source in _ABSTRACT_SOURCE_BY_PUBLISHER.items():
        if needle in name:
            return source
    return "openalex"


def doi_prefix(item: Item) -> Optional[str]:
    doi = item.ids.doi
    return doi.split("/", 1)[0] if doi and "/" in doi else None


def publisher_of(item: Item) -> str:
    prefix = doi_prefix(item)
    if not prefix:
        return "unknown"
    return PUBLISHER_BY_PREFIX.get(prefix, prefix)


def needs_abstract(item: Item) -> bool:
    return not (item.bibliography.abstract or "").strip()


def _jats_to_text(value: Optional[str]) -> Optional[str]:
    """Crossref returns abstracts as JATS XML snippets.

    `<jats:p>` and friends are removed, and so is the `Abstract` heading many
    publishers deposit as the first element — it is markup furniture, not the
    abstract.
    """
    text = strip_markup(value)
    if not text:
        return None
    for lead in ("Abstract ", "ABSTRACT ", "Summary ", "Abstract: "):
        if text.startswith(lead):
            text = text[len(lead) :]
            break
    return text.strip() or None


class AbstractEnricher:
    """Shared throttling, retry and bookkeeping for the two abstract sources."""

    def __init__(self, run: Run, client: Optional[httpx.Client] = None):
        self.run = run
        self._client = client
        self._last_request = 0.0
        self.interval = 1.0 / float(cfg("crossref.requests_per_second", 5))

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={"User-Agent": (
                    f"UrbanCurrents/0.1 (+https://github.com/youngjour/urban-currents; "
                    f"mailto:{contact_email()})"
                )},
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_request
        if gap < self.interval:
            time.sleep(self.interval - gap)
        self._last_request = time.monotonic()

    # -- Crossref --------------------------------------------------------

    def from_crossref(self, item: Item) -> Optional[str]:
        """Crossref's polite pool. Free, no key, `mailto` in the User-Agent."""
        doi = item.ids.doi
        if not doi:
            return None
        self._throttle()
        try:
            r = self._http().get(
                f"{CROSSREF_API}{doi}", params={"mailto": contact_email()}
            )
        except Exception as e:  # noqa: BLE001
            self.run.error(f"crossref {doi}: {type(e).__name__}: {e}")
            return None
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            self.run.error(f"crossref {doi}: HTTP {r.status_code}")
            return None
        try:
            message = r.json().get("message") or {}
        except Exception:  # noqa: BLE001
            return None
        return _jats_to_text(message.get("abstract"))

    # -- Springer Nature -------------------------------------------------

    def from_springer(self, item: Item) -> Optional[str]:
        """Springer's own Metadata API, which still carries what it withdrew.

        Returns None when there is no key: the caller records SKIPPED rather
        than failing, so a run without the key produces the same content as one
        with it, minus the recovered abstracts.
        """
        key = secret("SPRINGER_API_KEY")
        if not key:
            return None
        doi = item.ids.doi
        if not doi:
            return None
        self._throttle()
        try:
            r = self._http().get(
                SPRINGER_API, params={"q": f"doi:{doi}", "api_key": key, "p": 1}
            )
        except Exception as e:  # noqa: BLE001
            self.run.error(f"springer {doi}: {type(e).__name__}: {e}")
            return None
        if r.status_code in (429, 403):
            # The free tier's monthly quota is undocumented, so the only way to
            # learn it is to hit it. Stop the pass rather than burn retries.
            self.run.error(f"springer: quota or rate limit reached (HTTP {r.status_code})")
            raise SpringerQuotaExceeded(f"HTTP {r.status_code}")
        if r.status_code != 200:
            self.run.error(f"springer {doi}: HTTP {r.status_code}")
            return None
        try:
            records = r.json().get("records") or []
        except Exception:  # noqa: BLE001
            return None
        for rec in records:
            text = strip_markup(rec.get("abstract"))
            if text:
                return text
        return None


class SpringerQuotaExceeded(RuntimeError):
    """Free-tier quota is undocumented; this is how it announces itself."""


def enrich_abstracts(
    items: Iterable[Item],
    run: Run,
    enricher: Optional[AbstractEnricher] = None,
    sources: tuple[str, ...] = ("crossref", "springer"),
) -> dict[str, int]:
    """Fill missing abstracts in place. Returns counts by source.

    Items that already have one are marked `openalex` and left alone — nothing
    is re-fetched, because the abstract we have is the abstract we publish.
    """
    enricher = enricher or AbstractEnricher(run)
    counts = {"openalex": 0, "crossref": 0, "springer_api": 0, "none": 0, "attempted": 0}
    springer_open = "springer" in sources

    for item in items:
        if not needs_abstract(item):
            # Only claim OpenAlex for an abstract that arrived unattributed;
            # a re-run must not relabel one this pass already recovered.
            if item.provenance.abstract_source == "none":
                item.provenance.abstract_source = "openalex"
            counts["openalex"] += 1
            continue

        counts["attempted"] += 1
        text = None
        if "crossref" in sources:
            text = enricher.from_crossref(item)
            if text:
                item.bibliography.abstract = text
                item.provenance.abstract_source = "crossref"
                counts["crossref"] += 1
                continue

        if springer_open and (doi_prefix(item) in SPRINGER_PREFIXES):
            try:
                text = enricher.from_springer(item)
            except SpringerQuotaExceeded:
                # Keep what this pass already recovered and stop asking.
                springer_open = False
                text = None
            if text:
                item.bibliography.abstract = text
                item.provenance.abstract_source = "springer_api"
                counts["springer_api"] += 1
                continue

        item.provenance.abstract_source = "none"
        counts["none"] += 1

    return counts
