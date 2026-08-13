"""Abstract enrichment — Crossref and Springer Nature (phase 0c, P1/P2).

No test touches the network. The HTTP client is injected, exactly as the arXiv
collector's is, so the parsing and the ordering are what get exercised.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from pipeline.collectors.abstracts import (
    AbstractEnricher,
    SpringerQuotaExceeded,
    doi_prefix,
    enrich_abstracts,
    publisher_of,
)
from pipeline.collectors.base import strip_markup
from pipeline.metrics import Run
from pipeline.models import Bibliography, Ids, Item

DAY = date(2026, 8, 11)


def _item(doi: str, abstract: str | None = None) -> Item:
    return Item(
        work_key=f"doi:{doi}",
        first_published=DAY,
        ids=Ids(doi=doi),
        bibliography=Bibliography(title="A Paper", abstract=abstract),
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _enricher(repo, handler) -> AbstractEnricher:
    e = AbstractEnricher(Run.for_date(DAY), client=_client(handler))
    e.interval = 0.0  # the throttle is real; waiting for it in tests is not
    return e


# --------------------------------------------------------------------------
# Markup (P4-1) — the same stripper serves titles and abstracts
# --------------------------------------------------------------------------


def test_jats_tags_are_stripped_but_their_words_are_kept():
    """From the Q1b label data: a title that reached the labeller as markup."""
    raw = "<scp>DIFFERENTIATED INFRASTRUCTURAL CITIZENSHIP</scp> : Claims‐Making"
    assert strip_markup(raw) == "DIFFERENTIATED INFRASTRUCTURAL CITIZENSHIP: Claims‐Making"


def test_nested_and_self_closing_markup_is_removed():
    assert strip_markup("H<sub>2</sub>O and <i>in situ</i><br/> data") == "H2O and in situ data"
    assert strip_markup("<jats:p>Text</jats:p>") == "Text"


def test_an_unrecognised_entity_is_left_alone_rather_than_guessed():
    """A literal ampersand in a title is likelier than a typo'd entity."""
    assert strip_markup("Cities &amp; Regions") == "Cities & Regions"
    assert strip_markup("Cities & Regions") == "Cities & Regions"


def test_markup_stripping_is_idempotent():
    once = strip_markup("<scp>A</scp> : B")
    assert strip_markup(once) == once


# --------------------------------------------------------------------------
# Crossref
# --------------------------------------------------------------------------


def test_crossref_abstract_is_unwrapped_from_jats(repo):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "10.1177/x" in str(request.url)
        return httpx.Response(200, json={"message": {
            "abstract": "<jats:p>Abstract We study <jats:italic>cities</jats:italic>.</jats:p>"
        }})

    item = _item("10.1177/x")
    counts = enrich_abstracts([item], Run.for_date(DAY), _enricher(repo, handler))

    assert item.bibliography.abstract == "We study cities."
    assert item.provenance.abstract_source == "crossref"
    assert counts["crossref"] == 1


def test_a_crossref_miss_is_not_an_error(repo):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    item = _item("10.1016/x")
    counts = enrich_abstracts([item], Run.for_date(DAY), _enricher(repo, handler))

    assert item.bibliography.abstract is None
    assert item.provenance.abstract_source == "none"
    assert counts["none"] == 1


def test_an_item_that_already_has_an_abstract_is_never_fetched(repo):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    item = _item("10.1177/x", abstract="Already here.")
    counts = enrich_abstracts([item], Run.for_date(DAY), _enricher(repo, handler))

    assert item.provenance.abstract_source == "openalex"
    assert counts["attempted"] == 0


# --------------------------------------------------------------------------
# Springer
# --------------------------------------------------------------------------


def test_springer_is_asked_only_for_its_own_dois(repo, monkeypatch):
    monkeypatch.setenv("SPRINGER_API_KEY", "test-key")
    from pipeline import config

    config._ENV_LOADED = True
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "crossref" in str(request.url):
            return httpx.Response(404)
        asked.append(str(request.url))
        return httpx.Response(200, json={"records": [{"abstract": "<p>Springer text.</p>"}]})

    springer = _item("10.1007/x")   # Springer prefix
    elsevier = _item("10.1016/y")   # not Springer: asking would be a wasted call
    counts = enrich_abstracts(
        [springer, elsevier], Run.for_date(DAY), _enricher(repo, handler)
    )

    assert springer.bibliography.abstract == "Springer text."
    assert springer.provenance.abstract_source == "springer_api"
    assert elsevier.provenance.abstract_source == "none"
    assert len(asked) == 1, "only the Springer DOI is worth a request"
    assert counts["springer_api"] == 1
    config._ENV_LOADED = False


def test_without_a_key_springer_is_skipped_not_failed(repo, monkeypatch):
    monkeypatch.delenv("SPRINGER_API_KEY", raising=False)
    from pipeline import config

    config._ENV_LOADED = True

    def handler(request: httpx.Request) -> httpx.Response:
        if "springernature" in str(request.url):
            raise AssertionError("must not call Springer without a key")
        return httpx.Response(404)

    item = _item("10.1007/x")
    counts = enrich_abstracts([item], Run.for_date(DAY), _enricher(repo, handler))

    assert item.provenance.abstract_source == "none"
    assert counts["springer_api"] == 0
    config._ENV_LOADED = False


def test_a_springer_quota_stops_the_pass_and_keeps_what_it_recovered(repo, monkeypatch):
    """The free tier's monthly quota is undocumented, so hitting it is the only
    way to learn it. What was already recovered must survive."""
    monkeypatch.setenv("SPRINGER_API_KEY", "test-key")
    from pipeline import config

    config._ENV_LOADED = True
    calls = {"springer": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "crossref" in str(request.url):
            return httpx.Response(404)
        calls["springer"] += 1
        if calls["springer"] == 1:
            return httpx.Response(200, json={"records": [{"abstract": "First."}]})
        return httpx.Response(429)

    first, second, third = _item("10.1007/a"), _item("10.1007/b"), _item("10.1007/c")
    counts = enrich_abstracts(
        [first, second, third], Run.for_date(DAY), _enricher(repo, handler)
    )

    assert first.bibliography.abstract == "First."
    assert second.provenance.abstract_source == "none"
    assert third.provenance.abstract_source == "none"
    assert calls["springer"] == 2, "asking again after a 429 wastes the quota"
    assert counts["springer_api"] == 1
    config._ENV_LOADED = False


def test_crossref_is_tried_before_springer(repo, monkeypatch):
    """Cheapest and most permissive source first; the loser is never called."""
    monkeypatch.setenv("SPRINGER_API_KEY", "test-key")
    from pipeline import config

    config._ENV_LOADED = True

    def handler(request: httpx.Request) -> httpx.Response:
        if "springernature" in str(request.url):
            raise AssertionError("Crossref answered; Springer must not be called")
        return httpx.Response(200, json={"message": {"abstract": "<jats:p>From Crossref.</jats:p>"}})

    item = _item("10.1007/x")
    enrich_abstracts([item], Run.for_date(DAY), _enricher(repo, handler))

    assert item.provenance.abstract_source == "crossref"
    config._ENV_LOADED = False


# --------------------------------------------------------------------------
# Publisher attribution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doi,publisher",
    [
        ("10.1016/j.cities.2026.1", "Elsevier"),
        ("10.1080/23748834.2026.1", "Taylor & Francis"),
        ("10.1007/s44212-026-1", "Springer"),
        ("10.9999/unknown.1", "10.9999"),
    ],
)
def test_publisher_is_named_from_the_doi_prefix(doi, publisher):
    assert publisher_of(_item(doi)) == publisher
    assert doi_prefix(_item(doi)) == doi.split("/")[0]
