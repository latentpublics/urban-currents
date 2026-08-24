"""The JSON, the metadata and the structured data (Launch A).

Three properties this file exists to hold, each of which is cheap to break and
expensive to discover broken:

  1. **The API states no new fact.** Every field is already in `content/`; the
     endpoints are a reshaping. A field invented here would become the only
     source of something, and the first consumer would make it real.
  2. **The licence is three licences.** One notice over all of it would be a
     claim about summaries and abstracts that we cannot support, so the
     document says which is which and the API says so too.
  3. **The `noindex` switch still governs everything.** A canonical, an
     `og:url` and an `llms.txt` are all invitations to index; publishing them
     next to `Disallow: /` states two intentions in one directory.

No network, no keys.
"""

from __future__ import annotations

import json
import re
from datetime import date

import pytest

from pipeline import paths, store
from pipeline.models import (
    Bibliography,
    Headline,
    Issue,
    Item,
    PrimaryLocation,
    ScanMeta,
    SummaryEn,
)
from pipeline.outcome import NOT_PUBLISHED, Outcome, record
from pipeline.render import site as site_mod
from pipeline.render.api import build_api
from pipeline.render.site import (
    build_api_docs,
    build_archive,
    build_home,
    build_issue_pages,
    build_llms_txt,
    build_sitemap,
    ld_issue,
    load_issues,
    item_index,
)

DAY = date(2026, 8, 18)
QUIET = date(2026, 8, 17)
FILLED = date(2026, 8, 16)


def _item(key: str, title: str) -> Item:
    it = Item(
        work_key=key,
        first_published=DAY,
        bibliography=Bibliography(
            title=title,
            abstract="An abstract that belongs to somebody else.",
            primary_location=PrimaryLocation(
                source_name="arXiv", landing_page_url=f"https://arxiv.org/abs/{key[-7:]}"
            ),
        ),
    )
    it.ids.doi = "10.48550/arxiv.0000000"
    it.summary.en = SummaryEn(what="What it did.", why="Why it matters.")
    store.save_item(it)
    return it


def _issue(d, keys, *, backfilled=False, headline=True) -> Issue:
    issue = Issue(
        date=d,
        items=sorted(keys),
        headline=Headline(
            present=bool(headline and keys),
            work_key=keys[0] if (headline and keys) else None,
            line="A line about the day." if (headline and keys) else None,
        ),
        scan_meta=ScanMeta(
            items_published=len(keys), candidates_scanned=100, journals=96,
            arxiv_categories=7, unreadable_count=4,
        ),
        backfilled=backfilled,
    )
    store.save_issue(issue)
    return issue


@pytest.fixture
def built(repo):
    _item("arxiv:2608.50001", 'A paper with a "quoted" phrase in its title')
    _item("arxiv:2608.50002", "Another paper")
    _issue(DAY, ["arxiv:2608.50001", "arxiv:2608.50002"])
    _issue(QUIET, [])
    _issue(FILLED, ["arxiv:2608.50002"], backfilled=True)
    record(Outcome(date=FILLED, status=NOT_PUBLISHED, reasons=["collect.arxiv failed"],
                   published=0))
    build_api()
    build_api_docs()
    build_home()
    build_archive()
    build_issue_pages()
    return paths.ROOT / "site"


def _json(site, name):
    return json.loads((site / "api" / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# A1 — the JSON
# --------------------------------------------------------------------------


def test_the_three_addresses_exist(built):
    assert (built / "api" / "index.json").exists()
    assert (built / "api" / "latest.json").exists()
    assert (built / "api" / "issues" / f"{DAY}.json").exists()


def test_the_state_is_derived_not_read_from_the_stored_flag(built):
    """0Z's Z1, exported. `quiet_day` on disk is true on days that published
    nine papers; a field called `quiet` carrying that would ship the confusion
    the screen just stopped making."""
    day = _json(built, f"issues/{DAY}.json")
    quiet = _json(built, f"issues/{QUIET}.json")

    assert day["state"] == "published"
    assert quiet["state"] == "quiet"
    assert quiet["counts"]["published"] == 0


def test_both_facts_survive_for_a_withheld_and_backfilled_day(built):
    """0Y's Y1-2 reaches the API: one date, two facts, neither replacing the
    other."""
    filled = _json(built, f"issues/{FILLED}.json")

    assert filled["backfilled"] is True
    assert filled["withheld"] == "collect.arxiv failed"
    assert filled["counts"]["published"] == 1


def test_no_abstract_is_served(built):
    """Third-party expressive text; an API is a redistribution."""
    raw = (built / "api" / "issues" / f"{DAY}.json").read_text(encoding="utf-8")

    assert "belongs to somebody else" not in raw
    assert "abstract" not in json.loads(raw)["items"][0]


def test_no_person_is_served(built):
    """A per-day machine-readable list of people is a different artefact from a
    citation, and every item carries an identifier that reaches the real one."""
    item = _json(built, f"issues/{DAY}.json")["items"][0]

    for field in ("authors", "orcid", "institutions", "affiliations"):
        assert field not in item
    assert item["ids"]["doi"], "and this is what stands in for them"


def test_the_api_invents_nothing(built):
    """No issue number, no generated timestamp. The first would be a fact whose
    only source is this file; the second would make the build non-idempotent
    for nobody's benefit."""
    raw = (built / "api" / "index.json").read_text(encoding="utf-8")
    index = json.loads(raw)

    for invented in ("issue_number", "number", "generated", "generated_at", "timestamp"):
        assert invented not in index
    assert "latest_date" in index, "which is the question a timestamp gets asked for"


def test_building_twice_changes_nothing(built):
    """Idempotent, like the rest of the pipeline — and the reason there is no
    timestamp in it."""
    before = (built / "api" / "index.json").read_text(encoding="utf-8")
    build_api()
    assert (built / "api" / "index.json").read_text(encoding="utf-8") == before


def test_a_day_with_no_issue_is_in_the_catalogue(repo):
    """A gap would say nothing happened on a day the sources did not answer —
    the one claim the outcome model exists to avoid, reinvented by any consumer
    iterating over dates."""
    _item("arxiv:2608.50003", "A paper")
    _issue(DAY, ["arxiv:2608.50003"])
    record(Outcome(date=date(2026, 8, 15), status=NOT_PUBLISHED,
                   reasons=["the sources did not answer"], published=0))
    build_api()

    index = json.loads((paths.ROOT / "site" / "api" / "index.json").read_text(encoding="utf-8"))
    gap = [d for d in index["issues"] if d["date"] == "2026-08-15"]

    assert gap and gap[0]["state"] == "not_seen"
    assert gap[0]["url"] is None


def test_the_licence_is_three_licences(built):
    """One notice over all of it would claim something about the summaries and
    the bibliographic records that we have not established."""
    licence = _json(built, "index.json")["licence"]

    assert licence["selection_and_metadata"] == "CC BY 4.0"
    assert "no open licence" in licence["summaries"]
    assert "arXiv" in licence["bibliographic_records"]


# --------------------------------------------------------------------------
# A1-3 — the document
# --------------------------------------------------------------------------


def test_the_document_says_what_is_not_served_and_why(built):
    html = (built / "api.html").read_text(encoding="utf-8")

    assert "Abstracts" in html and "redistribution" in html
    assert "no open licence offered" in html.lower() or "no open licence" in html
    assert "CC BY 4.0" in html
    # arXiv's terms require the wording, and this page redistributes their data.
    assert "was not reviewed or approved by" in html


def test_the_document_explains_the_fields_whose_names_have_lied(built):
    """`headline.present` and `state` both had a name that disagreed with what
    they held. A consumer cannot see that from the JSON."""
    html = (built / "api.html").read_text(encoding="utf-8")

    assert "headline.present" in html
    assert "quiet_day" in html, "the stored flag is named, and distinguished"
    assert "backfilled" in html
    assert "withheld" in html


def test_the_document_is_reachable_from_the_pages(built):
    for name in ("index.html", "archive.html"):
        assert 'href="api.html"' in (built / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# A2 — the home page explains itself, unflatteringly
# --------------------------------------------------------------------------


def test_the_home_page_states_the_measured_precision_and_the_miss(built, monkeypatch):
    """A2. The number we would rather not print is the point of printing it."""
    monkeypatch.setattr(
        site_mod,
        "selection_quality",
        lambda: {
            "k": 10, "n_labels": 148, "days": 5, "target": 0.7,
            "sources": {
                "arxiv": {"label": "arXiv", "precision": 0.6, "n_labels": 73,
                          "days": 5, "depth_holding_target": 8},
                "journal": {"label": "journal", "precision": 0.66, "n_labels": 75,
                            "days": 5, "depth_holding_target": 8},
            },
        },
    )
    html = build_home().read_text(encoding="utf-8")

    assert "0.60" in html and "0.70" in html
    assert "miss it" in html, "and says so, rather than leaving it to arithmetic"
    assert "73 labels" in html and "75 labels" in html, "with the population"


def test_the_precision_sentence_says_precision_and_not_a_count_of_papers(built, monkeypatch):
    """★ The figure is a rate, and the sentence has to read as one.

    It said "0.60 were worth publishing", which cannot be true of papers —
    there is no such thing as 0.60 of a paper. The tempting repair is worse:
    "6 of the ten" reads more confident and states an integer for one day,
    while the number is the **mean of each labelled day's precision** and the
    days behind it are not alike. This is the paragraph where the number *is*
    the claim, so it says what was measured.
    """
    monkeypatch.setattr(
        site_mod,
        "selection_quality",
        lambda: {
            "k": 10, "n_labels": 148, "days": 5, "target": 0.7,
            "sources": {
                "arxiv": {"label": "arXiv", "precision": 0.6, "n_labels": 73,
                          "days": 5, "depth_holding_target": 8},
            },
        },
    )
    html = build_home().read_text(encoding="utf-8")
    body = html[html.index("How well that works") :][:400]

    assert "precision was 0.60 on the" in body
    assert "were worth publishing" not in body, "papers are counted, not rated"
    assert not re.search(r"\d+ of the ten", body), "a mean is not a count"
    # And the source is spelled the way its own organisation spells it, in the
    # one paragraph where getting a name wrong costs the most.
    assert "arXiv path" in body
    assert "arxiv path" not in body


def test_the_home_page_leaves_the_sentence_out_when_it_cannot_measure(built, monkeypatch):
    """The rule this project keeps restating: leave the line out rather than
    print a zero."""
    monkeypatch.setattr(site_mod, "selection_quality", lambda: None)
    html = build_home().read_text(encoding="utf-8")

    assert "miss it" not in html
    assert "How this is made" in html, "the rest of the section is still there"


def test_the_precision_figures_are_not_written_into_the_template():
    """Same rule `test_no_number_in_the_chrome_is_written_into_the_template`
    holds for the visible chrome. A figure typed into HTML stops being true
    without anything failing."""
    template = (
        paths.ROOT / "pipeline" / "render" / "templates" / "home.html.j2"
    )
    if not template.exists():  # running against an installed package
        import pipeline.render
        from pathlib import Path

        template = Path(pipeline.render.__file__).parent / "templates" / "home.html.j2"
    text = template.read_text(encoding="utf-8")

    for figure in ("0.60", "0.66", "0.600", "0.660"):
        assert figure not in text, f"{figure} is written into the template"


# --------------------------------------------------------------------------
# A3 — the metadata
# --------------------------------------------------------------------------


def test_every_page_has_a_derived_description(built):
    for name in ("index.html", "archive.html", f"issues/{DAY}.html", "api.html"):
        html = (built / name).read_text(encoding="utf-8")
        m = re.search(r'<meta name="description" content="([^"]*)"', html)
        assert m and m.group(1).strip(), f"{name} has no description"

    issue = (built / "issues" / f"{DAY}.html").read_text(encoding="utf-8")
    assert "A line about the day." in re.search(
        r'<meta name="description" content="([^"]*)"', issue
    ).group(1), "an issue describes itself with its own headline"


def test_a_quote_in_a_line_cannot_break_out_of_an_attribute(repo):
    """A description is built from a headline, and a headline is a sentence a
    model wrote. A double quote in it would close the attribute and spill the
    rest into the markup as tags."""
    _item("arxiv:2608.50004", "A paper")
    issue = _issue(DAY, ["arxiv:2608.50004"])
    issue.headline.line = 'Measuring "smart" cities, and what that costs'
    store.save_issue(issue)
    build_issue_pages()
    html = (paths.ROOT / "site" / "issues" / f"{DAY}.html").read_text(encoding="utf-8")

    m = re.search(r'<meta name="description" content="([^"]*)"', html)
    assert m and "&quot;smart&quot;" in m.group(1)
    # And the same string in the Open Graph copy, which is a second attribute.
    assert '<meta property="og:description" content="Measuring &quot;smart&quot;' in html


def test_the_canonical_is_absolute_and_never_root_relative(built):
    """A sub-path deploy: `/issues/…` would resolve against the organisation's
    domain root and 404."""
    for name in ("index.html", "archive.html", f"issues/{DAY}.html", "api.html"):
        html = (built / name).read_text(encoding="utf-8")
        m = re.search(r'<link rel="canonical" href="([^"]*)"', html)
        assert m, f"{name} has no canonical"
        assert m.group(1).startswith("https://"), m.group(1)
        assert "/urban-currents/" in m.group(1), m.group(1)


def test_there_is_no_image_tag(built):
    """An `og:image` pointing at a file that does not exist renders as a broken
    box instead of as plain text."""
    for name in ("index.html", f"issues/{DAY}.html"):
        html = (built / name).read_text(encoding="utf-8")
        assert "og:image" not in html
        assert 'twitter:card" content="summary"' in html, "the card that needs none"


def test_the_sitemap_holds_every_page(built):
    build_sitemap()
    xml = (built / "sitemap.xml").read_text(encoding="utf-8")

    for name in ("index.html", "archive.html", "api.html", f"issues/{DAY}.html"):
        assert f"/{name}<" in xml, f"{name} is not in the sitemap"


# --------------------------------------------------------------------------
# A4 — the structured data
# --------------------------------------------------------------------------


def _ld(html: str) -> dict:
    m = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)
    assert m, "no structured data"
    return json.loads(m.group(1))


def test_the_home_page_is_a_periodical_not_a_blog(built):
    node = _ld((built / "index.html").read_text(encoding="utf-8"))

    assert node["@type"] == "WebSite"
    assert node["mainEntity"]["@type"] == "Periodical"
    assert "issn" not in node["mainEntity"], "there isn't one"


def test_an_issue_mentions_the_papers_and_does_not_claim_them(built):
    """★ The over-claim this batch had to avoid: our summaries are not our
    scholarly articles. `mentions` says the issue points at a work; `hasPart`,
    `citation` and `isBasedOn` all say more than we are entitled to."""
    node = _ld((built / "issues" / f"{DAY}.html").read_text(encoding="utf-8"))

    assert node["@type"] == "PublicationIssue"
    assert node["isPartOf"]["@type"] == "Periodical"
    assert node["datePublished"] == str(DAY)
    assert len(node["mentions"]) == 2
    assert node["mentions"][0]["@type"] == "ScholarlyArticle"
    for forbidden in ("hasPart", "citation", "isBasedOn", "author"):
        assert forbidden not in node, f"{forbidden} claims more than we may"


def test_no_abstract_reaches_the_structured_data_either(built):
    html = (built / "issues" / f"{DAY}.html").read_text(encoding="utf-8")
    node = _ld(html)
    for paper in node["mentions"]:
        assert "abstract" not in paper
        assert set(paper) <= {"@type", "name", "url", "identifier"}


def test_the_api_page_is_a_dataset_whose_licence_is_a_page(built):
    """Three licences cannot be one `license` URL, so it points at the page
    that separates them."""
    node = _ld((built / "api.html").read_text(encoding="utf-8"))

    assert node["@type"] == "Dataset"
    assert node["license"].endswith("api.html")
    assert any(d["encodingFormat"] == "application/json" for d in node["distribution"])


def test_the_structured_data_parses_on_every_page(built):
    for name in ("index.html", f"issues/{DAY}.html", "api.html"):
        node = _ld((built / name).read_text(encoding="utf-8"))
        assert node["@context"] == "https://schema.org"
        assert node["@type"]


def test_an_issue_with_a_missing_item_still_renders(repo):
    """`ld_issue` walks `issue.items`; a key with no item on disk must not take
    the page down with it."""
    _item("arxiv:2608.50005", "A paper")
    issue = _issue(DAY, ["arxiv:2608.50005"])
    issue.items = issue.items + ["arxiv:2608.99999"]
    store.save_issue(issue)

    node = ld_issue(load_issues()[0], item_index(), "https://example.org", "x")
    assert len(node["mentions"]) == 1


# --------------------------------------------------------------------------
# The publish switch still governs all of it
# --------------------------------------------------------------------------


def test_nothing_new_escapes_the_noindex(built):
    """G5 has not happened. Every page this batch added carries the same
    `noindex` as the ones that were already here."""
    for name in ("index.html", "archive.html", "api.html", f"issues/{DAY}.html"):
        html = (built / name).read_text(encoding="utf-8")
        assert 'content="noindex, nofollow"' in html, name

    from pipeline.render.site import build_robots

    assert "Disallow: /" in build_robots().read_text(encoding="utf-8")


def test_llms_txt_waits_for_the_publish_switch(built, monkeypatch):
    """It is discovered by convention at a fixed path, so writing it *is*
    advertising — the same argument that keeps the sitemap line out of
    `robots.txt` until then."""
    assert build_llms_txt() is None
    assert not (paths.ROOT / "site" / "llms.txt").exists()

    monkeypatch.setattr(site_mod, "is_published", lambda: True)
    written = build_llms_txt()

    assert written is not None
    text = written.read_text(encoding="utf-8")
    assert "api/index.json" in text
    assert "CC BY 4.0" in text and "no open licence" in text
