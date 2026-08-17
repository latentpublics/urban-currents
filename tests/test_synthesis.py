"""The daily synthesis layer (phase 0i, V3).

The layer's whole claim is that it says only what was measured. These tests are
about the ways it could stop being true: a baseline that does not exist, a
connection that runs backwards in time, a paragraph written from nothing.
"""

from __future__ import annotations

from datetime import date, timedelta

from pipeline import store
from pipeline.synthesis import (
    MIN_BASELINE_DAYS,
    PARAGRAPH_MIN_MATERIAL,
    affiliations,
    build_facts,
    clean_title,
    deviations,
    material_for_paragraph,
    render_facts,
    write_paragraph,
)
from tests.test_selection_paths import _whitelist_source_id, arxiv_item, journal_item

DAY = date(2026, 8, 11)


def _tagged(item, labels):
    from pipeline.models import EntityRef

    item.entities.methods = [EntityRef(id=f"method:{v}", label=v) for v in labels]
    return item


def _empty_facts(**over):
    facts = {
        "date": str(DAY),
        "composition": {"published": 0, "journal": 0, "arxiv": 0, "unreadable": 0},
        "deviations": {"found": [], "status": "OK", "baseline_days": 30},
        "anchors": [],
        "clusters": [],
        "affiliations": {"today": [], "in_window": [], "window_days": 30},
        "repeat_authors": [],
        "first_internal_citation": None,
    }
    facts.update(over)
    return facts


# --------------------------------------------------------------------------
# Deviations
# --------------------------------------------------------------------------


def test_a_short_archive_reports_no_baseline_rather_than_a_zero_one(repo):
    """An empty baseline is not a low baseline.

    The first version compared today against an archive of one day and reported
    four spikes, every one of them "3 today against 0.0 per day". True
    arithmetic, no information, and it would have shipped in the issue.
    """
    wl = _whitelist_source_id()
    items = [_tagged(journal_item(i, wl), ["street view"]) for i in range(4)]
    out = deviations(DAY, items)

    assert out["status"] == "NO_BASELINE"
    assert out["found"] == []
    assert str(MIN_BASELINE_DAYS) in out["note"]


def test_the_vocabulary_bottleneck_is_still_measured_when_the_baseline_is_short(repo):
    # Nothing here reaches the issue; it exists so the size of the prize for the
    # pending vocabulary curation is a number rather than an intuition.
    wl = _whitelist_source_id()
    items = [_tagged(journal_item(i, wl), ["street view"]) for i in range(4)]
    out = deviations(DAY, items)
    assert "would_find_if_baseline_were_long_enough" in out
    assert out["distinct_tags_today"] == 1


# --------------------------------------------------------------------------
# The paragraph
# --------------------------------------------------------------------------


def test_no_paragraph_without_a_measured_link(repo):
    """Recurrence of a name is not a connection.

    2026-08-05 had three facts — two repeat authors and one recurring
    institution — and the paragraph written from them was a roster read aloud.
    """
    facts = _empty_facts(
        repeat_authors=[{"name": "A", "papers_today": 2}, {"name": "B", "papers_today": 2}],
        affiliations={"today": [], "in_window": [{"name": "X", "papers": 4}], "window_days": 30},
    )
    out = write_paragraph(facts)

    assert out["omitted"] is True
    assert out["links"] == 0
    assert out["text"] is None


def test_no_paragraph_when_there_is_almost_nothing(repo):
    out = write_paragraph(_empty_facts())
    assert out["omitted"] is True
    assert out["material"] < PARAGRAPH_MIN_MATERIAL


def test_the_model_may_refuse_and_the_refusal_is_honoured(repo):
    from pipeline.llm import LLMClient, LLMResponse

    facts = _empty_facts(
        clusters=[
            {"scope": "today", "work_keys": ["a", "b"], "titles": ["A", "B"],
             "shared": 4, "shared_titles": ["R"], "partner_date": str(DAY)}
        ],
        anchors=[
            {"openalex_id": "openalex:W1", "title": "T", "year": "2010", "authors": ["X"],
             "citing_today": 2, "citing_work_keys": [], "days_since_last_cited": 5,
             "first_in_window": False}
        ],
        repeat_authors=[{"name": "A", "papers_today": 2}],
    )
    client = LLMClient(
        task="synthesis",
        caller=lambda system, user: LLMResponse(text="NOTHING TO SAY"),
    )
    out = write_paragraph(facts, client=client)
    assert out["omitted"] is True
    assert "too thin" in out["reason"]


def test_the_facts_block_never_states_the_issue_size(repo):
    """It is printed directly above the paragraph; restating it costs a sentence."""
    facts = _empty_facts(
        composition={"published": 24, "journal": 12, "arxiv": 12, "unreadable": 3},
        repeat_authors=[{"name": "A", "papers_today": 2}],
    )
    block = render_facts(facts)
    assert "24" not in block


def test_the_facts_block_carries_the_shared_references_by_name(repo):
    facts = _empty_facts(
        clusters=[{
            "scope": "archive", "work_keys": ["a", "b"], "titles": ["A", "B"],
            "shared": 5, "shared_titles": ["Built Environment Correlates of Walking"],
            "partner_date": "2026-08-07",
        }],
    )
    block = render_facts(facts)
    # Naming the shared works is what makes an invented theme impossible: there
    # is nowhere in the output for one to go.
    assert "Built Environment Correlates of Walking" in block
    assert "5 references" in block


# --------------------------------------------------------------------------
# Clusters
# --------------------------------------------------------------------------


def test_an_issue_never_links_to_a_paper_published_after_it(repo):
    """The coupling window is symmetric; a daily issue is not.

    Before this, the 2026-08-07 issue reported sharing references with a paper
    published on 2026-08-11 — true about the archive, impossible for an issue
    that went out four days earlier.
    """
    from pipeline.synthesis import clusters

    wl = _whitelist_source_id()
    today = [journal_item(1, wl)]
    for it in today:
        it.first_published = DAY
        store.save_item(it, today=DAY)

    future = journal_item(2, wl)
    future.first_published = DAY + timedelta(days=4)
    store.save_item(future, today=future.first_published)

    for c in clusters(DAY, today):
        if c["partner_date"]:
            assert c["partner_date"] <= str(DAY)


# --------------------------------------------------------------------------
# Institutions
# --------------------------------------------------------------------------


def test_institutions_are_counted_never_ranked(repo):
    from pipeline.models import Author, Institution

    wl = _whitelist_source_id()
    items = []
    for i in range(3):
        it = journal_item(i, wl)
        it.bibliography.authors = [
            Author(name=f"A{i}", institutions=[Institution(name="TU Delft")])
        ]
        items.append(it)

    out = affiliations(DAY, items)
    # A count and a name. No score, no rank, no "top" anything anywhere in it.
    assert out["today"] == [{"name": "TU Delft", "papers": 3}]
    assert not any("rank" in k or "score" in k for k in out)


# --------------------------------------------------------------------------
# Titles
# --------------------------------------------------------------------------


def test_publisher_markup_is_stripped_before_a_title_is_quoted():
    # The model faithfully reproduced "<scp>NOCTURNAL INFORMALITY</scp>" in its
    # prose the first time this ran.
    assert clean_title("<scp>NOCTURNAL INFORMALITY</scp> : Rethinking") == (
        "NOCTURNAL INFORMALITY : Rethinking"
    )
    assert clean_title("<i>Exiles in New York City</i> , by Philip T. Yanos") == (
        "Exiles in New York City , by Philip T. Yanos"
    )


def test_build_facts_runs_on_an_empty_day(repo):
    facts = build_facts(DAY, [], 0)
    assert facts["composition"]["published"] == 0
    assert material_for_paragraph(facts) == 0
