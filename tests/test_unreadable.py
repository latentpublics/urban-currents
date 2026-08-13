"""`Also published today` — items no source could give an abstract for (P3).

The rule the tests defend: these items publish, they never touch an LLM, and
nothing about them is inferred beyond what the bibliography and their own title
already say.
"""

from __future__ import annotations

from datetime import date

from pipeline import run_stages, store
from pipeline.metrics import Run
from pipeline.models import (
    Author,
    Bibliography,
    Ids,
    Institution,
    Item,
    PrimaryLocation,
    TopicRef,
)
from pipeline.render.preview import build_unreadable_row, render_issue
from pipeline.stages import write_stage

DAY = date(2026, 8, 11)


def _whitelist_source_id() -> str:
    from pipeline.config import journals_vocab

    for s in journals_vocab().get("sources") or []:
        if s.get("id") and s.get("include", True):
            return s["id"]
    raise AssertionError("no included whitelist journal in the test config")


def _journal_item(n: int, source_id: str, abstract: str | None) -> Item:
    return Item(
        work_key=f"doi:10.1016/j.test.{n}",
        first_published=DAY,
        ids=Ids(doi=f"10.1016/j.test.{n}"),
        bibliography=Bibliography(
            title=f"Urban Paper {n}",
            abstract=abstract,
            authors=[
                Author(name="Ada Lovelace", institutions=[Institution(name="Test University")]),
                Author(name="Alan Turing"),
            ],
            primary_location=PrimaryLocation(
                source_id=source_id,
                source_name="Cities",
                landing_page_url=f"https://example.org/{n}",
            ),
        ),
    )


def _select(repo, items: list[Item]) -> tuple[list[Item], list[Item]]:
    run = Run.for_date(DAY)
    write_stage(run, "classify", items)
    selected = run_stages.stage_select(run)
    from pipeline.stages import read_stage

    return selected, read_stage(run, run_stages.UNREADABLE_STAGE)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_an_item_without_an_abstract_leaves_the_slot_competition(repo):
    """Not dropped and not ranked last — a separate set entirely. Ranking it
    last was the earlier half-measure; two mechanisms for one decision is one
    too many."""
    wl = _whitelist_source_id()
    readable = [_journal_item(i, wl, "An abstract.") for i in range(3)]
    dark = [_journal_item(100 + i, wl, None) for i in range(2)]

    selected, unreadable = _select(repo, readable + dark)

    assert {it.work_key for it in unreadable} == {it.work_key for it in dark}
    assert not ({it.work_key for it in dark} & {it.work_key for it in selected})
    assert {it.work_key for it in readable} <= {it.work_key for it in selected}


def test_an_empty_abstract_counts_as_no_abstract(repo):
    wl = _whitelist_source_id()
    _, unreadable = _select(repo, [_journal_item(1, wl, "   ")])
    assert len(unreadable) == 1


# --------------------------------------------------------------------------
# The issue
# --------------------------------------------------------------------------


def test_the_issue_lists_them_apart_from_its_items(repo):
    wl = _whitelist_source_id()
    run = Run.for_date(DAY)
    readable = [_journal_item(i, wl, "An abstract.") for i in range(2)]
    dark = [_journal_item(100 + i, wl, None) for i in range(3)]
    write_stage(run, "classify", readable + dark)
    run_stages.stage_select(run)
    issue = run_stages.stage_issue(run, DAY)

    assert sorted(issue.unreadable) == sorted(it.work_key for it in dark)
    assert not (set(issue.unreadable) & set(issue.items))
    assert issue.scan_meta.unreadable_count == 3
    assert issue.scan_meta.unreadable_by_publisher == {"Elsevier": 3}

    # Stored as ordinary Items, with no summary, so that the day an abstract
    # turns up the same record is promoted rather than published twice.
    for wk in issue.unreadable:
        stored = store.load_item(wk)
        assert stored is not None
        assert not (stored.summary.en and stored.summary.en.what)


def test_an_unreadable_item_is_not_marked_as_already_published(repo):
    """`published_index` reads `items` only. That is what lets the same record
    become a real card the day its abstract appears."""
    wl = _whitelist_source_id()
    run = Run.for_date(DAY)
    write_stage(run, "classify", [_journal_item(1, wl, None)])
    run_stages.stage_select(run)
    run_stages.stage_issue(run, DAY)

    assert store.published_index() == {}


# --------------------------------------------------------------------------
# The rendered section
# --------------------------------------------------------------------------


def test_the_row_states_facts_and_infers_nothing(repo):
    wl = _whitelist_source_id()
    item = _journal_item(1, wl, None)
    item.bibliography.title = "Clustering Urban Mobility with Random Forest Models"
    item.bibliography.authors.append(Author(name="Grace Hopper"))
    item.bibliography.authors.append(Author(name="Katherine Johnson"))
    item.entities.topics = [
        TopicRef(id="openalex:T1", label="Urban Studies", score=0.9),
        TopicRef(id="openalex:T2", label="Transport", score=0.8),
    ]

    row = build_unreadable_row(item)

    assert row["authors"] == "Ada Lovelace, Alan Turing, Grace Hopper, et al."
    assert row["affiliation"] == "Test University"
    assert row["journal"] == "Cities"
    assert row["topics"] == ["Urban Studies", "Transport"]
    # Vocabulary terms matched against the title, which is all the evidence
    # there is. They are display only — never written into `entities`.
    assert row["title_terms"], "the title carries controlled-vocabulary terms"
    assert not item.entities.methods and not item.entities.data


def test_the_section_renders_with_its_factual_subtitle(repo):
    wl = _whitelist_source_id()
    run = Run.for_date(DAY)
    dark = [_journal_item(100 + i, wl, None) for i in range(2)]
    write_stage(run, "classify", [_journal_item(1, wl, "An abstract.")] + dark)
    run_stages.stage_select(run)
    issue = run_stages.stage_issue(run, DAY)

    html = render_issue(issue, [], unreadable=dark)
    flat = " ".join(html.lower().split())  # the template wraps its copy

    assert "also published today" in flat
    assert "their abstracts are not openly available" in flat
    assert "uc-also__item" in html
    # No publisher is named in the reader-facing copy; the tally is data.
    assert "Elsevier" not in html


def test_the_arxiv_attribution_is_present_verbatim(repo):
    """arXiv's brand guidelines require this wording exactly (P6-1)."""
    wl = _whitelist_source_id()
    run = Run.for_date(DAY)
    write_stage(run, "classify", [_journal_item(1, wl, "An abstract.")])
    run_stages.stage_select(run)
    issue = run_stages.stage_issue(run, DAY)

    html = render_issue(issue, [])
    assert "Thank you to arXiv for use of its open access interoperability." in html
    assert "was not reviewed or approved by" in html
