"""Duplicate merge and the preprint → published transition (PRD §5.2, §9)."""

from __future__ import annotations

from datetime import date


from pipeline import run_stages, store
from pipeline.collectors.base import ARXIV_SOURCE_ID
from pipeline.dedup.merge import fuzzy_same, merge_candidates, merge_keys
from pipeline.metrics import Run
from pipeline.models import (
    Author,
    Bibliography,
    Graph,
    Ids,
    Item,
    PrimaryLocation,
    Provenance,
    PublicationStatus,
    TopicRef,
)

TITLE = "Street-View Imagery and Pedestrian Volume: A Twelve-City Model"


def preprint(**over) -> Item:
    base = Item(
        work_key="arxiv:2608.01234",
        first_published=date(2026, 8, 11),
        ids=Ids(arxiv="2608.01234", doi="10.48550/arxiv.2608.01234"),
        bibliography=Bibliography(
            title=TITLE,
            authors=[Author(name="Rui Alvarez"), Author(name="Mina Park")],
            publication_date=date(2026, 8, 11),
            primary_location=PrimaryLocation(
                source_id=ARXIV_SOURCE_ID, source_name="arXiv", type="repository",
                landing_page_url="https://arxiv.org/abs/2608.01234",
            ),
            abstract="We train a model on 3.4M street-view images across 12 cities.",
            categories=["cs.CV", "cs.CY"],
        ),
        publication_status=PublicationStatus(state="preprint"),
        provenance=Provenance(collectors=["arxiv"]),
    )
    for k, v in over.items():
        setattr(base, k, v)
    return base


def journal_version(**over) -> Item:
    """The same paper as OpenAlex sees it once Cities publishes it."""
    base = Item(
        work_key="arxiv:2608.01234",
        first_published=date(2026, 12, 1),
        ids=Ids(
            arxiv="2608.01234",
            openalex="W4392000001",
            doi="10.1016/j.cities.2026.104999",
        ),
        bibliography=Bibliography(
            title=TITLE,
            authors=[Author(name="Rui Alvarez"), Author(name="Mina Park")],
            publication_date=date(2026, 12, 1),
            primary_location=PrimaryLocation(
                source_id="S137445289", source_name="Cities", type="journal",
                landing_page_url="https://example.org/cities/104999",
            ),
            abstract="We train a model on 3.4M street-view images across 12 cities.",
        ),
        publication_status=PublicationStatus(
            state="published", journal="Cities", source_id="S137445289",
            doi="10.1016/j.cities.2026.104999",
        ),
        graph=Graph(referenced_works=["openalex:W2145"], cited_by_count=3),
        provenance=Provenance(collectors=["openalex"]),
    )
    base.entities.topics = [TopicRef(id="openalex:T10746", label="Urban Transport")]
    for k, v in over.items():
        setattr(base, k, v)
    return base


# --------------------------------------------------------------------------
# Merge keys
# --------------------------------------------------------------------------


def test_arxiv_doi_reduces_to_the_arxiv_key():
    """Rule 1: 10.48550/arxiv.* must not look like a separate DOI identity."""
    keys = merge_keys(preprint())
    assert "arxiv:2608.01234" in keys
    assert not any(k.startswith("doi:10.48550") for k in keys)


def test_journal_version_shares_the_arxiv_key():
    assert "arxiv:2608.01234" in merge_keys(journal_version())


def test_fuzzy_match_needs_both_title_and_first_author():
    a = preprint()
    b = preprint(work_key="openalex:W1")
    b.ids = Ids()
    # Rule 3 is for punctuation and casing drift between sources, not for
    # rewordings — the threshold is 95 precisely so it stays conservative.
    b.bibliography.title = "STREET VIEW IMAGERY AND PEDESTRIAN VOLUME; A TWELVE CITY MODEL"
    assert fuzzy_same(a, b)

    c = preprint(work_key="openalex:W2")
    c.ids = Ids()
    c.bibliography.title = a.bibliography.title
    c.bibliography.authors = [Author(name="Someone Else")]
    assert not fuzzy_same(a, c)


def test_merge_collapses_the_pair_and_keeps_the_arxiv_work_key():
    result = merge_candidates([preprint(), journal_version()], run_date=date(2026, 12, 1))
    assert len(result.items) == 1
    merged = result.items[0]
    # work_key priority is arXiv → DOI → OpenAlex, and it never changes.
    assert merged.work_key == "arxiv:2608.01234"
    assert merged.ids.openalex == "W4392000001"
    assert merged.publication_status.state == "published"
    assert merged.publication_status.journal == "Cities"
    assert merged.graph.referenced_works == ["openalex:W2145"]
    assert merged.entities.topics[0].id == "openalex:T10746"
    assert merged.cluster.merge_basis in ("doi_match", "arxiv_location")
    assert set(merged.provenance.collectors) == {"arxiv", "openalex"}


def test_merge_scales_to_a_backfill_sized_batch():
    """The fuzzy pass is blocked by first-author surname. Unblocked it is O(n^2):
    a 90-day backfill (~40,000 candidates) never finishes."""
    import time

    from pipeline.models import Ids as _Ids

    cands = [
        Item(
            work_key=f"arxiv:2608.{i:05d}",
            ids=_Ids(arxiv=f"2608.{i:05d}"),
            bibliography=Bibliography(
                title=f"Distinct study {i} of a completely separate subject",
                authors=[Author(name=f"Given{i} Surname{i}")],
            ),
        )
        for i in range(4000)
    ]
    t0 = time.monotonic()
    result = merge_candidates(cands, run_date=date(2026, 8, 11))
    elapsed = time.monotonic() - t0

    assert len(result.items) == 4000  # all distinct
    assert elapsed < 10, f"merge took {elapsed:.1f}s for 4,000 candidates"


def test_unrelated_papers_are_not_merged():
    other = preprint(work_key="arxiv:2608.09999")
    other.ids = Ids(arxiv="2608.09999")
    other.bibliography.title = "Completely Different Work on Soil Chemistry"
    other.bibliography.authors = [Author(name="Ada Nkemdirim")]
    result = merge_candidates([preprint(), other])
    assert len(result.items) == 2


# --------------------------------------------------------------------------
# The branch that matters: preprint → published is an update, not a new item
# --------------------------------------------------------------------------


def _publish_day(repo, d: date, items: list[Item]) -> None:
    run = Run.for_date(d)
    from pipeline.stages import write_stage

    write_stage(run, "collect", items)
    run_stages.stage_dedup(run, d)
    run_stages.stage_gate(run)
    run_stages.stage_classify(run)
    run_stages.stage_select(run, threshold=0.0)
    run_stages.stage_score(run)
    run_stages.stage_issue(run, d)


def test_preprint_to_published_updates_the_item_and_creates_no_new_one(repo):
    """Miss this branch and the same paper headlines twice, four months apart."""
    day1, day2 = date(2026, 8, 11), date(2026, 12, 1)

    _publish_day(repo, day1, [preprint()])
    assert len(store.all_item_files()) == 1
    first = store.load_issue(day1)
    assert first.items == ["arxiv:2608.01234"]
    assert store.load_item("arxiv:2608.01234").publication_status.state == "preprint"

    _publish_day(repo, day2, [journal_version()])

    # Still exactly one Item on disk — no second file, no second publication.
    assert len(store.all_item_files()) == 1
    updated = store.load_item("arxiv:2608.01234")
    assert updated.publication_status.state == "published"
    assert updated.publication_status.journal == "Cities"

    second = store.load_issue(day2)
    assert second.items == []          # not re-published
    assert len(second.status_changes) == 1
    change = second.status_changes[0]
    assert change.work_key == "arxiv:2608.01234"
    assert change.from_ == "preprint"
    assert change.to == "published"
    assert change.journal == "Cities"


def test_merge_preserves_category_order(repo):
    """arXiv puts the primary category first. A merge that re-sorted the list
    also made the second run of a day rewrite every file."""
    a = preprint()
    b = preprint()
    b.bibliography.categories = ["cs.AI", "cs.CV"]
    merged = merge_candidates([a, b]).items[0]
    assert merged.bibliography.categories[:2] == ["cs.CV", "cs.CY"]
    assert "cs.AI" in merged.bibliography.categories


def test_same_day_rerun_still_publishes(repo):
    """The 'already published' check keys off issue membership, so re-running a
    day is idempotent rather than producing an empty issue."""
    day = date(2026, 8, 11)
    _publish_day(repo, day, [preprint()])
    _publish_day(repo, day, [preprint()])
    issue = store.load_issue(day)
    assert issue.items == ["arxiv:2608.01234"]


def test_badges_reflect_publication_state(repo):
    day1, day2 = date(2026, 8, 11), date(2026, 12, 1)
    _publish_day(repo, day1, [preprint()])
    assert "preprint" in store.load_item("arxiv:2608.01234").badges
    _publish_day(repo, day2, [journal_version()])
    assert "published" in store.load_item("arxiv:2608.01234").badges
