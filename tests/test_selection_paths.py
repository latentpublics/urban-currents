"""Two entry paths: whitelist membership, or classifier probability (roadmap §2.1).

Phase 0 put both through one classifier and then imposed an arXiv quota to stop
journal articles taking every slot. These tests pin the replacement: each path
owns its slots, the classifier never sees a whitelist article, and a short day
is recorded rather than hidden.
"""

from __future__ import annotations

from datetime import date

import pytest

from pipeline import run_stages
from pipeline.config import journals_vocab
from pipeline.metrics import Run
from pipeline.models import Bibliography, Ids, Item, PrimaryLocation
from pipeline.stages import write_stage

DAY = date(2026, 8, 11)


def _whitelist_source_id() -> str:
    for s in journals_vocab().get("sources") or []:
        if s.get("include", True):
            return s["id"]
    raise AssertionError("no included journals in the whitelist")


def journal_item(n: int, wl_id: str) -> Item:
    return Item(
        work_key=f"doi:10.1000/j{n}",
        first_published=DAY,
        ids=Ids(doi=f"10.1000/j{n}", openalex=f"W{n}"),
        bibliography=Bibliography(
            title=f"A journal paper {n} about cities",
            abstract="We study urban form using census data across 30 cities.",
            primary_location=PrimaryLocation(
                source_id=wl_id, source_name="Whitelist Journal", type="journal"
            ),
        ),
    )


def arxiv_item(n: int, relevance: float) -> Item:
    it = Item(
        work_key=f"arxiv:2608.{n:05d}",
        first_published=DAY,
        ids=Ids(arxiv=f"2608.{n:05d}"),
        bibliography=Bibliography(
            title=f"An arXiv preprint {n}",
            abstract="We train a graph neural network on street view imagery.",
            primary_location=PrimaryLocation(
                source_id="S4306400194", source_name="arXiv", type="repository"
            ),
            categories=["cs.LG"],
        ),
    )
    it.scores.relevance = relevance
    it.scores.components.relevance = relevance
    return it


def _run_select(repo, items, **kw):
    run = Run.for_date(DAY)
    write_stage(run, "classify", items)
    return run, run_stages.stage_select(run, **kw)


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------


def test_whitelist_membership_is_the_entry_ticket(repo):
    """A journal article enters without a classifier score of its own."""
    wl = _whitelist_source_id()
    items = [journal_item(i, wl) for i in range(3)]
    for it in items:
        assert it.scores.relevance == 0.0  # never classified

    _, selected = _run_select(repo, items)
    assert len(selected) == 3


def test_arxiv_below_threshold_does_not_enter(repo):
    items = [arxiv_item(1, 0.9), arxiv_item(2, 0.10), arxiv_item(3, 0.34)]
    _, selected = _run_select(repo, items, threshold=0.35)
    assert [it.work_key for it in selected] == ["arxiv:2608.00001"]


def test_each_path_owns_its_slots(repo):
    """The old failure: 23 of 24 slots going to journal articles."""
    wl = _whitelist_source_id()
    items = [journal_item(i, wl) for i in range(30)]
    items += [arxiv_item(i, 0.9 - i * 0.001) for i in range(30)]

    run, selected = _run_select(repo, items, threshold=0.35)

    n_journal = sum(1 for it in selected if not it.ids.arxiv)
    n_arxiv = sum(1 for it in selected if it.ids.arxiv)
    assert n_journal == 12
    assert n_arxiv == 12
    assert run.metrics.counts.__pydantic_extra__["selected_journal"] == 12
    assert run.metrics.counts.__pydantic_extra__["selected_arxiv"] == 12


def test_a_path_lends_its_unused_slots_and_the_shortfall_is_recorded(repo):
    wl = _whitelist_source_id()
    items = [journal_item(i, wl) for i in range(2)]
    items += [arxiv_item(i, 0.9) for i in range(30)]

    run, selected = _run_select(repo, items, threshold=0.35)

    assert sum(1 for it in selected if not it.ids.arxiv) == 2
    assert sum(1 for it in selected if it.ids.arxiv) == 22  # borrowed the spare 10
    assert any("short day" in e for e in run.metrics.errors)


def test_arxiv_path_is_ranked_by_probability(repo):
    items = [arxiv_item(1, 0.51), arxiv_item(2, 0.99), arxiv_item(3, 0.72)]
    _, selected = _run_select(repo, items, threshold=0.35)
    arxiv_only = [it for it in selected if it.ids.arxiv]
    assert [it.work_key for it in arxiv_only][0] == "arxiv:2608.00002"


# --------------------------------------------------------------------------
# The classifier never sees a whitelist article
# --------------------------------------------------------------------------


def test_classify_skips_whitelist_journals(repo):
    wl = _whitelist_source_id()
    items = [journal_item(1, wl), arxiv_item(1, 0.0)]
    run = Run.for_date(DAY)
    write_stage(run, "gate", items)

    out = run_stages.stage_classify(run)
    by_key = {it.work_key: it for it in out}

    journal = by_key["doi:10.1000/j1"]
    assert journal.scores.relevance == 1.0
    # Labelled as membership, not dressed up as a model prediction.
    assert journal.provenance.classifier_version == "whitelist-membership"
    assert run.metrics.counts.__pydantic_extra__["classify_skipped_journal"] == 1
    assert run.metrics.counts.classified == 1


# --------------------------------------------------------------------------
# The journal ranking is a placeholder and says so
# --------------------------------------------------------------------------


def test_journal_ranking_ignores_relevance(repo):
    """Relevance is 1.0 for every whitelist article, so it cannot rank them.
    The placeholder uses the components that do vary."""
    wl = _whitelist_source_id()
    a, b = journal_item(1, wl), journal_item(2, wl)
    a.scores.relevance = b.scores.relevance = 1.0
    a.scores.components.artifact_completeness = 1.0
    b.scores.components.artifact_completeness = 0.0

    assert run_stages.journal_rank_score(a) > run_stages.journal_rank_score(b)


def test_journal_rank_score_is_documented_as_a_placeholder():
    doc = run_stages.journal_rank_score.__doc__ or ""
    assert "PLACEHOLDER" in doc
    assert "Q1b" in doc, "the replacement path has to be named in the docstring"


def test_the_old_quota_config_is_gone(repo):
    from pipeline.config import cfg

    assert cfg("classifier.arxiv_min_share") is None
    assert cfg("selection.slots.journal") == 12
    assert not hasattr(run_stages, "_select_with_source_quota")


@pytest.mark.parametrize("top_n,expected", [(6, 6), (10, 10)])
def test_explicit_top_n_splits_evenly(repo, top_n, expected):
    wl = _whitelist_source_id()
    items = [journal_item(i, wl) for i in range(20)]
    items += [arxiv_item(i, 0.9) for i in range(20)]
    _, selected = _run_select(repo, items, threshold=0.35, top_n=top_n)
    assert len(selected) == expected


# --------------------------------------------------------------------------
# Model selection is a decision, not a filename sort
# --------------------------------------------------------------------------


def test_model_version_is_pinned_not_inferred(repo, monkeypatch):
    """Sorting `clf-*.joblib` picks clf-v3, the worst of the three variants."""
    from pipeline.filters import classifier as clf_mod
    from pipeline.paths import MODELS

    MODELS.mkdir(parents=True, exist_ok=True)
    for name in ("clf-v1-2026-08-13", "clf-v2-2026-08-13", "clf-v3-2026-08-13"):
        (MODELS / f"{name}.joblib").write_bytes(b"stub")

    monkeypatch.setattr(clf_mod, "cfg", lambda k, d=None: "clf-v2-2026-08-13"
                        if k == "classifier.model_version" else d)
    assert clf_mod.latest_model_path().stem == "clf-v2-2026-08-13"


def test_an_unresolvable_pin_is_an_error_not_a_silent_fallback(repo, monkeypatch):
    from pipeline.filters import classifier as clf_mod

    monkeypatch.setattr(clf_mod, "cfg", lambda k, d=None: "clf-does-not-exist"
                        if k == "classifier.model_version" else d)
    with pytest.raises(FileNotFoundError):
        clf_mod.latest_model_path()


def test_unsummarisable_items_rank_last_in_the_journal_path(repo):
    """An item with no abstract yields a card with only a title. Measured on the
    prepared days: 5-10 of 24 published cards were in that state."""
    wl = _whitelist_source_id()
    with_abstract = journal_item(1, wl)
    without = journal_item(2, wl)
    without.bibliography.abstract = None

    assert run_stages.journal_rank_score(without) < run_stages.journal_rank_score(
        with_abstract
    )

    # 16 journal candidates for 12 slots, with the arXiv path full so no slots
    # are lent: the one that cannot be summarised is the one left out. Still
    # publishable in principle — just last in line.
    items = [without] + [journal_item(i, wl) for i in range(10, 25)]
    items += [arxiv_item(i, 0.9) for i in range(12)]
    _, selected = _run_select(repo, items)
    assert without.work_key not in {it.work_key for it in selected}
