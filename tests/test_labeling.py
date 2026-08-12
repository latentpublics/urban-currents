"""The Q1b labelling pass.

150 labels cannot be collected twice, so the sampling, the label vocabulary, the
stored format and the aggregation are all pinned here before any are collected.
"""

from __future__ import annotations

import json
from datetime import date


from pipeline.labeling import (
    DROP_LABELS,
    LABEL_KEYS,
    labels_path,
    load_labels,
    precision_at_k,
    run_labeling_session,
    stratified_sample,
)
from pipeline.metrics import Run
from pipeline.models import SummaryEn
from pipeline.stages import write_stage
from tests.test_selection_paths import _whitelist_source_id, arxiv_item, journal_item

DAY = date(2026, 8, 11)


def _seed_candidates(repo, n_arxiv=25, n_journal=25, with_summary=True):
    wl = _whitelist_source_id()
    items = []
    for i in range(n_arxiv):
        it = arxiv_item(i, 0.95 - i * 0.01)
        if with_summary:
            it.summary.en = SummaryEn(what=f"What {i}.", why=f"Why {i}.")
        items.append(it)
    for i in range(n_journal):
        it = journal_item(i, wl)
        it.scores.relevance = 1.0
        it.scores.components.artifact_completeness = 1.0 - i * 0.01
        if with_summary:
            it.summary.en = SummaryEn(what=f"Journal what {i}.", why=f"Journal why {i}.")
        items.append(it)
    run = Run.for_date(DAY)
    write_stage(run, "classify", items)
    return run


def _answers(*seq):
    it = iter(seq)

    def prompt(_message: str) -> str:
        return next(it, "s")

    return prompt


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_sample_is_stratified_by_source(repo):
    _seed_candidates(repo)
    sample = stratified_sample(DAY, per_source=15)

    assert len(sample) == 30
    sources = [s for _, s, _ in sample]
    assert sources.count("arxiv") == 15
    assert sources.count("journal") == 15


def test_ranks_are_per_source_and_start_at_one(repo):
    _seed_candidates(repo)
    sample = stratified_sample(DAY, per_source=15)
    for source in ("arxiv", "journal"):
        ranks = [r for _, s, r in sample if s == source]
        assert ranks == list(range(1, 16))


def test_sample_is_drawn_from_candidates_not_the_published_issue(repo):
    """precision@10 measures the ranking, so sampling only what already cleared
    the daily slots would measure the slots instead."""
    run = _seed_candidates(repo, n_arxiv=25, n_journal=25)
    from pipeline import run_stages

    published = run_stages.stage_select(run)
    assert len(published) == 24

    sample = stratified_sample(DAY, per_source=15)
    sampled_keys = {it.work_key for it, _, _ in sample}
    assert len(sampled_keys) == 30
    assert sampled_keys - {it.work_key for it in published}, (
        "the sample must reach past the published cut"
    )


def test_arxiv_side_respects_the_threshold(repo):
    wl = _whitelist_source_id()
    items = [arxiv_item(1, 0.9), arxiv_item(2, 0.10)] + [journal_item(1, wl)]
    run = Run.for_date(DAY)
    write_stage(run, "classify", items)

    sample = stratified_sample(DAY, per_source=15, threshold=0.35)
    arxiv_keys = [it.work_key for it, s, _ in sample if s == "arxiv"]
    assert arxiv_keys == ["arxiv:2608.00001"]


# --------------------------------------------------------------------------
# Label vocabulary and stored format
# --------------------------------------------------------------------------


def test_label_vocabulary_separates_the_two_kinds_of_drop():
    """`n` is a classifier error; `q` is a coverage-definition question. One
    precision number mixes them, which is the reason this tool exists."""
    assert LABEL_KEYS["n"] == "drop_not_urban"
    assert LABEL_KEYS["q"] == "drop_not_our_kind"
    assert LABEL_KEYS["w"] == "drop_weak"
    assert set(DROP_LABELS) == {"drop_not_urban", "drop_not_our_kind", "drop_weak"}


def test_stored_row_carries_everything_needed_to_train_on_it(repo):
    _seed_candidates(repo)
    run_labeling_session(DAY, top=4, prompt=_answers("k", "n", "q", "w"), printer=lambda *a: None)

    rows = load_labels()
    assert len(rows) == 4
    required = {
        "date", "work_key", "source", "rank", "label", "score", "title",
        "has_summary", "classifier_version", "threshold", "labelled_at",
    }
    for r in rows:
        assert required <= set(r), f"missing {required - set(r)}"
    assert {r["label"] for r in rows} == {
        "keep", "drop_not_urban", "drop_not_our_kind", "drop_weak"
    }


def test_labels_are_appended_as_jsonl(repo):
    _seed_candidates(repo)
    run_labeling_session(DAY, top=2, prompt=_answers("k", "n"), printer=lambda *a: None)
    text = labels_path().read_text(encoding="utf-8")
    assert text.count("\n") == 2
    for line in text.splitlines():
        json.loads(line)


def test_skip_is_not_stored_so_it_can_be_offered_again(repo):
    _seed_candidates(repo)
    out = run_labeling_session(DAY, top=2, prompt=_answers("s", "k"), printer=lambda *a: None)
    assert out["labelled"] == 1
    assert load_labels()[0]["label"] == "keep"


def test_unrecognised_input_is_treated_as_skip(repo):
    _seed_candidates(repo)
    out = run_labeling_session(DAY, top=2, prompt=_answers("zzz", "k"), printer=lambda *a: None)
    assert out["labelled"] == 1


# --------------------------------------------------------------------------
# Resuming
# --------------------------------------------------------------------------


def test_session_resumes_where_it_stopped(repo):
    """Five days of labelling is not done in one sitting."""
    _seed_candidates(repo)
    first = run_labeling_session(
        DAY, top=30, prompt=_answers(*(["k"] * 5 + ["quit"])), printer=lambda *a: None
    )
    assert first["labelled"] == 5
    assert first["stopped_early"] is True
    assert first["remaining"] == 25

    second = run_labeling_session(
        DAY, top=30, prompt=_answers(*(["k"] * 3 + ["quit"])), printer=lambda *a: None
    )
    assert second["labelled"] == 3

    keys = [r["work_key"] for r in load_labels()]
    assert len(keys) == len(set(keys)), "resuming must not re-offer a labelled item"


def test_labelling_time_is_recorded(repo):
    _seed_candidates(repo)
    run_labeling_session(DAY, top=2, prompt=_answers("k", "k"), printer=lambda *a: None)
    assert "label_s" in Run.for_date(DAY).metrics.timing


def test_summary_is_shown_when_present(repo):
    """Without a summary this takes 45 minutes instead of 15."""
    _seed_candidates(repo, with_summary=True)
    shown: list[str] = []
    run_labeling_session(
        DAY, top=2, prompt=_answers("k", "k"), printer=lambda *a: shown.append(" ".join(map(str, a)))
    )
    blob = "\n".join(shown)
    assert "WHAT:" in blob and "WHY :" in blob


def test_missing_summary_falls_back_to_the_abstract(repo):
    _seed_candidates(repo, with_summary=False)
    shown: list[str] = []
    run_labeling_session(
        DAY, top=2, prompt=_answers("k", "k"), printer=lambda *a: shown.append(" ".join(map(str, a)))
    )
    assert "(no summary)" in "\n".join(shown)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_precision_is_reported_per_source_never_blended(repo):
    _seed_candidates(repo)
    # arXiv side: 8 keeps then 2 drops. Journal side: all keeps.
    answers = ["k"] * 8 + ["n", "q"] + ["k"] * 10
    run_labeling_session(DAY, top=20, prompt=_answers(*answers), printer=lambda *a: None)

    result = precision_at_k(k=10)
    assert set(result["by_source"]) == {"arxiv", "journal"}
    assert result["by_source"]["arxiv"]["precision_at_10"] == 0.8
    assert result["by_source"]["journal"]["precision_at_10"] == 1.0
    # No single blended figure is offered at all.
    assert "precision_at_10" not in result


def test_drop_reasons_are_counted_separately(repo):
    _seed_candidates(repo)
    answers = ["n", "n", "q", "w"] + ["k"] * 16
    run_labeling_session(DAY, top=20, prompt=_answers(*answers), printer=lambda *a: None)

    arxiv = precision_at_k()["by_source"]["arxiv"]["drop_reasons"]
    assert arxiv["not_urban"] == 2
    assert arxiv["not_our_kind"] == 1
    assert arxiv["weak"] == 1


def test_precision_reports_absence_rather_than_zero(repo):
    result = precision_at_k()
    assert result["n_labels"] == 0
    assert result["by_source"] == {}
    assert "no labels yet" in result["note"]
