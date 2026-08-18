"""The Q1b labelling pass.

150 labels cannot be collected twice, so the sampling, the label vocabulary, the
stored format and the aggregation are all pinned here before any are collected.
"""

from __future__ import annotations

import json
from datetime import date

import pytest


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


def test_label_vocabulary_separates_the_kinds_of_drop():
    """Four reasons, not one number.

    `n` is a classifier error, `q` is a coverage-definition question, and weak
    splits again into method and results (M1) because one of those is learnable
    from an abstract and the other is not.
    """
    assert LABEL_KEYS["n"] == "drop_not_urban"
    assert LABEL_KEYS["q"] == "drop_not_our_kind"
    assert LABEL_KEYS["m"] == "drop_weak_method"
    assert LABEL_KEYS["r"] == "drop_weak_results"
    assert set(DROP_LABELS) == {
        "drop_not_urban",
        "drop_not_our_kind",
        "drop_weak_method",
        "drop_weak_results",
        "drop_weak",
    }


def test_the_old_weak_key_is_refused_rather_than_guessed():
    """`w` meant two things. Mapping it to either would put a guess in the file."""
    from pipeline.labeling import LABEL_KEYS as KEYS, LEGACY_WEAK_KEY, _ask_label

    assert LEGACY_WEAK_KEY not in KEYS

    asked = iter(["w", "m"])
    said = []
    key = _ask_label(lambda _p: next(asked), said.append)

    assert key == "m"
    assert any("two labels" in line for line in said)


def test_precision_is_unchanged_by_the_split(repo):
    """The split is diagnostic. Moving a metric by relabelling would be moving
    the goalposts, so all three weak labels stay drops and stay grouped."""
    from pipeline.labeling import is_weak

    assert is_weak("drop_weak")
    assert is_weak("drop_weak_method")
    assert is_weak("drop_weak_results")
    assert not is_weak("keep")


def test_stored_row_carries_everything_needed_to_train_on_it(repo):
    _seed_candidates(repo)
    run_labeling_session(DAY, top=4, prompt=_answers("k", "n", "q", "m"), printer=lambda *a: None)

    rows = load_labels()
    assert len(rows) == 4
    required = {
        "date", "work_key", "source", "rank", "label", "score", "title",
        "has_summary", "classifier_version", "threshold", "labelled_at",
    }
    for r in rows:
        assert required <= set(r), f"missing {required - set(r)}"
    assert {r["label"] for r in rows} == {
        "keep", "drop_not_urban", "drop_not_our_kind", "drop_weak_method"
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
    answers = ["n", "n", "q", "r"] + ["k"] * 16
    run_labeling_session(DAY, top=20, prompt=_answers(*answers), printer=lambda *a: None)

    source = precision_at_k()["by_source"]["arxiv"]
    arxiv = source["drop_reasons"]
    assert arxiv["not_urban"] == 2
    assert arxiv["not_our_kind"] == 1
    # Grouped in the headline count, broken out alongside it.
    assert arxiv["weak"] == 1
    assert source["weak_detail"] == {"method": 0, "results": 1, "unsplit": 0}


def test_precision_reports_absence_rather_than_zero(repo):
    result = precision_at_k()
    assert result["n_labels"] == 0
    assert result["by_source"] == {}
    assert "no labels yet" in result["note"]


def test_items_without_an_abstract_are_excluded_from_the_sample(repo):
    """A label guessed from a title alone is noise, and these labels are
    training data. Measured on 2026-08-05: 6 of 30 sampled items had no
    abstract, all journal-side."""
    wl = _whitelist_source_id()
    items = []
    for i in range(20):
        it = arxiv_item(i, 0.9)
        if i % 2 == 0:
            it.bibliography.abstract = ""
        items.append(it)
    for i in range(20):
        it = journal_item(i, wl)
        it.scores.relevance = 1.0
        if i % 2 == 0:
            it.bibliography.abstract = None
        items.append(it)

    run = Run.for_date(DAY)
    write_stage(run, "classify", items)

    sample = stratified_sample(DAY, per_source=15)
    assert sample, "the sample must not be empty"
    assert all((it.bibliography.abstract or "").strip() for it, _, _ in sample)


def test_precision_is_reported_at_every_depth_not_only_at_k(repo):
    """A path that holds 1.0 to rank 4 and 0.5 by rank 12 is not failing at
    ranking — it is being asked for more items than it has. precision@10 alone
    cannot tell those apart, and the slot split is the decision it informs."""
    _seed_candidates(repo)
    # arXiv side: 4 keeps, then drops. Journal side: all keeps.
    answers = ["k"] * 4 + ["q"] * 6 + ["k"] * 10
    run_labeling_session(DAY, top=20, prompt=_answers(*answers), printer=lambda *a: None)

    arxiv = precision_at_k(k=10)["by_source"]["arxiv"]
    curve = arxiv["precision_by_depth"]

    assert curve[0] == 1.0 and curve[3] == 1.0, "clean to rank 4"
    assert curve[9] == 0.4, "and 4 of 10 by rank 10"
    assert arxiv["depth_holding_0.7"] == 5, "0.7 survives to rank 5, not beyond"

    journal = precision_at_k(k=10)["by_source"]["journal"]
    assert journal["depth_holding_0.7"] == len(journal["precision_by_depth"])


def test_precision_is_broken_out_by_score_band(repo):
    """precision@10 averages over a range where the classifier's confidence
    varies by 60 points. The band table is what answers "where should the
    threshold go" — measurement only; nothing here moves a default."""
    _seed_candidates(repo)
    # arXiv candidates are seeded in descending score order.
    answers = ["k"] * 4 + ["n"] * 6 + ["k"] * 10
    run_labeling_session(DAY, top=20, prompt=_answers(*answers), printer=lambda *a: None)

    arxiv = precision_at_k(k=10)["by_source"]["arxiv"]
    assert arxiv["score_is_single_valued"] is False
    assert arxiv["score_bands"], "arXiv scores span more than one value"
    for band in arxiv["score_bands"]:
        assert 0.0 <= band["keep_rate"] <= 1.0
        assert band["n"] > 0
        assert set(band["drop_reasons"]) == {"not_urban", "not_our_kind", "weak"}


def test_the_journal_path_reports_no_bands_because_it_has_one_score(repo):
    """Every whitelist article scores exactly 1.0 by membership (N4), so a band
    table would be one row pretending to be a distribution."""
    _seed_candidates(repo)
    run_labeling_session(
        DAY, top=20, prompt=_answers(*(["k"] * 20)), printer=lambda *a: None
    )

    journal = precision_at_k(k=10)["by_source"]["journal"]
    assert journal["score_is_single_valued"] is True
    assert journal["score_bands"] == []
    # The drop reasons are still reported: they are what varies on that path.
    assert "drop_reasons" in journal


# --------------------------------------------------------------------------
# The affinity probe's band split (phase 0h)
# --------------------------------------------------------------------------


def test_affinity_bands_cut_at_the_66th_percentile_of_the_positives():
    from pipeline.labeling import affinity_bands

    pool = {f"W{i}": {"canon_affinity": float(i)} for i in range(0, 10)}
    pool["Wz"] = {"canon_affinity": 0.0}
    spec = affinity_bands(pool)
    # Positives are 1..9; the 66th percentile of nine values is the sixth.
    assert spec["high_cut"] == 6.0
    assert spec["sizes"]["high"] == 4  # 6, 7, 8, 9
    assert spec["sizes"]["mid"] == 5  # 1..5
    assert spec["sizes"]["zero"] == 2  # W0 and Wz
    assert "66th percentile" in spec["high_cut_basis"]


def test_zero_affinity_and_no_references_are_not_the_same_band():
    # A candidate with no reference list has affinity zero because nothing was
    # available to score it with. Counting it as "cites no canon" would put a
    # coverage gap inside the probe's negative band.
    from pipeline.labeling import affinity_bands

    pool = {
        "W1": {"canon_affinity": 0.0, "refs_total": 30},
        "W2": {"canon_affinity": 5.0, "refs_total": 40},
    }
    spec = affinity_bands(pool)
    assert spec["bands"]["zero"] == ["W1"]
    # The refs-less ones never reach this function; `affinity_pool` drops them
    # and returns the count separately, which is what makes the band honest.
    import inspect

    from pipeline.labeling import affinity_pool

    assert inspect.signature(affinity_pool).parameters["require_refs"].default is True


# --------------------------------------------------------------------------
# Moving a labelling session between machines (phase 0j, W7)
# --------------------------------------------------------------------------


def test_a_labelling_set_survives_the_round_trip(repo):
    """An export nobody has read back is a backup nobody has restored."""
    import json as _json
    from pathlib import Path

    from pipeline.labeling import export_labeling_set, import_labeling_set
    from pipeline.metrics import Run
    from pipeline.stages import read_stage

    _seed_candidates(repo)
    run = Run.for_date(DAY)
    before = {
        stage: [it.work_key for it in (read_stage(run, stage) or [])]
        for stage in ("classify", "labeling_pool", "summarize")
    }

    out = Path(repo) / "export.json"
    meta = export_labeling_set([DAY], out)
    assert meta["bytes"] > 0
    assert meta["dates"] == [str(DAY)]

    # Wipe the stage files, then restore from the export alone.
    for stage in before:
        path = run.dir / "stages" / f"{stage}.jsonl"
        if path.exists():
            path.unlink()

    restored = import_labeling_set(out)
    assert restored["dates"] == [str(DAY)]

    after = {
        stage: [it.work_key for it in (read_stage(run, stage) or [])]
        for stage in ("classify", "labeling_pool", "summarize")
    }
    assert after["classify"] == before["classify"]
    assert after["classify"], "the export must actually carry the candidate pool"


def test_an_export_from_a_future_version_is_refused(repo):
    import json as _json
    from pathlib import Path

    from pipeline.labeling import import_labeling_set

    path = Path(repo) / "bad.json"
    path.write_text(_json.dumps({"version": "labeling-set@99", "days": {}}), encoding="utf-8")
    try:
        import_labeling_set(path)
    except ValueError as e:
        assert "labeling-set@99" in str(e)
        return
    raise AssertionError("expected a refusal")


def test_the_export_leaves_out_the_raw_responses(repo):
    """Only what a labelling session reads. Raw API responses are the bulk of a
    run directory and nothing in labelling touches them."""
    import json as _json
    from pathlib import Path

    from pipeline.labeling import export_labeling_set

    _seed_candidates(repo)
    out = Path(repo) / "export.json"
    export_labeling_set([DAY], out)
    payload = _json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["days"][str(DAY)]) <= {"classify", "labeling_pool", "summarize"}


def test_a_correction_record_survives_a_read(repo):
    """`corrected_from` / `corrected_by` / `corrected_at` must not be dropped.

    A label is a person's judgement and a person revises judgements. One row in
    the real file carries that history, and it is the only record that the
    revision happened — `load_labels` returns the parsed dict as-is, and this
    pins that, so a later move to a typed row model cannot silently discard it.
    """
    from pipeline.labeling import append_labels, load_labels

    append_labels("relevance", [{
        "date": str(DAY),
        "work_key": "doi:10.1000/corrected",
        "source": "journal",
        "rank": 7,
        "label": "keep",
        "score": 1.0,
        "sampling": "ranked_top_n",
        "corrected_from": "drop_weak",
        "corrected_by": "YJUN",
        "corrected_at": "2026-08-17T16:35:07+00:00",
    }])

    row = load_labels("relevance")[0]
    assert row["label"] == "keep"
    assert row["corrected_from"] == "drop_weak"
    assert row["corrected_by"] == "YJUN"
    assert row["corrected_at"].startswith("2026-08-17")


# --------------------------------------------------------------------------
# Three sampling frames, three questions (phase 0L, N2)
# --------------------------------------------------------------------------


def test_the_three_label_files_cannot_be_pooled(repo):
    """Any two concatenated give a number that looks fine and means nothing."""
    from pipeline.labeling import (
        LabelSetMisuse,
        PROBE_FACETS,
        RANKED_FACETS,
        append_labels,
        precision_at_k,
        sampling_of,
    )

    assert sampling_of("relevance") == "ranked_top_n"
    assert sampling_of("affinity_probe") == "band_stratified"
    assert sampling_of("code_probe") == "code_stratified"
    assert "code_probe" in PROBE_FACETS
    assert "code_probe" not in RANKED_FACETS

    # precision@k is undefined over a probe and says so by name.
    with pytest.raises(LabelSetMisuse):
        precision_at_k("code_probe")

    # And a row from one frame cannot be written into another file.
    with pytest.raises(LabelSetMisuse):
        append_labels(
            "code_probe",
            [{"work_key": "arxiv:1", "label": "keep", "sampling": "ranked_top_n"}],
        )
    with pytest.raises(LabelSetMisuse):
        append_labels(
            "relevance",
            [{"work_key": "arxiv:1", "label": "keep", "sampling": "code_stratified"}],
        )
