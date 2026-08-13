"""`uc review` and `uc report`.

`uc review` needs a human, so per the brief it is verified **only** by tests —
the interactive prompt and the browser opener are injected here.
"""

from __future__ import annotations

from datetime import date

from pipeline import run_stages, store
from pipeline.metrics import Run
from pipeline.report import build_report, cost_summary
from pipeline.review import (
    ReviewOutcome,
    changed_paths,
    flatten,
    run_review_session,
)

DAY = date(2026, 8, 11)


def _seed_day(repo) -> None:
    run_stages.run_all(DAY, fixture=True, use_llm=False)


def _answers(*seq):
    """Prompt stub returning each answer in turn, then 's' forever."""
    it = iter(seq)

    def prompt(_message: str) -> str:
        return next(it, "s")

    return prompt


# --------------------------------------------------------------------------
# Edit-path bookkeeping
# --------------------------------------------------------------------------


def test_flatten_produces_field_paths():
    flat = flatten({"summary": {"en": {"what": "a"}}, "badges": ["code", "preprint"]})
    assert flat["summary.en.what"] == "a"
    assert flat["badges[0]"] == "code"


def test_changed_paths_lists_only_what_moved():
    before = {"summary": {"en": {"what": "a", "why": "b"}}}
    after = {"summary": {"en": {"what": "A", "why": "b"}}}
    assert changed_paths(before, after) == ["summary.en.what"]


# --------------------------------------------------------------------------
# Review session
# --------------------------------------------------------------------------


def test_review_records_decisions_and_elapsed_time(repo):
    _seed_day(repo)
    opened: list = []

    outcome = run_review_session(
        DAY, prompt=_answers("a", "r", "s"), opener=lambda p: opened.append(p)
    )

    assert outcome.approved == 1
    assert outcome.rejected == 1
    assert outcome.skipped == 1
    assert outcome.seconds >= 0
    assert opened and opened[0].name == "preview.html"

    # Q4's only evidence: the clock, written to metrics rather than remembered.
    run = Run.for_date(DAY)
    assert "review_s" in run.metrics.timing
    assert run.metrics.stages["review"] == "OK"

    issue = store.load_issue(DAY)
    statuses = {store.load_item(k).review.status for k in issue.items}
    assert statuses == {"approved", "rejected", "pending"}


def test_edit_records_the_changed_field_paths(repo):
    """review.edits is the material for fixing prompts in Phase 1."""
    _seed_day(repo)

    def fake_editor(item):
        from pipeline.models import SummaryEn

        edited = item.model_copy(deep=True)
        edited.summary.en = SummaryEn(what="Rewritten by hand.", why="Because.")
        return edited

    outcome = run_review_session(
        DAY, prompt=_answers("e"), opener=lambda p: None, editor=fake_editor
    )

    assert outcome.edited == 1
    work_key = sorted(outcome.edits)[0]
    item = store.load_item(work_key)
    assert item.review.status == "edited"
    assert "summary.en.what" in item.review.edits
    # The bookkeeping fields themselves are not logged as edits.
    assert not any(p.startswith("review.") for p in item.review.edits)


def test_discarded_edit_leaves_the_item_alone(repo):
    _seed_day(repo)
    outcome = run_review_session(
        DAY, prompt=_answers("e"), opener=lambda p: None, editor=lambda item: None
    )
    # A discarded edit counts as a skip, so all three items end up skipped.
    assert outcome.edited == 0
    assert outcome.skipped == 3
    assert outcome.edits == {}


def test_review_of_a_missing_issue_is_not_an_error(repo):
    outcome = run_review_session(
        date(2030, 1, 1), prompt=_answers(), opener=lambda p: None
    )
    assert outcome == ReviewOutcome()


# --------------------------------------------------------------------------
# Labelling mode moved to pipeline/labeling.py — see tests/test_labeling.py.
# What stays here is the full review mode, because Q4's timing depends on it and
# the labelling rework was not allowed to touch it.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def test_report_marks_unmeasured_things_as_unmeasured(repo):
    _seed_day(repo)
    path = build_report()
    text = path.read_text(encoding="utf-8")

    assert "# Urban Currents — Phase 0 report" in text
    # No labels and no trained model yet, so those rows must not claim a result.
    assert "PENDING-HUMAN" in text
    assert "No trained model found" in text
    assert "Not calibrated" in text
    assert "not run" in text


def test_report_includes_measured_costs_and_archive_counts(repo):
    _seed_day(repo)
    text = build_report().read_text(encoding="utf-8")
    assert "## Cost, measured" in text
    assert "## Archive" in text
    assert "| items | 3 |" in text


def test_report_is_regenerable_without_drift(repo):
    """Two runs differ only by the generation timestamp line."""
    _seed_day(repo)
    first = build_report().read_text(encoding="utf-8").splitlines()
    second = build_report().read_text(encoding="utf-8").splitlines()
    assert [ln for ln in first if not ln.startswith("Generated ")] == [
        ln for ln in second if not ln.startswith("Generated ")
    ]


def test_cost_summary_extrapolates_only_from_real_days(repo):
    _seed_day(repo)
    from pipeline.report import load_runs

    costs = cost_summary(load_runs())
    assert costs["days"] == 1
    assert costs["published"] == 3
    assert costs["total_usd"] == 0.0


# --------------------------------------------------------------------------
# The report must describe the model that is actually running
# --------------------------------------------------------------------------


def _write_model_meta(repo, version: str, auc: float) -> None:
    import json

    from pipeline import paths

    paths.MODELS.mkdir(parents=True, exist_ok=True)
    (paths.MODELS / f"{version}.json").write_text(
        json.dumps({
            "version": version,
            "variant": version.split("-")[1],
            "embedding_model": "test-embeddings",
            "embedding_dim": 8,
            "threshold": 0.35,
            "n_train": 100,
            "headline_task": "arxiv_urban_vs_arxiv_other",
            "metrics": {"auc": auc, "average_precision": 0.9, "n": 80,
                        "at_threshold": {"0.35": {"precision": 0.8, "recall": 0.9,
                                                  "flagged_rate": 0.3}}},
        }),
        encoding="utf-8",
    )


def test_the_report_reads_the_pinned_model_not_the_last_filename(repo, monkeypatch):
    """With v1/v2/v3 on disk a filename sort picks v3 — the variant the
    comparison rejected — and its AUC became the headline figure while the
    pipeline ran v2."""
    from pipeline import config, report

    _write_model_meta(repo, "clf-v1-2026-08-13", 0.11)
    _write_model_meta(repo, "clf-v2-2026-08-13", 0.22)
    _write_model_meta(repo, "clf-v3-2026-08-13", 0.33)

    monkeypatch.setattr(
        report, "cfg",
        lambda k, d=None: "clf-v2-2026-08-13" if k == "classifier.model_version" else d,
    )
    assert report.load_classifier_meta()["metrics"]["auc"] == 0.22

    config.reset_caches()


def test_a_pin_with_no_metadata_is_named_not_swapped_for_another_model(repo, monkeypatch):
    from pipeline import report

    _write_model_meta(repo, "clf-v3-2026-08-13", 0.33)
    monkeypatch.setattr(
        report, "cfg",
        lambda k, d=None: "clf-v9-nonexistent" if k == "classifier.model_version" else d,
    )
    meta = report.load_classifier_meta()
    assert "clf-v9-nonexistent" in meta["error"]
    assert "metrics" not in meta
