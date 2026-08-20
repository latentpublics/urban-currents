"""What happens when the classifier is not the classifier (phase 0X, X1).

`load_classifier` falls back to a keyword heuristic when the trained model or
its embeddings cannot be loaded, and that fallback is right: a stale joblib
should cost accuracy, not a day. What was wrong is that it happened in silence.

The silence mattered more than it looks. `selection.arxiv.floor` is 0.80 and
was calibrated against the trained model; the heuristic scores an ordinary
abstract between 0.05 and 0.4. So a fallback does not make the arXiv path
worse — it **empties** it, and the issue goes out as a journal-only digest
still claiming to be a scan of the field. That is exactly the failure
`REQUIRED_SOURCES` exists for, one layer down, so it is now reported the same
way: a reason on the prediction, DEGRADED on the stage, and a refusal from
`looked()`.

The second test is the one that keeps the first honest — it loads the real
model, and **skips with a stated reason** when there is none, so "did not run"
can never be read as "ran and passed".
"""

from __future__ import annotations

from datetime import date

import pytest

from pipeline import run_stages
from pipeline.filters import classifier as clf_mod
from pipeline.metrics import Run
from pipeline.outcome import looked

DAY = date(2026, 8, 11)


def test_the_heuristic_carries_the_reason_it_was_reached(repo):
    """The reason used to be eaten by a bare `except Exception`."""
    heuristic = clf_mod.HeuristicClassifier("clf-v2.joblib could not be used: "
                                            "EmbeddingUnavailable: no sentence-transformers")
    pred = heuristic.predict([])

    assert pred.version == "heuristic-v0"
    assert "sentence-transformers" in pred.fallback_reason


def test_a_trained_prediction_carries_no_fallback_reason(repo):
    """`None` is the signal that the real model ran. It has to stay `None`."""
    from pipeline.filters.classifier import Prediction

    assert Prediction(probabilities=[0.9], version="clf-v2-2026-08-13").fallback_reason is None


def test_a_fallback_marks_the_stage_degraded_and_says_why(repo, sample_date, monkeypatch):
    """The whole point of X1's production half."""
    from pipeline.filters.classifier import Prediction

    run = Run.for_date(sample_date)
    items = run_stages.stage_collect(run, sample_date, fixture=True)
    run_stages.write_stage(run, "gate", items)

    def heuristic_pred(arxiv_items, clf=None):
        return Prediction(
            probabilities=[0.12] * len(arxiv_items),
            version="heuristic-v0",
            fallback_reason="clf-v2-2026-08-13.joblib could not be used: "
                            "EmbeddingUnavailable: sentence-transformers is not installed",
        )

    monkeypatch.setattr(run_stages, "score_items", heuristic_pred)

    run_stages.stage_classify(run)

    assert run.metrics.stages["classify"] == "DEGRADED"
    assert run.metrics.stages["classify.model"] == "heuristic-v0"

    said = " ".join(run.metrics.errors)
    assert "fell back to heuristic-v0" in said
    assert "sentence-transformers" in said, "the reason has to travel, not just the fact"
    assert "0.8" in said, "the floor is why this matters; name it"


def test_looked_refuses_a_day_the_heuristic_scored(repo, sample_date):
    """`DEGRADED` is a verdict-changing fact, like SKIPPED before it."""
    run = Run.for_date(sample_date)
    run.metrics.stages.update({
        "collect": "OK", "collect.arxiv": "OK", "collect.openalex": "OK",
        "classify": "DEGRADED", "select": "OK", "summarize": "OK", "issue": "OK",
    })

    ok, reasons = looked(run)

    assert ok is False
    assert any("classify ran degraded" in r for r in reasons)


def test_an_ok_classify_is_still_ok(repo, sample_date):
    """Guard against the check above refusing everything."""
    run = Run.for_date(sample_date)
    run.metrics.stages.update({
        "collect": "OK", "collect.arxiv": "OK", "collect.openalex": "OK",
        "classify": "OK", "select": "OK", "summarize": "OK", "issue": "OK",
    })

    ok, reasons = looked(run)

    assert ok is True, reasons


# --------------------------------------------------------------------------
# The real model, when it is there
# --------------------------------------------------------------------------


@pytest.mark.real_classifier
def test_the_trained_classifier_loads_when_it_is_present(repo, trained_classifier):
    """The one test that does need the 440 MB.

    It opts out of the stub, and when the model or `sentence-transformers` is
    absent it **skips with the reason printed** (`-rs` in pyproject.toml)
    rather than passing quietly. On CI this skips; on a developer machine it
    runs, and that asymmetry is stated rather than hidden.
    """
    assert trained_classifier.version.startswith("clf-")
    assert not isinstance(trained_classifier, clf_mod.HeuristicClassifier)


@pytest.mark.real_classifier
def test_the_trained_model_clears_the_floor_on_the_fixtures(repo, sample_date, trained_classifier):
    """What the stub stands in for, checked against the real thing.

    The stub is calibrated to put the built-in fixtures above
    `selection.arxiv.floor`. This is the assertion that says the stub is
    standing in for something true rather than inventing a world where the
    pipeline works.
    """
    from pipeline.config import cfg

    items = run_stages.stage_collect(Run.for_date(sample_date), sample_date, fixture=True)
    pred = trained_classifier.predict(items)
    floor = float(cfg("selection.arxiv.floor", 0.80))

    assert pred.fallback_reason is None
    assert max(pred.probabilities) >= floor, (
        f"the trained model scores the fixtures {pred.probabilities}, "
        f"all below the {floor} floor — then the stub is not standing in for it"
    )
