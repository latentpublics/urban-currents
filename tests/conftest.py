"""Test fixtures.

Every test runs against a throwaway repo root so nothing touches the real
``content/`` or ``runs/``. Tests must pass with no API keys and no network.

**And with no embedding model** (phase 0X, X1). That promise was written in
`ci.yml` and was not true: without `sentence-transformers` the classifier falls
back to a keyword heuristic whose scores sit far below
`selection.arxiv.floor`, so `select` dropped everything and twenty tests failed
on a clean checkout while passing on a laptop that happened to have the model
and a warm cache. A test suite that is green only where there is 440 MB of
downloaded weights is not testing what it claims to test.

The fix is the one CLAUDE.md already applies to the LLM — *"Unit tests never
hit a real API. Inject `LLMClient(caller=...)`"*. The embedding model is the
same kind of heavy external dependency, so `stub_classifier` injects a
deterministic scorer for every test that uses `repo`. Nothing is asserted more
weakly because of it: the pipeline tests assert order, publication and
idempotency, and they still have to earn those. What they no longer assert is
"a 440 MB download is present".

Tests that are *about* the trained classifier ask for `trained_classifier`,
which skips with a visible reason when the model or its embeddings are absent.
"""

from __future__ import annotations

import importlib
import os
from datetime import date
from pathlib import Path

import pytest


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch):
    """A temporary UC_ROOT with config and vocab copied from the real repo."""
    import shutil

    real_root = Path(__file__).resolve().parent.parent
    for sub in ("config", "vocab"):
        shutil.copytree(real_root / sub, tmp_path / sub)

    monkeypatch.setenv("UC_ROOT", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENALEX_KEY", raising=False)

    from pipeline import config, paths

    # Reloading `paths` re-executes it inside the same module object, so every
    # module that looks up `paths.X` at call time picks up the new root. Only
    # modules that copied a path into a module-level constant need reloading —
    # and reloading more than that would break exception-class identity
    # (`except LLMBudgetExceeded` stops matching a freshly created class).
    importlib.reload(paths)
    config.reset_caches()
    importlib.reload(importlib.import_module("pipeline.filters.embed"))
    paths.ensure_dirs()
    yield tmp_path
    config.reset_caches()


@pytest.fixture()
def sample_date() -> date:
    return date(2026, 8, 11)


@pytest.fixture()
def fake_env(monkeypatch):
    """Pretend both keys exist without ever reading the real .env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENALEX_KEY", "test-key-not-real")
    from pipeline import config

    config._ENV_LOADED = True
    yield
    config._ENV_LOADED = False


def pytest_configure():
    os.environ.setdefault("UC_TESTING", "1")


# --------------------------------------------------------------------------
# The classifier the tests run against (phase 0X, X1)
# --------------------------------------------------------------------------


class StubClassifier:
    """Deterministic relevance, on the same scale as the trained model.

    Not the heuristic. The heuristic is production's fallback and saturates at
    0.95 while scoring an ordinary abstract 0.05-0.4 — it exists to keep a run
    alive, not to be calibrated. This is calibrated by construction: text that
    looks like urban data science scores above `selection.arxiv.floor`, text
    that does not scores below it, and the same input always gives the same
    number.

    The version string says what it is. If a test ever asserts a real model
    number it will fail here rather than quietly pass on a stub.
    """

    version = "stub-classifier-v1"

    # Deliberately short and readable. Every fixture item hits several; the
    # negative fixtures in the gate tests hit none.
    TERMS = (
        "urban", "city", "cities", "metropolitan", "street", "transit",
        "pedestrian", "accessibility", "mobility", "travel", "land use",
        "neighbourhood", "neighborhood", "housing", "census", "spatial",
        "gis", "openstreetmap", "built environment", "walkability",
        "origin-destination", "commute", "traffic", "planning",
    )

    def predict(self, items):
        from pipeline.filters.classifier import Prediction

        probs = []
        for it in items:
            text = f"{it.bibliography.title}\n{it.bibliography.abstract or ''}".lower()
            hits = sum(1 for t in self.TERMS if t in text)
            # Two distinct urban terms is enough to clear the 0.80 floor, which
            # is roughly where the trained model puts these fixtures. One term
            # or none stays below it, so a test can still write an item that
            # must be rejected and have it rejected.
            probs.append(round(min(0.97, 0.35 + 0.28 * hits), 4) if hits else 0.05)
        return Prediction(probabilities=probs, version=self.version)


@pytest.fixture(autouse=True)
def stub_classifier(request, monkeypatch):
    """Every test that touches the pipeline gets the stub, everywhere.

    Autouse and unconditional on purpose. Making it conditional on whether the
    real model happens to be installed would reproduce the bug it exists to
    fix: two different behaviours, one of them only visible in CI.

    Opt out with `@pytest.mark.real_classifier` — see `trained_classifier`.
    """
    if "real_classifier" in request.keywords:
        return
    from pipeline.filters import classifier as clf_mod

    monkeypatch.setattr(clf_mod, "load_classifier", lambda *a, **k: StubClassifier())


@pytest.fixture()
def trained_classifier():
    """The real thing, or a skip that says why.

    The distinction this keeps is the project's own: a test that did not run is
    not a test that passed. `-rs` in pyproject.toml prints the reason, so
    "3 skipped" cannot hide a model that quietly stopped loading.
    """
    from pipeline.filters.classifier import latest_model_path, load_classifier

    if latest_model_path() is None:
        pytest.skip("no trained model in models/ — nothing to load")
    try:
        clf = load_classifier(allow_fallback=False)
    except Exception as e:  # noqa: BLE001 - the reason is the point
        pytest.skip(f"trained classifier unavailable: {type(e).__name__}: {e}")
    return clf

