"""What a clean checkout needs to actually run (hotfix H1).

The pipeline worked on the machine that built it and failed on the first CI run,
because the trained classifier was in `.gitignore` as a "build output". It is
4KB of logistic-regression weights, `classify` cannot degrade without it —
`latest_model_path()` raises on a pin it cannot resolve, deliberately — and so
every fresh checkout failed the stage.

**Local success proved nothing about a clean tree.** These tests ask the
question the old ones did not: is what the config points at actually in the
repository?

They run `git ls-files`, not `Path.exists()`. Existence on this disk is exactly
the thing that misled us.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def tracked(path: str) -> bool:
    """Whether git tracks this path. Not whether it is on disk."""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def test_the_pinned_classifier_is_in_the_repository():
    """Change the pin without committing the model and this fails, not CI.

    The whole point of the hotfix: the failure has to happen where someone is
    looking, and a CI job two steps into a turn-on checklist is not that place.
    """
    from pipeline.config import cfg

    pinned = cfg("classifier.model_version")
    assert pinned, "classifier.model_version is unset; nothing pins the model"

    relative = f"models/{pinned}.joblib"
    assert (ROOT / relative).exists(), f"{relative} is missing from disk"
    assert tracked(relative), (
        f"{relative} exists here but git does not track it — a clean checkout "
        f"would fail `classify` with FileNotFoundError, which is what broke CI"
    )


def test_every_model_with_committed_metadata_has_its_weights_committed_too():
    """Metadata without weights describes a model nobody can load."""
    orphans = []
    for meta in sorted((ROOT / "models").glob("*.json")):
        weights = f"models/{meta.stem}.joblib"
        if tracked(f"models/{meta.name}") and not tracked(weights):
            orphans.append(weights)
    assert not orphans, f"training metadata committed without the model: {orphans}"


def test_the_config_does_not_point_at_untracked_vocabulary():
    """The same shape of defect, looked for everywhere it could hide.

    Anything the pipeline reads by name from config has to be in the repository,
    or it works here and fails on a fresh clone in exactly the same way.
    """
    from pipeline.config import cfg

    missing = []
    for key, prefix in (
        ("citation.canon_exclusions_file", "vocab/"),
        ("citation.canon_instrument_file", "vocab/"),
    ):
        name = cfg(key, None)
        if name and not tracked(f"{prefix}{name}"):
            missing.append(f"{key} -> {prefix}{name}")
    assert not missing, f"config points at untracked files: {missing}"


@pytest.mark.parametrize(
    "path",
    [
        "vocab/methods.yaml",
        "vocab/data.yaml",
        "vocab/tools.yaml",
        "vocab/canon_exclude_subfields.yaml",
        "vocab/canon_instrument_topics.yaml",
        "vocab/sources/journals.yaml",
        "config/pipeline.yaml",
        "config/scoring.yaml",
    ],
)
def test_the_files_a_run_reads_are_tracked(path: str):
    """A run on a clean clone reads all of these. None may be a local artefact."""
    assert tracked(path), f"{path} is not tracked; a clean checkout cannot run"


# --------------------------------------------------------------------------
# H2 — a missing key is a skip, not a failure
# --------------------------------------------------------------------------


def test_a_missing_key_is_recorded_as_skipped(repo, monkeypatch):
    """`docs/OPERATIONS.md` promised SKIPPED for eight batches; the code said FAILED.

    The two get investigated in different places — one is "the source is down",
    the other is "nobody put a key in the repository settings" — and YJUN's
    first CI run reported the wrong one.
    """
    from pipeline.collectors.openalex import OpenAlexUnavailable, configure_pyalex
    from pipeline.metrics import Run
    from pipeline.run_stages import StageSkipped, _guard

    monkeypatch.delenv("OPENALEX_KEY", raising=False)
    assert issubclass(OpenAlexUnavailable, StageSkipped)

    run = Run.for_date(__import__("datetime").date(2026, 8, 20))
    _guard(run, "collect.openalex", configure_pyalex)

    assert run.metrics.stages["collect.openalex"] == "SKIPPED"
    assert any("OPENALEX_KEY" in e for e in run.metrics.errors)
    assert not any("FAILED" in e for e in run.metrics.errors)


def test_a_skipped_required_source_still_blocks_the_issue(repo, monkeypatch):
    """The verdict does not move. Only the reason gets more accurate.

    Half the declared scope missing is a different claim whether the source was
    unconfigured or broken, so X3's rule stands either way.
    """
    from datetime import date

    from pipeline.metrics import Run
    from pipeline.outcome import NOT_PUBLISHED, decide

    day = date(2026, 8, 20)
    run = Run.for_date(day)
    run.metrics.stages.update({
        "collect": "OK",
        "collect.arxiv": "OK",
        "collect.openalex": "SKIPPED",
    })

    outcome = decide(run, day, published_count=8)

    assert outcome.status == NOT_PUBLISHED
    assert outcome.writes_issue is False
    assert any("collect.openalex" in r and "SKIPPED" in r for r in outcome.reasons)


def test_an_unavailable_llm_is_also_a_skip(repo):
    """Same shape, swept for rather than fixed one at a time."""
    from pipeline.llm import LLMUnavailable
    from pipeline.skips import StageSkipped

    assert issubclass(LLMUnavailable, StageSkipped)
