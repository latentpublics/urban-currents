"""Test fixtures.

Every test runs against a throwaway repo root so nothing touches the real
``content/`` or ``runs/``. Tests must pass with no API keys and no network.
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

    importlib.reload(paths)
    config.reset_caches()
    # Modules that captured module-level path constants at import time.
    for name in (
        "pipeline.store",
        "pipeline.metrics",
        "pipeline.llm",
        "pipeline.validate",
        "pipeline.graph.build",
        "pipeline.filters.embed",
    ):
        importlib.reload(importlib.import_module(name))
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
