"""Repository layout. Every path in the pipeline resolves through here."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("UC_ROOT", Path(__file__).resolve().parent.parent))

# `UC_CONTENT` redirects only the published archive, leaving `runs/`, the LLM
# cache and the models where they are.
#
# It exists for verification. `verify_phase0.py` runs the whole pipeline against
# live APIs, and doing that against the real archive left a ghost issue behind
# in phase 0h — `content/issues/2026-08-14.json`, a quiet day with no items,
# created by a test rather than by a day's work. Run daily, that accumulates.
# D127 stopped a verification run from *changing* a published issue; this stops
# it from writing one at all, which is the version that does not need a guard.
#
# Deliberately not `UC_ROOT`: pointing that at a sandbox would also hide the
# summary cache, so a verification run would re-pay for every LLM call it makes.
CONTENT = Path(os.environ.get("UC_CONTENT", ROOT / "content"))
ITEMS = CONTENT / "items"
ISSUES = CONTENT / "issues"
ENTITIES = CONTENT / "entities"
GRAPH = CONTENT / "graph"

VOCAB = ROOT / "vocab"
CONFIG = ROOT / "config"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
DOCS = ROOT / "docs"
SCHEMAS = ROOT / "pipeline" / "schemas"

LLM_CACHE = RUNS / "cache"
LABELS = RUNS / "labels"
STATE = RUNS / "state"


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def ensure_dirs() -> None:
    for p in (ITEMS, ISSUES, ENTITIES, GRAPH, RUNS, MODELS, LLM_CACHE, LABELS, STATE):
        p.mkdir(parents=True, exist_ok=True)
