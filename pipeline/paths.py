"""Repository layout. Every path in the pipeline resolves through here."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("UC_ROOT", Path(__file__).resolve().parent.parent))

CONTENT = ROOT / "content"
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
