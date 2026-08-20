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

# Scratch state: regenerable, or only meaningful within a run. Stays under
# `runs/`, which is gitignored.
STATE = RUNS / "state"

# ★ State that has to outlive the runner (phase 0U, U6).
#
# `runs/` is gitignored and CI keeps only `runs/cache`, so everything in
# `runs/state/` was starting from nothing on every scheduled run. Three things
# in there are **not regenerable**, and each was quietly broken by that:
#
#   `llm_usage.json`               cumulative spend. `llm.max_calls_total` and
#                                  `max_spend_usd` are compared against it, so
#                                  resetting it daily means **the caps can
#                                  never fire** and `uc status` reports $0
#                                  forever.
#   `openalex_enrich_pending.json` the retry queue config describes as "tried
#                                  first the next day". There was no next day.
#   `canon_unresolvable.jsonl`     ids parked after three failed lookups (0T).
#                                  Lose it and the dead ids return to the head
#                                  of the queue and are asked for again daily.
#
# Deliberately **not** moved:
#
#   `runs/cache/`               regenerable by paying for it again, and already
#                               carried between runs by `actions/cache`.
#   `canon_resolved.jsonl`      69 MB and rewritten daily. A commit that size
#                               every day would bury the archive it lives in;
#                               it goes in the workflow cache instead, where
#                               size is cheap and a daily job keeps it warm.
#   `canon_pending.jsonl`       rebuilt from the reference base minus resolved
#                               minus parked on every run.
#   `backfill_issues.json`      a finished one-off's checkpoint.
#   `review_progress.json`      one person's place in a list.
PERSISTENT_STATE = CONTENT / "state"


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def ensure_dirs() -> None:
    for p in (ITEMS, ISSUES, ENTITIES, GRAPH, RUNS, MODELS, LLM_CACHE, LABELS, STATE):
        p.mkdir(parents=True, exist_ok=True)


def persistent_state(name: str) -> Path:
    """A state file that must outlive the runner — see `paths.PERSISTENT_STATE`.

    Reads from the old `runs/state/` location when the new one has nothing yet,
    so a checkout that predates 0U keeps its accumulated spend and its retry
    queue instead of silently starting from zero — which is the exact failure
    this move exists to fix, and it would be poor to cause it once on the way
    past.
    """
    new = PERSISTENT_STATE / name
    if new.exists():
        return new
    legacy = STATE / name
    if legacy.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        try:
            new.write_bytes(legacy.read_bytes())
            return new
        except OSError:
            return legacy
    return new
