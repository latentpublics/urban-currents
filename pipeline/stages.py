"""Stage plumbing.

Every stage reads one JSONL file of Items from the run directory and writes
another. That is what makes ``uc <stage> --date …`` independently runnable
(PRD §5) — re-running summarize must not require re-collecting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .metrics import Run
from .models import Item

STAGE_ORDER = [
    "collect",
    "dedup",
    "gate",
    "classify",
    "select",
    "link",
    "summarize",
    "score",
    "issue",
    "preview",
]


def stage_path(run: Run, stage: str) -> Path:
    d = run.dir / "stages"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stage}.jsonl"


def write_stage(run: Run, stage: str, items: Iterable[Item]) -> int:
    """Items are written sorted by work_key so the file is byte-stable."""
    rows = sorted(items, key=lambda it: it.work_key)
    lines = [it.model_dump_json(by_alias=True) for it in rows]
    text = "\n".join(lines) + ("\n" if lines else "")
    p = stage_path(run, stage)
    p.write_text(text, encoding="utf-8", newline="\n")
    return len(rows)


def read_stage(run: Run, stage: str) -> list[Item]:
    p = stage_path(run, stage)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(Item.model_validate_json(line))
    return out


def previous_stage(stage: str) -> str:
    i = STAGE_ORDER.index(stage)
    return STAGE_ORDER[i - 1]


def read_input(run: Run, stage: str) -> list[Item]:
    """Read the output of whichever earlier stage most recently produced data.

    Lets a stage run even when an optional middle stage was skipped (a missing
    API key must not stop the pipeline).
    """
    idx = STAGE_ORDER.index(stage)
    for prev in reversed(STAGE_ORDER[:idx]):
        items = read_stage(run, prev)
        if items or stage_path(run, prev).exists():
            return items
    return []
