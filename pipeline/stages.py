"""Stage plumbing.

Every stage reads one JSONL file of Items from the run directory and writes
another. That is what makes ``uc <stage> --date …`` independently runnable
(PRD §5) — re-running summarize must not require re-collecting.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Iterable

from .metrics import Run
from .models import Item

STAGE_ORDER = [
    "collect",
    "dedup",
    "gate",
    # Before `classify`, not after: the classifier reads title + abstract, and
    # the journal ranking sends an item with no abstract to the back. An
    # abstract recovered after either of those has already been judged without
    # it. Only gate survivors are enriched, so the request count is small.
    "enrich",
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


def read_stage(run: Run, stage: str, old_schema: bool = False) -> list[Item]:
    """Items a stage wrote.

    `old_schema` is for readers that walk **historic** runs. `Item` forbids
    extra fields, so a stage file written before a schema change no longer
    validates: 60 run directories still carry the `lens` field 0k removed, and
    one of them was enough to kill the whole citation base with a
    `ValidationError`.

    Strictness is right inside a run — a file this stage just wrote failing to
    parse means something is badly wrong. It is wrong for a chore that reads
    every run the repository has ever made.

    **Recovered, not skipped.** The first version of this simply dropped
    unreadable lines, and measured across the archive that was 866 items
    carrying **7,001 references** — a hole in the very base this batch exists to
    finish filling. A field that was *removed* from the schema is not corrupt
    data; it is data with one key too many. So an unknown top-level key is
    dropped and the item re-validated, and only something that still fails is
    given up on.

    Off by default so nothing silently loosens.
    """
    p = stage_path(run, stage)
    if not p.exists():
        return []
    out = []
    recovered = lost = 0
    known = set(Item.model_fields)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Item.model_validate_json(line))
            continue
        except Exception:  # noqa: BLE001
            if not old_schema:
                raise
        try:
            doc = json.loads(line)
            out.append(Item.model_validate({k: v for k, v in doc.items() if k in known}))
            recovered += 1
        except Exception:  # noqa: BLE001
            lost += 1
    if recovered or lost:
        print(
            f"{p.name}: recovered {recovered} item(s) from an older schema"
            + (f", lost {lost}" if lost else "")
        )
    return out



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
