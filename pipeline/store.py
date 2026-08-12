"""Content store — the only place that reads and writes ``content/``.

Two invariants matter here:

1. **Deterministic serialisation.** Same logical content → byte-identical file.
   Running the pipeline twice for the same date must not churn git (PRD §9
   idempotency). That means sorted keys, fixed indent, trailing newline, and
   ``updated`` only advancing when something actually changed.
2. **Atomic writes.** Write to a temp file and replace, so an interrupted run
   never leaves a half-written Item behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel

from . import paths
from .models import Entity, Issue, Item, work_key_to_filename


def dumps(model: BaseModel) -> str:
    """Canonical JSON for a model: mode='json', sorted keys, 2-space indent."""
    data = model.model_dump(mode="json", by_alias=True, exclude_none=False)
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_text_atomic(path: Path, text: str) -> bool:
    """Write ``text`` to ``path``. Returns True if the file changed on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return True


def write_json_atomic(path: Path, data: object) -> bool:
    return write_text_atomic(
        path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


# --------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------


def item_path(work_key: str) -> Path:
    return paths.ITEMS / work_key_to_filename(work_key)


def load_item(work_key: str) -> Optional[Item]:
    p = item_path(work_key)
    if not p.exists():
        return None
    return Item.model_validate_json(p.read_text(encoding="utf-8"))


def save_item(item: Item, today: Optional[date] = None) -> bool:
    """Persist an Item. ``updated`` advances only if the rest of it changed."""
    p = item_path(item.work_key)
    existing = None
    if p.exists():
        try:
            existing = Item.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:  # corrupt file → overwrite rather than crash the run
            existing = None

    if existing is not None:
        a = existing.model_dump(mode="json", by_alias=True)
        b = item.model_dump(mode="json", by_alias=True)
        a.pop("updated", None)
        b.pop("updated", None)
        if a == b:
            return False  # genuinely unchanged: leave the file (and mtime) alone
        item.updated = today or item.updated or existing.updated
        if item.first_published is None:
            item.first_published = existing.first_published
    else:
        item.first_published = item.first_published or today
        item.updated = item.updated or today

    return write_text_atomic(p, dumps(item))


def iter_items() -> Iterator[Item]:
    for p in sorted(paths.ITEMS.glob("*.json")):
        yield Item.model_validate_json(p.read_text(encoding="utf-8"))


def all_item_files() -> list[Path]:
    return sorted(paths.ITEMS.glob("*.json"))


# --------------------------------------------------------------------------
# Issues
# --------------------------------------------------------------------------


def issue_path(d: date | str) -> Path:
    return paths.ISSUES / f"{d}.json"


def load_issue(d: date | str) -> Optional[Issue]:
    p = issue_path(d)
    if not p.exists():
        return None
    return Issue.model_validate_json(p.read_text(encoding="utf-8"))


def save_issue(issue: Issue) -> bool:
    return write_text_atomic(issue_path(issue.date), dumps(issue))


def published_index() -> dict[str, str]:
    """``work_key`` → the date of the earliest Issue that carried it.

    Used to tell "this paper already headlined months ago" (a status change)
    from "this is today's issue being re-run" (idempotent republish). Keying off
    Item existence alone would make the second run of a day publish nothing.
    """
    index: dict[str, str] = {}
    for issue in iter_issues():
        d = str(issue.date)
        for wk in issue.items:
            if wk not in index or d < index[wk]:
                index[wk] = d
    return index


def iter_issues() -> Iterator[Issue]:
    for p in sorted(paths.ISSUES.glob("*.json")):
        yield Issue.model_validate_json(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------


def entity_filename(entity_id: str) -> str:
    return entity_id.split(":", 1)[1].replace("/", "_").replace(":", "_") + ".json"


def entity_path(facet: str, entity_id: str) -> Path:
    return paths.ENTITIES / facet / entity_filename(entity_id)


def load_entity(facet: str, entity_id: str) -> Optional[Entity]:
    p = entity_path(facet, entity_id)
    if not p.exists():
        return None
    return Entity.model_validate_json(p.read_text(encoding="utf-8"))


def save_entity(entity: Entity) -> bool:
    return write_text_atomic(entity_path(entity.facet, entity.id), dumps(entity))


def iter_entities() -> Iterator[Entity]:
    for p in sorted(paths.ENTITIES.glob("*/*.json")):
        yield Entity.model_validate_json(p.read_text(encoding="utf-8"))
