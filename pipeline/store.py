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


# U+2028 LINE SEPARATOR and U+2029 PARAGRAPH SEPARATOR, by codepoint so the
# source file itself never contains one.
_LS = chr(0x2028)
_PS = chr(0x2029)


def jsonl_line(data: object) -> str:
    """One JSONL record, safe to read back with `splitlines()`.

    `json.dumps(..., ensure_ascii=False)` emits U+2028 LINE SEPARATOR and U+2029
    literally — they are valid inside a JSON string — but Python's
    `str.splitlines()` treats both as line breaks. A paper title containing one
    therefore wrote a record that could never be parsed back, and the file only
    failed when something tried to read it. Escaping them keeps the file both
    human-readable and round-trippable.
    """
    text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return text.replace(_LS, "\\u2028").replace(_PS, "\\u2029")


def read_jsonl(path: Path, on_error: Optional[list] = None) -> list:
    """Parse a JSONL file, collecting unparseable lines rather than dying on one.

    A store that a single bad line makes unreadable is a store that loses
    everything to one interrupted write.
    """
    if not path.exists():
        return []
    out = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            if on_error is not None:
                on_error.append({"line": lineno, "error": str(e)})
    return out


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
        # `detected_at` means "when we first saw this publication state", not
        # "when we last looked". The collector stamps it with now() on every
        # pass, so without this the file is rewritten on every run and the
        # idempotency guarantee fails on a single timestamp.
        if (
            existing.publication_status.state == item.publication_status.state
            and existing.publication_status.detected_at is not None
        ):
            item.publication_status.detected_at = existing.publication_status.detected_at

        a = existing.model_dump(mode="json", by_alias=True)
        b = item.model_dump(mode="json", by_alias=True)
        # `updated` and `collected_at` describe the run, not the paper. Letting
        # them count as changes would rewrite every file on every run and bury
        # real diffs.
        for d in (a, b):
            d.pop("updated", None)
            d.get("provenance", {}).pop("collected_at", None)
        if a == b:
            # Equal *after parsing*, which is not the same as equal on disk. A
            # field with a normalising validator — `Institution.ror` — reads
            # back canonical from a file that stores the old form, so the
            # comparison above can never see the difference and the stale text
            # would survive every future run. Compare what we would write
            # against what is there, carrying over the two fields that describe
            # the run rather than the paper.
            item.updated = existing.updated
            item.provenance.collected_at = existing.provenance.collected_at
            if item.first_published is None:
                item.first_published = existing.first_published
            expected = dumps(item)
            if expected == p.read_text(encoding="utf-8"):
                return False  # genuinely unchanged: leave the file (and mtime) alone
            return write_text_atomic(p, expected)
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
