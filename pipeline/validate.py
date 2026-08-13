"""Validate everything under ``content/``.

Two checks, both of which the acceptance criteria call out (PRD §9):

- every Item / Issue / Entity file parses against the pydantic model
- ``entities`` contains **zero free strings** — every tag carries a canonical
  ID prefix. The models enforce this on load, so a violation shows up as a
  validation error; the explicit second pass exists so the failure message names
  the offending tag rather than a field path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .models import CANONICAL_PREFIXES, Entity, Issue, Item, work_key_to_filename


@dataclass
class ValidationResult:
    ok: bool = True
    checked: int = 0
    lines: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)
        self.lines.append(f"[FAIL] {msg}")

    def note(self, msg: str) -> None:
        self.lines.append(f"[OK] {msg}")


def _validate_dir(result: ValidationResult, directory: Path, model, label: str) -> list:
    parsed = []
    files = sorted(directory.glob("*.json")) if directory.exists() else []
    for p in files:
        try:
            obj = model.model_validate_json(p.read_text(encoding="utf-8"))
            parsed.append(obj)
        except Exception as e:  # noqa: BLE001
            result.fail(f"{p.relative_to(paths.ROOT)}: {type(e).__name__}: {e}")
        result.checked += 1
    result.note(f"{label}: {len(parsed)}/{len(files)} valid")
    return parsed


def check_entity_ids(result: ValidationResult, items: list[Item]) -> int:
    """Zero free strings in ``entities`` (PRD §9). Every ID must be prefixed."""
    violations = 0
    for item in items:
        e = item.entities
        for facet in ("topics", "people", "orgs", "methods", "data", "tools", "places"):
            for ref in getattr(e, facet):
                if not ref.id.startswith(CANONICAL_PREFIXES):
                    violations += 1
                    result.fail(
                        f"{item.work_key}: entities.{facet} tag {ref.id!r} is a free string"
                    )
    if violations == 0:
        result.note("entities: 0 free strings across all items")
    return violations


def check_issue_references(
    result: ValidationResult, issues: list[Issue], items: list[Item]
) -> None:
    for issue in issues:
        for wk in issue.items:
            if not (paths.ITEMS / work_key_to_filename(wk)).exists():
                result.fail(f"issue {issue.date}: references missing item {wk}")
        if issue.headline.present and issue.headline.work_key not in issue.items:
            result.fail(f"issue {issue.date}: headline item not in items list")

    # Not an error: `store` never deletes, so an item dropped by a changed
    # selection rule stays behind. It is worth counting because the archive is
    # what novelty is measured against — unreferenced items quietly raise the
    # baseline, and a number here is how anyone would notice.
    published = {wk for issue in issues for wk in issue.items}
    orphans = [it.work_key for it in items if it.work_key not in published]
    if orphans:
        result.note(
            f"items: {len(orphans)} not referenced by any issue "
            f"(e.g. {', '.join(sorted(orphans)[:3])})"
        )


def check_edges(result: ValidationResult) -> None:
    p = paths.GRAPH / "edges.jsonl"
    if not p.exists():
        result.note("graph/edges.jsonl: not built")
        return
    bad = 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        try:
            json.loads(line)
        except json.JSONDecodeError:
            bad += 1
    if bad:
        result.fail(f"graph/edges.jsonl: {bad} malformed lines")
    else:
        result.note(f"graph/edges.jsonl: {n} edges valid")


def validate_content() -> ValidationResult:
    result = ValidationResult()
    items = _validate_dir(result, paths.ITEMS, Item, "items")
    issues = _validate_dir(result, paths.ISSUES, Issue, "issues")
    for facet_dir in sorted(paths.ENTITIES.glob("*")) if paths.ENTITIES.exists() else []:
        if facet_dir.is_dir():
            _validate_dir(result, facet_dir, Entity, f"entities/{facet_dir.name}")
    check_entity_ids(result, items)
    check_issue_references(result, issues, items)
    check_edges(result)
    result.lines.append(
        f"{'PASS' if result.ok else 'FAIL'}: {result.checked} files checked, "
        f"{len(result.errors)} errors"
    )
    return result
