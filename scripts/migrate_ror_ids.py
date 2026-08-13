"""Rewrite `ror:https://ror.org/02mhbdp94` as `ror:02mhbdp94` (phase 0d, Q0-1).

The prefix announces the scheme and the value then repeated it. Every other
canonical prefix in this schema carries a bare identifier — `orcid:0000-…`,
`openalex:W123`, `wikidata:Q60` — and the URL form also produced entity
filenames reading `https___ror.org_02mhbdp94.json`.

Doing it now because identifier meaning is the one thing that gets more
expensive to change every day: the archive is 1,500 files and grows daily, and
every consumer written against the old form is another thing to fix later.

**Reversible.** `--revert` puts the URL form back, and `--check` reports without
writing. The rewrite touches three places — `entities.orgs[].id`,
`bibliography.authors[].institutions[].ror`, and the entity node files — and
`edges.jsonl` is a build output, so it is regenerated rather than edited
(`uc graph`).

Usage:
    uv run python scripts/migrate_ror_ids.py --check
    uv run python scripts/migrate_ror_ids.py
    uv run python scripts/migrate_ror_ids.py --revert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.models import Entity, Item  # noqa: E402

URL_PREFIX = "https://ror.org/"


def to_bare(value: str | None) -> str | None:
    if not value:
        return value
    if value.startswith("ror:" + URL_PREFIX):
        return "ror:" + value[len("ror:" + URL_PREFIX) :]
    if value.startswith(URL_PREFIX):
        return value[len(URL_PREFIX) :]
    return value


def to_url(value: str | None) -> str | None:
    if not value:
        return value
    if value.startswith("ror:") and not value.startswith("ror:" + URL_PREFIX):
        return "ror:" + URL_PREFIX + value[len("ror:") :]
    return value


def migrate(revert: bool = False, check: bool = False) -> dict[str, int]:
    convert = to_url if revert else to_bare
    stats = {"items": 0, "org_refs": 0, "institutions": 0, "entities": 0, "removed": 0}

    for item in list(store.iter_items()):
        touched = False
        for ref in item.entities.orgs:
            new = convert(ref.id)
            if new != ref.id:
                ref.id, touched = new, True
                stats["org_refs"] += 1
        for author in item.bibliography.authors:
            for inst in author.institutions:
                new = convert(inst.ror)
                if new != inst.ror:
                    inst.ror, touched = new, True
                    stats["institutions"] += 1
        if touched:
            stats["items"] += 1
            if not check:
                _write_item(item)

    # Entity nodes are keyed by ID *and* by filename, so a rename is a new file
    # plus a delete — not an edit in place.
    org_dir = paths.ENTITIES / "orgs"
    if org_dir.exists():
        for p in sorted(org_dir.glob("*.json")):
            node = Entity.model_validate_json(p.read_text(encoding="utf-8"))
            new_id = convert(node.id)
            if new_id == node.id:
                continue
            stats["entities"] += 1
            if check:
                continue
            node.id = new_id
            store.save_entity(node)
            if store.entity_path("orgs", new_id) != p:
                p.unlink()
                stats["removed"] += 1

    return stats


def _write_item(item: Item) -> None:
    """Write without `save_item`'s `updated` bookkeeping.

    A migration is not new information about the paper, so advancing `updated`
    would misdate every record in the archive.
    """
    store.write_text_atomic(store.item_path(item.work_key), store.dumps(item))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help="put the URL form back")
    ap.add_argument("--check", action="store_true", help="report without writing")
    a = ap.parse_args()

    # Windows consoles default to a legacy code page, which turns a dash in a
    # status line into a crash after the migration has already been written.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    stats = migrate(revert=a.revert, check=a.check)
    verb = "would change" if a.check else ("reverted" if a.revert else "migrated")
    print(f"{verb}: {json.dumps(stats, sort_keys=True)}")
    if not a.check:
        print("now run `uv run uc graph`: edges.jsonl is a build output")


if __name__ == "__main__":
    main()
