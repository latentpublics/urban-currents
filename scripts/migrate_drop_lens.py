"""Remove the dead `lens` field from stored items (phase 0k, X0-2).

`Item.lens` was declared as `behavior | system` and no stage ever wrote it: all
224 items in the archive carry `"lens": null`. The card template had a slot for
it that could never render.

A field in the schema is a promise to whoever reads the data that the data
exists. This one could not be kept, and the alternative — having the summariser
fill it — means asking an LLM to sort papers into two categories with no ground
truth in this repo, no label supporting the split, and no consumer asking for
it. So it goes.

`Item` is a strict model, so removing the field makes every stored file
invalid until this has run. That is the reason this script exists rather than a
one-line edit: schema removals need a migration (CLAUDE.md), and the archive is
1,500 files.

**Reversible.** `--revert` puts `"lens": null` back, `--check` reports without
writing.

Usage:
    uv run python scripts/migrate_drop_lens.py --check
    uv run python scripts/migrate_drop_lens.py
    uv run python scripts/migrate_drop_lens.py --revert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Report without writing")
    ap.add_argument("--revert", action="store_true", help="Put `lens: null` back")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    changed = 0
    scanned = 0
    non_null = []
    for path in sorted((paths.CONTENT / "items").glob("*.json")):
        scanned += 1
        raw = path.read_text(encoding="utf-8")
        doc = json.loads(raw)

        if a.revert:
            if "lens" in doc:
                continue
            doc["lens"] = None
        else:
            if "lens" not in doc:
                continue
            # A non-null value would mean something did write it after all, and
            # this migration would be throwing away data rather than a promise.
            if doc["lens"] is not None:
                non_null.append((path.name, doc["lens"]))
                continue
            doc.pop("lens")

        changed += 1
        if not a.check:
            path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )

    verb = "would change" if a.check else ("reverted" if a.revert else "changed")
    print(json.dumps({
        "scanned": scanned,
        verb: changed,
        "non_null_left_alone": non_null,
    }, indent=2))
    if non_null:
        print(
            f"\n{len(non_null)} item(s) carry a non-null lens and were left alone. "
            f"Something wrote them; this migration will not discard that."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
