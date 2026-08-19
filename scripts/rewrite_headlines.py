"""Rewrite the archive's headline lines as titles (phase 0R, T4).

## Should the past 62 issues be rewritten at all?

`Issue` is immutable once published (D127), and that rule exists to protect what
a reader received. **No issue has ever reached a reader.** `deliver.backend` is
`file`, the ledger is empty, and every one of the 62 exists only as a file in
this repository and a page on a site with no domain.

So the choice is between two archives, not between honouring and breaking a
promise:

  * headlines rewritten only from today  → **62 issues in two voices**, and the
    archive is the first thing anyone opens
  * the whole archive rewritten          → one voice, and nothing anybody read
    has changed, because nobody read it

**The worst outcome is the mixed one**, and it is also the one that arrives by
default if nothing is decided. So: rewrite all of them, once, before there is a
reader whose copy would have to be honoured.

**The published item list is never touched.** Only `headline.line` and the new
`headline.basis` move; `items`, `unreadable`, `scan_meta` and everything else
are written back byte-identical.

Usage:
    uv run python scripts/rewrite_headlines.py --dry-run --limit 10
    uv run python scripts/rewrite_headlines.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.llm import LLMClient  # noqa: E402
from pipeline.summarize.headline import write_headline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Show pairs, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="Only the newest N issues")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    files = sorted(paths.ISSUES.glob("*.json"), reverse=True)
    if a.limit:
        files = files[: a.limit]

    client = LLMClient(task="headline")
    print(f"{len(files)} issue(s); model {client.model}; "
          f"prompt {client.prompt_version}; available={client.available()}\n")

    changed = 0
    bases: dict[str, int] = {}
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        head = doc.get("headline") or {}
        key = head.get("work_key")
        if not key:
            continue
        item = store.load_item(key)
        if item is None:
            print(f"{path.stem}  SKIP — headline item not on disk ({key})")
            continue

        old = head.get("line") or ""
        new, basis = write_headline(item, client=client)
        bases[basis.split(":")[0]] = bases.get(basis.split(":")[0], 0) + 1

        print(f"── {path.stem}  [{basis}]")
        print(f"   old: {old[:150]}")
        print(f"   new: {new}")

        if new != old:
            changed += 1
        if a.dry_run:
            continue

        head["line"] = new
        head["basis"] = basis
        doc["headline"] = head
        # Same serialisation the pipeline uses, so a re-run of the day still
        # produces a byte-identical file and the idempotency check holds.
        path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    print(f"\n{changed} of {len(files)} line(s) {'would change' if a.dry_run else 'changed'}")
    print(f"basis: {bases}")


if __name__ == "__main__":
    main()
