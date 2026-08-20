"""Regenerate the synthesis paragraph across the archive (phase 0S, U2).

The paragraph's **material changed** — from the measured citation graph to the
day's own items and the controlled-vocabulary tags they share — so every stored
paragraph answers a question that is no longer being asked. Leaving half the
archive on the old material would be the mixed state 0R already argued against
for headlines, and for the same reason: no issue has ever reached a reader, so
there is no copy to honour.

**Only `synthesis` moves.** `items`, `unreadable`, `headline` and `scan_meta`
are written back unchanged.

Usage:
    uv run python scripts/rewrite_synthesis.py --dry-run --limit 5
    uv run python scripts/rewrite_synthesis.py
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
from pipeline.synthesis import build_facts, write_paragraph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    files = sorted(paths.ISSUES.glob("*.json"), reverse=True)
    if a.limit:
        files = files[: a.limit]

    client = LLMClient(task="synthesis")
    print(f"{len(files)} issue(s); prompt {client.prompt_version}; "
          f"available={client.available()}\n")

    spoke = silent = 0
    for path in files:
        issue = store.load_issue(path.stem)
        if issue is None:
            continue
        items = [i for i in (store.load_item(k) for k in issue.items) if i]
        facts = build_facts(issue.date, items, issue.scan_meta.unreadable_count)
        result = write_paragraph(facts, client=client)

        doc = json.loads(path.read_text(encoding="utf-8"))
        syn = doc.get("synthesis") or {}
        old = syn.get("paragraph")

        print(f"── {path.stem}  {len(items)} items, "
              f"{len(facts['tag_groups'])} group(s)")
        print(f"   old: {(old or '(none)')[:150]}")
        print(f"   new: {(result['text'] or '(omitted: ' + str(result['reason']) + ')')[:150]}")

        if result["text"]:
            spoke += 1
        else:
            silent += 1

        if a.dry_run:
            continue

        syn["paragraph"] = result["text"]
        syn["paragraph_omitted_reason"] = result["reason"]
        doc["synthesis"] = syn
        path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    print(f"\n{spoke} issue(s) speak, {silent} silent "
          f"({'dry run' if a.dry_run else 'written'})")


if __name__ == "__main__":
    main()
