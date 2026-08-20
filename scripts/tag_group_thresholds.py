"""How often is there something to group? (phase 0S, U2)

The synthesis paragraph is moving from measured citation links to the day's own
items, so the condition for **silence** has to move with it. The old one —
three connective facts including at least one citation link — is no longer the
right question.

The new one is: **is any controlled-vocabulary tag shared by at least N of
today's items?** If nothing groups, the paragraph would be "today's seven papers
do not much resemble each other", which is the filler sentence 6a wrote and this
project refused.

This measures, over the whole archive, how many days would speak at N = 2, 3, 4
— beside the old condition, so the two are comparable.

Usage:
    uv run python scripts/tag_group_thresholds.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.synthesis import (  # noqa: E402
    PARAGRAPH_MIN_MATERIAL,
    build_facts,
    material_for_paragraph,
)
from pipeline.synthesis import _tags_of  # noqa: E402


def tag_counts(items) -> Counter:
    """How many of today's items carry each controlled-vocabulary tag.

    A tag is counted **once per item** even if the item carries it in two
    facets; the question is how many papers share it, not how many times it
    was written down.
    """
    counts: Counter = Counter()
    for it in items:
        for tag in set(_tags_of(it)):
            counts[tag] += 1
    return counts


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    days = []
    for path in sorted(paths.ISSUES.glob("*.json")):
        issue = store.load_issue(path.stem)
        if issue is None:
            continue
        items = [i for i in (store.load_item(k) for k in issue.items) if i]
        counts = tag_counts(items)
        facts = build_facts(issue.date, items, issue.scan_meta.unreadable_count)
        material = material_for_paragraph(facts)
        links = (
            len(facts["deviations"]["found"])
            + len(facts["anchors"])
            + len(facts["clusters"])
        )
        days.append({
            "date": path.stem,
            "items": len(items),
            "groups_at": {n: sum(1 for c in counts.values() if c >= n) for n in (2, 3, 4)},
            "biggest_group": max(counts.values()) if counts else 0,
            "old_speaks": material >= PARAGRAPH_MIN_MATERIAL and links >= 1,
            "quiet": issue.quiet_day or not items,
        })

    total = len(days)
    # Days that published items. **Not** `not quiet` — a quiet day is one with
    # no headline above the threshold, and it still publishes items, so using
    # that as the denominator produced percentages over 100.
    publishing = [d for d in days if d["items"]]
    print(f"issues: {total}   of which published items: {len(publishing)}\n")

    print(f"{'condition':<44}{'days speaking':>14}{'of published':>14}")
    old = sum(1 for d in days if d["old_speaks"])
    print(f"{'OLD: 3 facts incl. >=1 measured link':<44}{old:>14}{old / len(publishing):>13.0%}")
    for n in (2, 3, 4):
        speaks = sum(1 for d in days if d["groups_at"][n] >= 1)
        print(f"{f'NEW: a tag shared by >= {n} of the day items':<44}"
              f"{speaks:>14}{speaks / len(publishing):>13.0%}")

    print(f"\n{'':<44}{'mean groups':>14}{'max group':>14}")
    for n in (2, 3, 4):
        g = [d["groups_at"][n] for d in days if d["items"]]
        print(f"{f'at >= {n}':<44}{sum(g) / len(g):>14.2f}"
              f"{max(d['biggest_group'] for d in days):>14}")

    # Where the two conditions disagree is the interesting part: days that
    # would have spoken and now would not, and the reverse.
    print()
    print(f"{'stricter variants':<44}{'days speaking':>14}{'of published':>14}")
    for n, need in ((3, 2), (4, 1), (4, 2)):
        speaks = sum(1 for d in days if d["groups_at"][n] >= need)
        print(f"{f'>= {need} group(s) of >= {n} papers':<44}"
              f"{speaks:>14}{speaks / len(publishing):>13.0%}")

    for n in (2, 3, 4):
        gained = [d["date"] for d in days if d["groups_at"][n] >= 1 and not d["old_speaks"]]
        lost = [d["date"] for d in days if d["old_speaks"] and d["groups_at"][n] < 1]
        print(f"\nat >= {n}: {len(gained)} day(s) newly speak, {len(lost)} day(s) fall silent")
        if lost:
            print(f"  now silent: {', '.join(lost[:8])}")

    out = paths.RUNS / "tag_group_thresholds.json"
    out.write_text(json.dumps({"days": days}, indent=2), encoding="utf-8", newline="\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
