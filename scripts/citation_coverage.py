"""Q1: can citation structure be used as content at all? (phase 0d)

The gate for the whole citation layer. Abstracts turned out to be closed by
publisher policy, and the question here is whether reference lists repeat that
story or tell a different one. Reference openness has its own history — the
Initiative for Open Citations predates the abstract withdrawals and Elsevier
joined it in 2020, years before it pulled abstracts — so the two axes may not
have the same shape. That is a measurement, not an assumption.

Reports only. Nothing here writes to `content/`.

Usage:
    uv run python scripts/citation_coverage.py --json runs/citation_coverage.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import store  # noqa: E402
from pipeline.calibrate import backfill_dir  # noqa: E402
from pipeline.collectors.abstracts import publisher_of  # noqa: E402
from pipeline.models import Item  # noqa: E402


def _bucket(item: Item) -> str:
    """The population an item belongs to for this table.

    arXiv items are split by whether OpenAlex found them at all: a preprint
    OpenAlex has not indexed has no `referenced_works` for a reason that has
    nothing to do with publisher policy, and pooling the two would blame the
    wrong thing.
    """
    if item.ids.arxiv:
        return "arXiv (OpenAlex ID)" if item.ids.openalex else "arXiv (no OpenAlex ID)"
    return f"journal - {publisher_of(item)}"


def _row(items: list[Item]) -> dict:
    counts = [len(it.graph.referenced_works) for it in items]
    with_refs = [c for c in counts if c]
    has_abstract = sum(1 for it in items if (it.bibliography.abstract or "").strip())
    return {
        "items": len(items),
        "with_references": len(with_refs),
        "share_with_references": round(len(with_refs) / len(items), 4) if items else None,
        "median_references": round(statistics.median(with_refs), 1) if with_refs else 0,
        "zero_reference_share": round(1 - len(with_refs) / len(items), 4) if items else None,
        # The comparison the whole milestone turns on.
        "with_abstract": has_abstract,
        "share_with_abstract": round(has_abstract / len(items), 4) if items else None,
    }


def measure_archive() -> dict:
    by_bucket: dict[str, list[Item]] = {}
    for item in store.iter_items():
        by_bucket.setdefault(_bucket(item), []).append(item)

    rows = {name: _row(items) for name, items in sorted(by_bucket.items())}
    everything = [it for items in by_bucket.values() for it in items]
    journal_only = [
        it for name, items in by_bucket.items() if name.startswith("journal")
        for it in items
    ]
    return {
        "population": "content/items (published + unreadable)",
        "by_bucket": rows,
        "all": _row(everything),
        "journal_only": _row(journal_only),
    }


def measure_backfill() -> dict:
    """The 90-day backfill does not store Items, only scored rows.

    Those rows carry no `referenced_works`, so the backfill cannot answer this
    question without re-fetching. Said plainly rather than estimated.
    """
    scores = backfill_dir() / "scores.jsonl"
    if not scores.exists():
        return {"status": "NO_BACKFILL"}
    first = next(
        (json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines() if line.strip()),
        {},
    )
    return {
        "status": "NOT_MEASURABLE",
        "why": (
            "backfill/scores.jsonl stores scores, not Items: it has no "
            "referenced_works field, so reference coverage over the 90 days "
            "cannot be read from it without re-fetching every Work."
        ),
        "row_fields": sorted(first.keys()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full result here")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    result = {"archive": measure_archive(), "backfill": measure_backfill()}
    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )

    arch = result["archive"]
    print(f"{'bucket':<34} {'n':>5} {'refs':>6} {'med':>5} {'abstr':>7}")
    for name, r in arch["by_bucket"].items():
        print(
            f"{name:<34} {r['items']:>5} "
            f"{(r['share_with_references'] or 0):>5.0%} "
            f"{r['median_references']:>5} "
            f"{(r['share_with_abstract'] or 0):>6.0%}"
        )
    for label in ("journal_only", "all"):
        r = arch[label]
        print(
            f"{label:<34} {r['items']:>5} "
            f"{(r['share_with_references'] or 0):>5.0%} "
            f"{r['median_references']:>5} "
            f"{(r['share_with_abstract'] or 0):>6.0%}"
        )


if __name__ == "__main__":
    main()
