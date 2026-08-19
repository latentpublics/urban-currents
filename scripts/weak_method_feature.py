"""What does `drop_weak_method` change for the journal gate? (phase 0P, Q2)

M1 split one label into two because they teach different things:

  `drop_weak_method`   the method is thin, and an abstract shows it
  `drop_weak_results`  the results are thin, and an abstract mostly does not

Ten of the fifteen re-judged rows came back `method`. This measures what that
buys, and it **only measures** — nothing here changes a default.

Two questions, kept apart:

1. **Does the subfield gate already catch them?** If the six weak-method
   journal papers are in subfields the gate rejects anyway, the new label adds
   nothing the gate can act on.
2. **Does the evaluation change when the unlearnable rows leave it?**
   `drop_weak_results` cannot be predicted from an abstract, so scoring a gate
   on its ability to reject those rows measures noise. Removing them from the
   denominator is not the same as counting them as keeps — they leave.

The numbers are small (75 journal labels; 6 method, 4 results). Reported with
their n every time, because at this size a single row moves the third decimal.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from journal_gate import _index_stage_items, precision_at_k  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.config import cfg  # noqa: E402
from pipeline.held import paper_subfield, rejected_subfield_ids  # noqa: E402
from pipeline.labeling import load_labels, superseded  # noqa: E402

LEARNABLE_NEGATIVE = {"drop_not_urban", "drop_not_our_kind", "drop_weak_method"}
UNLEARNABLE = {"drop_weak_results"}


def rows_for(source: str) -> list[dict]:
    index = _index_stage_items()
    out = []
    for r in superseded(load_labels("relevance")):
        if r.get("source") != source:
            continue
        item = index.get(r["work_key"])
        out.append({
            "work_key": r["work_key"],
            "date": r.get("date"),
            "rank": r.get("rank"),
            "label": r["label"],
            "keep": r["label"] == "keep",
            "title": r.get("title", ""),
            "subfield": paper_subfield(item) if item else None,
            "found": item is not None,
        })
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rejected = rejected_subfield_ids()
    print(f"subfields the gate rejects: {sorted(rejected)}\n")

    for source in ("journal", "arxiv"):
        rows = rows_for(source)
        counts = Counter(r["label"] for r in rows)
        print(f"=== {source} path, n={len(rows)} ===")
        print(f"  {dict(counts)}")

        # --- 1. does the gate already catch the weak-method papers? --------
        method = [r for r in rows if r["label"] == "drop_weak_method"]
        results = [r for r in rows if r["label"] == "drop_weak_results"]
        caught = [r for r in method if r["subfield"] in rejected]
        print(f"  drop_weak_method: {len(method)}  "
              f"already rejected by the subfield gate: {len(caught)}")
        for r in method:
            mark = "gate rejects" if r["subfield"] in rejected else "gate passes"
            print(f"    [{r['subfield'] or 'unclassified':>12}] {mark}  {r['title'][:52]}")

        # Where do weak-method papers sit relative to keeps? If they share the
        # keeps' subfields, no subfield rule can separate them — which is the
        # answer, not a failure to find one.
        keep_subs = Counter(r["subfield"] for r in rows if r["keep"])
        shared = sum(1 for r in method if keep_subs.get(r["subfield"], 0) > 0)
        print(f"  weak-method papers sharing a subfield with a keep: "
              f"{shared}/{len(method)}  -> no subfield rule separates those")

        # --- 2. what changes when the unlearnable rows leave? --------------
        before = precision_at_k(rows)
        trimmed = [r for r in rows if r["label"] not in UNLEARNABLE]
        after = precision_at_k(trimmed)
        print(f"  precision@10 with every drop counted:  "
              f"{before['mean_precision_over_usable_days']} "
              f"({before['days_with_coverage_8_of_10']}/{before['days_total']} days)")
        print(f"  precision@10 with drop_weak_results removed: "
              f"{after['mean_precision_over_usable_days']} "
              f"({after['days_with_coverage_8_of_10']}/{after['days_total']} days) "
              f"[n {len(rows)} -> {len(trimmed)}]")
        print(f"    removing {len(results)} unlearnable row(s) is a change of "
              f"{'n/a' if after['mean_precision_over_usable_days'] is None or before['mean_precision_over_usable_days'] is None else round(after['mean_precision_over_usable_days'] - before['mean_precision_over_usable_days'], 4):+}"
              if isinstance(after['mean_precision_over_usable_days'], float)
              and isinstance(before['mean_precision_over_usable_days'], float)
              else f"    removing {len(results)} unlearnable row(s): not measurable")
        print()

    out = paths.RUNS / "weak_method_feature.json"
    out.write_text(json.dumps({
        "note": "measurement only (0P Q2); no default was changed",
        "learnable_negative": sorted(LEARNABLE_NEGATIVE),
        "excluded_as_unlearnable": sorted(UNLEARNABLE),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
