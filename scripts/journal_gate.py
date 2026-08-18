"""Does the paper's own topic gate the journal path? (phase 0L, N3)

arXiv has to clear a classifier probability. **The journal path has no gate at
all** — membership of `journals.yaml` is the whole test — and the labels show
the consequence: journal drops are `not_our_kind` 18 against `not_urban` 3,
where arXiv's are the other way round. The journal path's error is a
*scope-definition* error, and it is not the kind more classifier will fix.

**The cheapest hypothesis first, before any ML.** We select journals by the
journal's subfield and then ask nothing of the article. But the seven strays in
08-11's top fifteen — car insurance, V2G bus economics, Vietnamese green
logistics, a shipping-lane CBA, nano-TiO2 pore structure, asphalt ageing, Arctic
route cost — are nearly all identifiable from the paper's **own**
`primary_topic.subfield`, a field we have collected since phase 0 and never
read.

## What this reports, and what it must not do

Precision alone converges on publishing nothing, so **recall loss is reported
beside it every time**. A filter that lifts precision@10 to 0.9 by dropping
half the keeps is not an improvement, it is a smaller product.

It also does **not** tune anything to the seven known strays. Fitting a rule to
cases whose answer you already know is what the materials-keyword expansion was
refused for in 0i. The seven are reported as a check on a rule chosen for
independent reasons, never as its objective.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.config import cfg  # noqa: E402
from pipeline.held import paper_subfield  # noqa: E402
from pipeline.labeling import load_labels, superseded  # noqa: E402
from pipeline.models import Item  # noqa: E402

KEEP = "keep"


def _index_stage_items() -> dict[str, Item]:
    """Every candidate this project has on disk, by work_key.

    Labels are drawn from the candidate pool, so a labelled item may never have
    been published and will not be in `content/items/`. Falling back to the run
    stage files keeps the population the labels were actually drawn from.
    """
    index: dict[str, Item] = {}
    for run_dir in sorted(paths.RUNS.glob("run_*")):
        for name in ("classify.jsonl", "labeling_pool.jsonl", "select.jsonl"):
            path = run_dir / "stages" / name
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = Item.model_validate_json(line)
                except Exception:  # noqa: BLE001
                    continue
                index.setdefault(item.work_key, item)
    return index


def load_rows() -> list[dict]:
    """Journal labels joined to the paper's own subfield and topic."""
    stage_index = _index_stage_items()
    rows = []
    for label_row in superseded(load_labels("relevance")):
        if label_row.get("source") != "journal":
            continue
        key = label_row["work_key"]
        item = store.load_item(key) or stage_index.get(key)
        subfield = paper_subfield(item) if item else None
        topic = None
        if item and item.entities.topics:
            primary = next(
                (t for t in item.entities.topics if getattr(t, "is_primary", False)),
                item.entities.topics[0],
            )
            topic = primary.label
        rows.append({
            "work_key": key,
            "date": label_row.get("date"),
            "rank": label_row.get("rank"),
            "label": label_row.get("label"),
            "keep": label_row.get("label") == KEEP,
            "title": label_row.get("title", ""),
            "subfield": subfield,
            "topic": topic,
            "found": item is not None,
        })
    return rows


def precision_at_k(rows: list[dict], k: int = 10, keep_keys=None) -> dict:
    """precision@k per day over the labelled journal ranking.

    `keep_keys` filters the ranking first — that is what applying a gate means.
    Days are reported with their **label coverage**, because a day whose top-k
    is only half labelled cannot produce a precision figure, only a guess.
    """
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if keep_keys is not None and r["work_key"] not in keep_keys:
            continue
        by_day[r["date"]].append(r)

    per_day = {}
    for day, items in by_day.items():
        ranked = sorted(items, key=lambda r: r.get("rank") or 999)[:k]
        if not ranked:
            continue
        per_day[day] = {
            "n_labelled_in_top_k": len(ranked),
            "keeps": sum(1 for r in ranked if r["keep"]),
            "precision": round(sum(1 for r in ranked if r["keep"]) / len(ranked), 4),
        }

    usable = [d for d, v in per_day.items() if v["n_labelled_in_top_k"] >= 8]
    mean = (
        round(sum(per_day[d]["precision"] for d in usable) / len(usable), 4)
        if usable
        else None
    )
    return {
        "per_day": per_day,
        "days_with_coverage_8_of_10": len(usable),
        "days_total": len(per_day),
        "mean_precision_over_usable_days": mean,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    wanted = {str(s) for s in (cfg("openalex.whitelist_subfields", []) or [])}
    rows = load_rows()
    found = [r for r in rows if r["found"]]

    print(f"journal labels: {len(rows)}  (item on disk for {len(found)})")
    print(f"whitelist subfields: {sorted(wanted)}\n")

    # Keep rate by the paper's own subfield.
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for r in found:
        by_sub[r["subfield"] or "unclassified"].append(r)

    table = []
    for sub, items in sorted(by_sub.items(), key=lambda kv: -len(kv[1])):
        keeps = sum(1 for r in items if r["keep"])
        table.append({
            "subfield": sub,
            "in_whitelist": sub in wanted,
            "n": len(items),
            "keeps": keeps,
            "keep_rate": round(keeps / len(items), 4),
        })

    print(f"{'subfield':14} {'wl':4} {'n':>3} {'keep':>5} {'rate':>6}")
    for t in table:
        print(
            f"{t['subfield']:14} {'yes' if t['in_whitelist'] else '-':4} "
            f"{t['n']:>3} {t['keeps']:>5} {t['keep_rate']:>6}"
        )

    # The gate: keep only papers whose own subfield is in the whitelist.
    # `unclassified` is KEPT — "we could not check" is not "it failed".
    passing = {
        r["work_key"]
        for r in found
        if r["subfield"] is None or r["subfield"] in wanted
    }
    # Items with no record at all also pass: excluding them would be measuring
    # our own storage rather than the rule.
    passing |= {r["work_key"] for r in rows if not r["found"]}

    before = precision_at_k(rows)
    after = precision_at_k(rows, keep_keys=passing)

    total_keeps = sum(1 for r in rows if r["keep"])
    kept_keeps = sum(1 for r in rows if r["keep"] and r["work_key"] in passing)
    lost_keeps = total_keeps - kept_keeps
    total_drops = sum(1 for r in rows if not r["keep"])
    dropped_drops = total_drops - sum(
        1 for r in rows if not r["keep"] and r["work_key"] in passing
    )

    print(f"\nprecision@10 (journal path), days with >= 8/10 labelled:")
    print(f"  before  {before['mean_precision_over_usable_days']} "
          f"over {before['days_with_coverage_8_of_10']}/{before['days_total']} days")
    print(f"  after   {after['mean_precision_over_usable_days']} "
          f"over {after['days_with_coverage_8_of_10']}/{after['days_total']} days")

    print(f"\nrecall loss — the number precision alone hides:")
    print(f"  keeps  {total_keeps} -> {kept_keeps}   (lost {lost_keeps}, "
          f"{round(100 * lost_keeps / total_keeps, 1) if total_keeps else 0}%)")
    print(f"  drops  {total_drops} -> {total_drops - dropped_drops}   "
          f"(removed {dropped_drops})")

    lost = [r for r in rows if r["keep"] and r["work_key"] not in passing]
    if lost:
        print("\n  keeps this rule would have withheld:")
        for r in lost:
            print(f"    [{r['subfield']}] {r['title'][:72]}")

    result = {
        "whitelist_subfields": sorted(wanted),
        "n_journal_labels": len(rows),
        "n_with_item": len(found),
        "by_subfield": table,
        "precision_before": before,
        "precision_after": after,
        "keeps_before": total_keeps,
        "keeps_after": kept_keeps,
        "keeps_lost": lost_keeps,
        "drops_removed": dropped_drops,
        "keeps_lost_detail": [
            {"subfield": r["subfield"], "title": r["title"], "date": r["date"]}
            for r in lost
        ],
    }
    out = paths.RUNS / "journal_gate.json"
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
