"""Derive a paper-level subfield list from our own data (phase 0N, P2).

## What was wrong

`whitelist_subfields` — 3305, 3313, 3322 — was built to choose **journals**.
Using it to judge **articles** is a category error, and N3 measured the price:
gating the journal path on the paper's own subfield loses **27 of 44 keeps
(61.4%)**, and after filtering no day retains 8/10 label coverage, so precision
cannot be reported at all. *Landscape and Urban Planning* is an urban journal
whose articles carry environmental-science, ecology and public-health subfields.

So this does not tune the old list. It derives a **separate** one, from
articles, and leaves the journal-selection list untouched.

## The inclusion rule — written before the numbers were looked at

A subfield is included when **both** hold over our own labelled corpus:

1. **at least `MIN_OBSERVED` labelled papers** carry it, and
2. its **keep rate is at least `MIN_KEEP_RATE`**.

`MIN_OBSERVED = 3` and `MIN_KEEP_RATE = 0.50`. The threshold is the keep rate of
a coin flip: a subfield whose papers we keep more often than not belongs in the
scope, and one below that does not — while a subfield seen once or twice has not
told us anything either way and is **included by default**, because excluding on
absence of evidence is the measured-zero mistake this project keeps making.

## The adoption rule — also pre-registered

The new list is enforced only if **both**:

- **keep loss ≤ 10%** (the old rule's 61.4% is the baseline to beat), and
- **withheld rate ≤ 0.30** over the 20-day archive.

Otherwise it stays recording-only and this prints the table.

Usage:
    uv run python scripts/paper_subfields.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.config import cfg  # noqa: E402
from pipeline.held import paper_subfield  # noqa: E402
from pipeline.labeling import load_labels, superseded  # noqa: E402
from pipeline.models import Item  # noqa: E402

# ★ Raised from 3 in 0Q (R2). Three observations was the bar that excluded four
# subfields, and five targeted judgements per subfield overturned every one of
# them (0P Q3). YJUN, having labelled them: "서브필드에서 몇 편 되지 않더라도
# 내용상 urban에 가까우면 검토대상으로 두어야 할 것 같습니다."
#
# **Fewer than this many observations means the subfield passes. Always.**
# That is stated as its own condition rather than left to fall out of the
# arithmetic, because it is the part of the rule that keeps being violated.
MIN_OBSERVED = 10

# The keep rate a subfield has to clear to be included, unchanged: the keep rate
# of a coin flip.
MIN_KEEP_RATE = 0.50

# But the point estimate is not what excludes any more. **The upper bound of the
# 95% Wilson interval** on the keep rate has to be below `MIN_KEEP_RATE` — that
# is what "clearly low" means, and it is what the old rule was missing.
#
# Worked on the four that were wrongly excluded, all of which the old rule
# removed on a point estimate below 0.50:
#
#   subfield  observed   point   Wilson upper   old rule   new rule
#   1408      1 of 3     0.333   0.792          EXCLUDE    include
#   2208      2 of 5     0.400   0.769          EXCLUDE    include
#   2306      1 of 4     0.250   0.700          EXCLUDE    include
#   3312      1 of 6     0.167   0.564          EXCLUDE    include
#
# **Every one of them, without knowing the outcome.** A subfield seen three
# times and kept once is not a subfield we know anything about, and the interval
# says so where the point estimate did not.
CONFIDENCE_Z = 1.96

MAX_KEEP_LOSS = 0.10
MAX_WITHHELD_RATE = 0.30


def _stage_index() -> dict[str, Item]:
    """Every candidate on disk, so labelled-but-unpublished items are included."""
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


def labelled_rows() -> list[dict]:
    index = _stage_index()
    rows = []
    for row in superseded(load_labels("relevance")):
        key = row["work_key"]
        item = store.load_item(key) or index.get(key)
        if item is None:
            continue
        rows.append({
            "work_key": key,
            "date": row.get("date"),
            "rank": row.get("rank"),
            "source": row.get("source"),
            "keep": row.get("label") == "keep",
            "subfield": paper_subfield(item),
            "title": row.get("title", ""),
        })
    return rows


def wilson_upper(keeps: int, n: int, z: float = CONFIDENCE_Z) -> float:
    """Upper bound of the Wilson score interval on the keep rate.

    Chosen over the textbook normal interval because that one is useless at the
    sizes this rule actually sees — with 1 keep in 3 it produces a bound above
    1.0, and with 0 keeps in 4 it produces zero width, which would have made
    "never kept, four times" look like certainty. Wilson stays inside [0, 1] and
    stays wide when n is small, which is the entire property being bought here.
    """
    if n == 0:
        return 1.0
    phat = keeps / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / denom)


def derive(rows: list[dict]) -> tuple[set[str], list[dict]]:
    """Apply the inclusion rule. A subfield is EXCLUDED only if **both** hold:

    1. it has been observed at least `MIN_OBSERVED` times, and
    2. the **upper** bound of the 95% interval on its keep rate is below
       `MIN_KEEP_RATE`.

    Everything else is included. Both conditions exist to stop the same mistake
    from two directions: (1) refuses to judge a subfield we have barely seen,
    and (2) refuses to treat a low point estimate as a low rate.
    """
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sub[r["subfield"] or "unclassified"].append(r)

    table, included = [], set()
    for sub, items in sorted(by_sub.items(), key=lambda kv: -len(kv[1])):
        keeps = sum(1 for r in items if r["keep"])
        rate = keeps / len(items)
        upper = wilson_upper(keeps, len(items))
        thin = len(items) < MIN_OBSERVED
        # Thin evidence is included: absence of evidence is not evidence of
        # absence, and excluding on it is the mistake this project keeps making.
        confidently_low = upper < MIN_KEEP_RATE
        keep_it = thin or not confidently_low
        if keep_it and sub != "unclassified":
            included.add(sub)
        table.append({
            "subfield": sub,
            "n": len(items),
            "keeps": keeps,
            "keep_rate": round(rate, 4),
            "keep_rate_upper_95": round(upper, 4),
            "thin": thin,
            "confidently_below_half": confidently_low,
            "included": keep_it,
        })
    return included, table


def measure_recall(rows: list[dict], allowed: set[str]) -> dict:
    """Keep loss under the new list — reported first, as N3's lesson demands.

    `allowed` is the inclusion view; the rule that runs is its complement, the
    exclusion list. They agree for every subfield the labels have seen, which is
    every subfield in `rows` by construction. The difference appears only for
    subfields never labelled, and those are exactly what the deny-list exists
    for — see `pipeline.held.rejected_subfield_ids`.
    """
    total_keeps = sum(1 for r in rows if r["keep"])
    lost = [
        r for r in rows
        if r["keep"] and r["subfield"] is not None and r["subfield"] not in allowed
    ]
    return {
        "keeps": total_keeps,
        "keeps_lost": len(lost),
        "keep_loss_rate": round(len(lost) / total_keeps, 4) if total_keeps else None,
        "lost_titles": [f"[{r['subfield']}] {r['title'][:70]}" for r in lost[:10]],
    }


def measure_withheld_rate(allowed: set[str], rejected: set[str]) -> dict:
    """What the new list would have withheld across the backfilled archive."""
    held_dir = paths.CONTENT / "held"
    published = withheld = 0
    per_day = []
    for path in sorted(held_dir.glob("*.json")) if held_dir.exists() else []:
        doc = json.loads(path.read_text(encoding="utf-8"))
        day_pub = doc.get("published", 0)
        # Re-judge the off-subfield flags under the new list.
        day_withheld = 0
        for row in doc.get("items", []):
            if row.get("rule") != "off_subfield":
                if row.get("kind") == "withheld":
                    day_withheld += 1
                continue
            detail = row.get("detail", "")
            sub = None
            for token in detail.replace(",", " ").split():
                if token.isdigit() and len(token) == 4:
                    sub = token
                    break
            # Deny-list semantics: withheld only when the labels have judged
            # this subfield and judged against it. Unseen subfields pass.
            if sub is not None and sub in rejected:
                day_withheld += 1
        published += day_pub
        withheld += day_withheld
        denom = day_pub + day_withheld
        per_day.append({
            "date": doc["date"],
            "published": day_pub,
            "withheld": day_withheld,
            "rate": round(day_withheld / denom, 4) if denom else None,
        })

    denom = published + withheld
    return {
        "days": len(per_day),
        "published": published,
        "withheld": withheld,
        "withheld_rate": round(withheld / denom, 4) if denom else None,
        "days_over_0_30": sum(1 for d in per_day if (d["rate"] or 0) > 0.30),
        "per_day": per_day,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    journal_list = sorted({str(s) for s in (cfg("openalex.whitelist_subfields", []) or [])})
    rows = labelled_rows()
    included, table = derive(rows)

    print(f"labelled papers with a subfield: {len(rows)}")
    print(f"journal-selection list (untouched): {journal_list}")
    print(f"\nrule: EXCLUDE only if n >= {MIN_OBSERVED} AND the 95% upper bound on "
          f"the keep rate is < {MIN_KEEP_RATE}\n")
    print(f"{'subfield':14} {'n':>3} {'keep':>5} {'rate':>6} {'upper95':>8}  "
          f"{'thin':>5}  in")
    for t in table:
        print(
            f"{t['subfield']:14} {t['n']:>3} {t['keeps']:>5} {t['keep_rate']:>6} "
            f"{t['keep_rate_upper_95']:>8}  "
            f"{'yes' if t['thin'] else '-':>5}  {'yes' if t['included'] else 'NO'}"
        )

    recall = measure_recall(rows, included)
    rejected = {t['subfield'] for t in table
                if not t['included'] and t['subfield'] != 'unclassified'}
    withheld = measure_withheld_rate(included, rejected)

    print(f"\nderived list: {len(included)} subfields")
    print("\nrecall — reported first:")
    print(f"  keeps {recall['keeps']}, lost {recall['keeps_lost']} "
          f"({recall['keep_loss_rate']})   baseline to beat: 0.614")
    for t in recall["lost_titles"]:
        print(f"    {t}")

    print(f"\nwithheld rate over {withheld['days']} backfilled days: "
          f"{withheld['withheld_rate']}  (days over 0.30: {withheld['days_over_0_30']})")

    loss_ok = (recall["keep_loss_rate"] or 0) <= MAX_KEEP_LOSS
    rate_ok = (withheld["withheld_rate"] or 0) <= MAX_WITHHELD_RATE
    adopt = bool(loss_ok and rate_ok)

    print(f"\nkeep loss <= {MAX_KEEP_LOSS}: {'PASS' if loss_ok else 'FAIL'}")
    print(f"withheld rate <= {MAX_WITHHELD_RATE}: {'PASS' if rate_ok else 'FAIL'}")
    print(f"\nENFORCE: {adopt}")

    out = {
        "rule": {
            "statement": (
                "exclude only if n >= min_observed AND the upper bound of the "
                "95% Wilson interval on the keep rate is below min_keep_rate; "
                "a subfield with fewer than min_observed labelled papers always "
                "passes"
            ),
            "min_observed": MIN_OBSERVED,
            "min_keep_rate": MIN_KEEP_RATE,
            "confidence_z": CONFIDENCE_Z,
            "max_keep_loss": MAX_KEEP_LOSS,
            "max_withheld_rate": MAX_WITHHELD_RATE,
        },
        "journal_selection_list_untouched": journal_list,
        "derived_paper_subfields": sorted(included),
        "by_subfield": table,
        "recall": recall,
        "withheld": withheld,
        "enforce": adopt,
    }
    target = paths.RUNS / "paper_subfields.json"
    target.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"-> {target}")


if __name__ == "__main__":
    main()
