"""W9: does bibliographic coupling favour review papers? (phase 0j, measurement only)

Phase 0i's showcase output was 2026-08-11's synthesis paragraph, and the whole
paragraph stands on one paper:

    "Rethinking urban design for health: **a review of** built environment and
    physical activity correlates in high-density Asian cities" — 7 references
    shared with an 2026-08-07 paper.

**That paper was labelled `drop_weak` when this investigation started and YJUN
revised it to `keep` while the batch was running.** The sharper version of the
finding — the synthesis layer built the day's best sentence on a paper the
editor would not have run — is therefore no longer true, and the report says so.
What survives is the structural half, which never depended on that one label. A review has a long
bibliography, and the raw count of shared references grows with bibliography
length. V2 solved exactly this problem for `canon_affinity` by comparing five
normalisations against the labels; coupling never got the same treatment. The
code already knows — `citation.py` says in its own docstring that "5 shared out
of 12 is a stronger signal than 5 out of 80" — and then sorts by raw `shared`
anyway, and `synthesis.clusters` selects on `len(shared)`.

**Nothing here changes a default.** Three ranking criteria are applied to the
same data and the differences are reported. Deciding is a separate act, and the
five days already published cannot change anyway (D127).

Usage:
    uv run python scripts/coupling_bias.py --json runs/coupling_bias.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline import store  # noqa: E402
from pipeline.graph.citation import compute_coupling, load_reference_base  # noqa: E402
from pipeline.labeling import load_labels  # noqa: E402

DAYS = ["2026-08-05", "2026-08-06", "2026-08-07", "2026-08-10", "2026-08-11"]

# A review in the sense that matters here is a *survey of a literature* — the
# genre V1-2 was careful NOT to demote, because this digest covers it. That is a
# different set from `book-review`, and it is the set with the long
# bibliographies.
REVIEW_PHRASES = (
    "a review of", "systematic review", "scientometric review", "literature review",
    "a survey of", "systematic literature review", "meta-analysis", "meta analysis",
    "review and future", "hybrid scientometric", "state of the art review",
    "bibliometric", "an overview of",
)


def is_review(title: str, openalex_type: str | None = None) -> bool:
    """Survey papers, by title phrase or by OpenAlex's own `review` type."""
    low = (title or "").lower()
    if any(p in low for p in REVIEW_PHRASES):
        return True
    return (openalex_type or "").lower() == "review"


def review_index() -> dict[str, bool]:
    """Which works in the reference base are reviews.

    Titles come from `work_index.jsonl`, not from the store. The reference base
    itself carries only work_key, date, published and referenced_works — no
    title at all — and only 24 of any 400 of its records are in
    `content/items/`. Reading titles from the store put the review share of the
    base at 0.0% while the paper that started this investigation was correctly
    detected as a review, which is the kind of contradiction that means the
    denominator is wrong rather than the finding is surprising.
    """
    from pipeline.graph.citation import iter_raw_openalex_works, load_work_index

    types: dict[str, str] = {}
    for work in iter_raw_openalex_works():
        wid = (work.get("id") or "").rsplit("/", 1)[-1]
        if wid:
            types[wid] = work.get("type") or ""

    index = load_work_index()
    out: dict[str, bool] = {}
    for record in load_reference_base():
        key = record["work_key"]
        entry = index.get(key) or {}
        title = entry.get("title") or ""
        if not title:
            item = store.load_item(key)
            title = item.bibliography.title if item else ""
        out[key] = is_review(title, types.get(entry.get("openalex") or ""))
    return out


def criteria(pair: dict) -> dict[str, float]:
    """The three ranking criteria under comparison.

    - `shared` — what the code sorts on today, and the only one a reader can
      read off the page ("shares 7 references").
    - `jaccard` — already computed and already stored, never used to rank.
    - `geometric` — shared over sqrt(len_a * len_b). Between the two: it
      discounts long bibliographies without letting a 5-reference note dominate
      the way a plain share does.
    """
    a, b = pair["a_references"], pair["b_references"]
    return {
        "shared": float(pair["shared"]),
        "jaccard": float(pair["jaccard"]),
        "geometric": round(pair["shared"] / math.sqrt(a * b), 6) if a and b else 0.0,
    }


def anchors_for_day(
    d: str, pairs: list[dict], reviews: dict[str, bool], criterion: str, limit: int = 3
) -> list[dict]:
    """The clusters a day's issue would show, ranked by one criterion.

    Same scope rule as `synthesis.clusters`: exactly one side in today's issue,
    and the partner published earlier. The anchor is today's paper — the one the
    sentence is built on.
    """
    issue = store.load_issue(date.fromisoformat(d))
    if issue is None:
        return []
    todays = set(issue.items)

    candidates = []
    for p in pairs:
        a, b = p["a"], p["b"]
        if (a in todays) == (b in todays):
            continue
        here, there = (a, b) if a in todays else (b, a)
        other = store.load_item(there)
        if not other or not other.first_published or str(other.first_published) >= d:
            continue
        item = store.load_item(here)
        candidates.append({
            "anchor": here,
            "anchor_title": (item.bibliography.title if item else here)[:80],
            "anchor_is_review": reviews.get(here, False),
            "partner": there,
            "partner_date": str(other.first_published),
            **criteria(p),
        })
    candidates.sort(key=lambda r: -r[criterion])
    return candidates[:limit]


def label_index() -> dict[str, str]:
    return {r["work_key"]: r["label"] for r in load_labels("relevance")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    reviews = review_index()
    labels = label_index()
    pairs = compute_coupling()

    review_share_of_base = round(
        sum(1 for v in reviews.values() if v) / len(reviews), 4
    ) if reviews else 0.0

    # 1. How often is a cluster anchor a review, under each criterion?
    per_criterion: dict[str, dict] = {}
    per_day: dict[str, dict] = {}
    for criterion in ("shared", "jaccard", "geometric"):
        anchors: list[dict] = []
        for d in DAYS:
            picks = anchors_for_day(d, pairs, reviews, criterion)
            per_day.setdefault(d, {})[criterion] = picks
            anchors += picks
        labelled = [(x, labels[x["anchor"]]) for x in anchors if x["anchor"] in labels]
        per_criterion[criterion] = {
            "n_clusters_over_5_days": len(anchors),
            "anchors_that_are_reviews": sum(1 for x in anchors if x["anchor_is_review"]),
            "review_share": round(
                sum(1 for x in anchors if x["anchor_is_review"]) / len(anchors), 4
            ) if anchors else None,
            "labelled_anchors": len(labelled),
            "keep_among_labelled": sum(1 for _, lab in labelled if lab == "keep"),
            "keep_rate": round(
                sum(1 for _, lab in labelled if lab == "keep") / len(labelled), 4
            ) if labelled else None,
            "labels": dict(Counter(lab for _, lab in labelled)),
        }

    # 2. The backfill-wide view: every pair in the base, not only the five days.
    top_by = {}
    for criterion in ("shared", "jaccard", "geometric"):
        ranked = sorted(pairs, key=lambda p: -criteria(p)[criterion])[:100]
        involved = [k for p in ranked for k in (p["a"], p["b"])]
        top_by[criterion] = {
            "top_100_pairs": len(ranked),
            "works_involved": len(set(involved)),
            "review_share_of_works": round(
                sum(1 for k in set(involved) if reviews.get(k)) / len(set(involved)), 4
            ) if involved else None,
            "median_reference_length": sorted(
                [p["a_references"] for p in ranked] + [p["b_references"] for p in ranked]
            )[len(ranked)] if ranked else None,
        }

    # 3. The specific case that started this.
    showcase = {
        "work_key": None,
        "title": "Rethinking urban design for health: a review of built environment…",
        "label": None,
    }
    for key, lab in labels.items():
        item = store.load_item(key)
        if item and item.bibliography.title.startswith("Rethinking urban design for health"):
            showcase = {
                "work_key": key,
                "title": item.bibliography.title[:90],
                "label": lab,
                "detected_as_review": reviews.get(key),
                "reference_count": next(
                    (len(r.get("referenced_works") or []) for r in load_reference_base()
                     if r["work_key"] == key), None,
                ),
            }
            break

    out = {
        "population": (
            f"{len(pairs)} coupling pairs over the reference base "
            f"({len(reviews)} works, {review_share_of_base:.1%} of them reviews); "
            f"cluster anchors measured over the 5 prepared days"
        ),
        "review_share_of_reference_base": review_share_of_base,
        "by_criterion": per_criterion,
        "backfill_top_100": top_by,
        "showcase_case": showcase,
        "per_day": per_day,
        "applied": False,
        "note": "no default changed; citation.py still sorts on raw `shared`",
        "recommendation": {
            "proposal": (
                "Select clusters on `geometric` (shared / sqrt(refs_a * refs_b)) "
                "and keep displaying the raw shared count."
            ),
            "why": (
                "Over the 90-day backfill's top 100 pairs, ranking on raw shared "
                "count involves works with a median of 59 references and 4.6% "
                "reviews — 2.4x the 1.9% base rate. Jaccard gives median 41 and "
                "1.7%, below the base rate; geometric gives 40 and 2.3%. The bias "
                "is real and it is a bias towards long bibliographies, which is "
                "what a review has."
            ),
            "counter_argument": (
                "The raw count is the only one of the three a reader can read: "
                "'shares 7 references' is a fact about the world, 'jaccard 0.06' "
                "is a fact about our arithmetic. Splitting them — select on the "
                "normalised value, display the raw count — keeps both, at the "
                "cost of a page whose ordering cannot be derived from what it "
                "shows. That is a real cost and this digest has avoided it "
                "elsewhere."
            ),
            "against_acting_now": (
                "The five labelled days produce 3 clusters and all three criteria "
                "pick the same 3. The difference only appears at backfill scale, "
                "so the labelled evidence for the *choice* is n=3 and the "
                "evidence for the *bias* is the backfill. Those are different "
                "strengths and should not be reported as one."
            ),
            "what_the_labels_say_about_the_anchors": (
                "2 of the 3 cluster anchors over the 5 labelled days are "
                "drop_not_our_kind; the third was drop_weak and YJUN revised it "
                "to keep mid-batch. So the keep rate among anchors is 1 of 3 "
                "under every criterion — all three pick the same 3 clusters at "
                "this scale. n=3 decides nothing either way, and the earlier "
                "reading of this line (that every anchor had been dropped) was "
                "true of an older version of the label file and is not true now."
            ),
        },
    }
    if a.json:
        Path(a.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n",
        )

    print(out["population"])
    print(f"\nreviews are {review_share_of_base:.1%} of the reference base — the baseline")
    print(f"\n{'criterion':<12} {'clusters':>9} {'review anchors':>15} {'share':>7} "
          f"{'labelled':>9} {'keep rate':>10}")
    for name, v in out["by_criterion"].items():
        print(f"{name:<12} {v['n_clusters_over_5_days']:>9} "
              f"{v['anchors_that_are_reviews']:>15} {str(v['review_share']):>7} "
              f"{v['labelled_anchors']:>9} {str(v['keep_rate']):>10}")

    print("\nbackfill, top 100 pairs by each criterion")
    for name, v in out["backfill_top_100"].items():
        print(f"   {name:<12} works {v['works_involved']:>4}  reviews "
              f"{str(v['review_share_of_works']):>7}  median refs "
              f"{v['median_reference_length']}")

    print(f"\nthe showcase case: {showcase.get('label')} — "
          f"{showcase.get('reference_count')} references, "
          f"detected as review: {showcase.get('detected_as_review')}")

    print("\nper-day cluster anchors")
    for d, by in out["per_day"].items():
        for criterion, picks in by.items():
            keys = [f"{'R' if p['anchor_is_review'] else '·'}{p['shared']:.0f}" for p in picks]
            print(f"   {d} {criterion:<10} {keys}")


if __name__ == "__main__":
    main()
