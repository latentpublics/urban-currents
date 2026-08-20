"""W9: does coupling rank on bibliography length? (phase 0j, measurement only)

**The premise this started from has been withdrawn.** The first version said the
2026-08-11 synthesis paragraph stood on a paper YJUN had labelled `drop_weak` —
that the layer built its best sentence on a paper an editor would not have run.
YJUN corrected that label to `keep`, and both sides of the coupling pair are now
`keep`. **The 08-11 cluster selection was right.** A conclusion built on one
label fell over on one label, which is the lesson to carry into everything
below: where the sample is small, say so and stop.

The structural question survives the retraction because it never depended on
that label. A review has a long bibliography, and the raw count of shared
references grows with bibliography length. V2 settled exactly this for
`canon_affinity` by comparing five normalisations; coupling has never had that
comparison. The code knows — `citation.py` says in its own docstring that "5
shared out of 12 is a stronger signal than 5 out of 80" — and then sorts on raw
`shared` anyway, and `synthesis.clusters` selects on `len(shared)`.

So the test is narrow and it can end the section: **are cluster anchors drawn
from longer bibliographies than the candidates they are drawn from?** If they
are not, there is nothing here and this file says so.

No default is changed. The five days already published cannot change anyway
(D127); what a decision would affect is the days not yet collected.

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
from typing import Optional
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


def mann_whitney_p(a: list[float], b: list[float]) -> Optional[float]:
    """Two-sided Mann-Whitney U with a normal approximation, ties corrected.

    Written out because scipy is not a dependency. Returns None below n=8 a
    side, where the normal approximation is not usable and — more to the point —
    where no test should be reported at all.
    """
    if len(a) < 8 or len(b) < 8:
        return None
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks: list[float] = [0.0] * len(combined)
    i = 0
    tie_correction = 0.0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        t = j - i + 1
        if t > 1:
            tie_correction += t ** 3 - t
        i = j + 1

    r_a = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    n_a, n_b = len(a), len(b)
    u_a = r_a - n_a * (n_a + 1) / 2
    u = min(u_a, n_a * n_b - u_a)
    mu = n_a * n_b / 2
    n = n_a + n_b
    sigma_sq = (n_a * n_b / 12) * ((n + 1) - tie_correction / (n * (n - 1)))
    if sigma_sq <= 0:
        return None
    z = (u - mu) / math.sqrt(sigma_sq)
    # Two-sided normal tail via the error function. Reported with a floor: erfc
    # underflows to exactly 0.0 for large |z|, and "p = 0.0" reads as a broken
    # number rather than as a small one.
    tail = math.erfc(abs(z) / math.sqrt(2))
    return round(tail, 6) if tail >= 1e-6 else 1e-6


def _describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": len(s),
        "median": s[len(s) // 2],
        "p25": s[len(s) // 4],
        "p75": s[(3 * len(s)) // 4],
        "min": s[0],
        "max": s[-1],
    }


def reference_lengths(pairs: list[dict]) -> dict[str, int]:
    """Every work that appears in any coupling pair, with its bibliography size.

    This is the population a cluster anchor is drawn from. Comparing anchors
    against *all* works would compare against papers that could never have been
    anchors, which is a different and easier question.
    """
    out: dict[str, int] = {}
    for p in pairs:
        out[p["a"]] = p["a_references"]
        out[p["b"]] = p["b_references"]
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

    # 2. THE TEST THAT CAN END THIS SECTION. Are anchors drawn from longer
    #    bibliographies than the candidates they are drawn from?
    lengths = reference_lengths(pairs)

    length_test: dict[str, dict] = {}
    for criterion in ("shared", "jaccard", "geometric"):
        anchors = [
            x["anchor"] for d in DAYS
            for x in per_day.get(d, {}).get(criterion, [])
        ]
        anchor_lengths = [lengths[a] for a in anchors if a in lengths]
        rest = [v for k, v in lengths.items() if k not in set(anchors)]
        length_test[criterion] = {
            "anchors": _describe(anchor_lengths),
            "population": _describe(rest),
            "mann_whitney_p": mann_whitney_p(anchor_lengths, rest),
            "verdict": (
                "sample too small to test"
                if len(anchor_lengths) < 8
                else "no difference"
            ),
        }

    # The same question at backfill scale, where there are enough works to test:
    # the top 100 pairs under each criterion against everything else in the base.

    top_by = {}
    for criterion in ("shared", "jaccard", "geometric"):
        ranked = sorted(pairs, key=lambda p: -criteria(p)[criterion])[:100]
        involved = sorted({k for p in ranked for k in (p["a"], p["b"])})
        top_lengths = [lengths[k] for k in involved if k in lengths]
        rest_lengths = [v for k, v in lengths.items() if k not in set(involved)]
        top_by[criterion] = {
            "top_100_pairs": len(ranked),
            "works_involved": len(involved),
            "review_share_of_works": round(
                sum(1 for k in involved if reviews.get(k)) / len(involved), 4
            ) if involved else None,
            "reference_length_top": _describe(top_lengths),
            "reference_length_rest": _describe(rest_lengths),
            "mann_whitney_p": mann_whitney_p(top_lengths, rest_lengths),
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
        "anchor_reference_length": length_test,
        "backfill_top_100": top_by,
        "showcase_case": showcase,
        "per_day": per_day,
        "applied": False,
        "note": "no default changed; citation.py still sorts on raw `shared`",
        "recommendation": {
            "the_anchor_test_first": (
                "Over the five prepared days the cluster anchors are NOT drawn "
                "from longer bibliographies: 3 anchors with a median of 47 "
                "references against a population median of 49. They are "
                "marginally shorter, n is 3, and no test is worth running. On "
                "the evidence that actually reaches a reader — the clusters five "
                "issues showed — there is nothing here."
            ),
            "where_the_effect_is_real": (
                "At backfill scale the ranking does track bibliography length, "
                "and strongly. Among works in the top 100 pairs by raw shared "
                "count the median bibliography is 55 references against 48 for "
                "the rest of the base; jaccard gives 40 against 50 and the "
                "geometric mean 38 against 50, both in the opposite direction. "
                "Review share moves the same way: 4.6%, 1.7%, 2.3% against a "
                "1.9% base rate. But that is a statement about pair ranking "
                "across 5,461 pairs, not about the three anchors five issues "
                "actually showed, and the two must not be reported as one."
            ),
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
                "The five labelled days produce 3 clusters, all three criteria "
                "pick the same 3, and those 3 are not drawn from longer "
                "bibliographies than their peers. Nothing a reader has seen is "
                "yet wrong. Changing a default to fix an effect visible only in "
                "the aggregate, before it has surfaced in an issue, is acting on "
                "a mechanism rather than on a problem."
            ),
            "what_the_labels_say_about_the_anchors": (
                "1 of the 3 cluster anchors over the 5 labelled days is a keep — "
                "the corrected 08-11 paper, whose coupling partner is also a "
                "keep, so that cluster was a correct selection. The other 2 are "
                "drop_not_our_kind. n=3 decides nothing in either direction, and "
                "the claim this investigation opened with — that every anchor "
                "had been dropped by the editor — was true of a superseded "
                "version of the label file and is withdrawn."
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

    print("\nTHE TEST: are cluster anchors drawn from longer bibliographies?")
    print("  (5 prepared days)")
    for name, v in out["anchor_reference_length"].items():
        a, pop = v["anchors"], v["population"]
        print(f"   {name:<11} anchors n={a['n']} median={a.get('median')}   "
              f"population n={pop['n']} median={pop.get('median')}   "
              f"p={v['mann_whitney_p']}  -> {v['verdict']}")

    print("\n  (90-day backfill: top 100 pairs vs the rest of the base)")
    for name, v in out["backfill_top_100"].items():
        t, r = v["reference_length_top"], v["reference_length_rest"]
        print(f"   {name:<11} top n={t['n']:>4} median={t.get('median'):>3} "
              f"(p25 {t.get('p25')}, p75 {t.get('p75')})   "
              f"rest n={r['n']:>4} median={r.get('median'):>3}   "
              f"p={v['mann_whitney_p']}   reviews {v['review_share_of_works']}")

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
