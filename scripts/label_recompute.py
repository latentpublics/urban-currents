"""W8: every label-derived figure, recomputed at n=148 (phase 0j).

Phase 0i chose an arXiv publication floor and reported a Q1b verdict from 90
labels over three days. YJUN has since labelled the remaining two, so all five
prepared days are judged and every one of those numbers has to be recomputed
rather than carried forward.

Two of them move in opposite directions, which is the reason this is a script
and not a paragraph:

- **Q1b's journal answer changes.** 0.733 over three days becomes 0.640 over
  five, which is below the 0.70 bar. It was "passing but volatile"; it is now
  not passing.
- **The 0.80 floor gets stronger.** 9 of 9 becomes 13 of 13, and the exact
  binomial lower bound moves from 0.717 to 0.794.

Nothing here changes configuration. `selection`, `classifier.threshold` and the
candidate ranking are untouched — this recomputes the evidence they were chosen
on, which is a different act from choosing again.

Usage:
    uv run python scripts/label_recompute.py --json runs/label_recompute.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.labeling import load_labels  # noqa: E402

BANDS = ((0.0, 0.50), (0.50, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01))
FLOORS = (0.35, 0.50, 0.70, 0.80, 0.90)


def exact_lower_bound(n: int, confidence: float = 0.95) -> float:
    """One-sided 95% lower bound on a proportion when every trial succeeded.

    `1 - 3/n` is the rule-of-three approximation and it is conservative at small
    n in the wrong direction — it understates. The exact Clopper-Pearson bound
    for k = n is `alpha^(1/n)`, which at n=13 gives 0.794 against the rule of
    three's 0.769. Both are reported because the approximation is the one
    quoted in the previous batch.
    """
    return round((1 - confidence) ** (1 / n), 4) if n else 0.0


def q1b(rows: list[dict], k: int = 10) -> dict:
    out: dict[str, dict] = {}
    for source in sorted({r["source"] for r in rows}):
        srows = [r for r in rows if r["source"] == source]
        by_day: dict[str, list[dict]] = defaultdict(list)
        for r in srows:
            by_day[r["date"]].append(r)
        per_day = {}
        for day, drows in sorted(by_day.items()):
            drows.sort(key=lambda r: r.get("rank", 0))
            top = drows[:k]
            per_day[day] = round(sum(1 for r in top if r["label"] == "keep") / len(top), 4)
        out[source] = {
            "n_labels": len(srows),
            "days": len(by_day),
            "precision_at_10": round(statistics.fmean(per_day.values()), 4),
            "per_day": per_day,
            "spread": round(max(per_day.values()) - min(per_day.values()), 4),
            "drop_reasons": dict(Counter(
                r["label"].replace("drop_", "") for r in srows if r["label"] != "keep"
            )),
        }
    return out


def score_bands(rows: list[dict]) -> dict:
    arxiv = [r for r in rows if r["source"] == "arxiv"]
    bands = []
    for low, high in BANDS:
        inb = [r for r in arxiv if low <= float(r["score"]) < high]
        if not inb:
            continue
        keeps = sum(1 for r in inb if r["label"] == "keep")
        bands.append({
            "band": f"{low}-{high}" if high <= 1.0 else f">={low}",
            "n": len(inb),
            "keep": keeps,
            "keep_rate": round(keeps / len(inb), 4),
        })
    floors = []
    for f in FLOORS:
        above = [r for r in arxiv if float(r["score"]) >= f]
        if not above:
            continue
        keeps = sum(1 for r in above if r["label"] == "keep")
        entry = {
            "floor": f,
            "n": len(above),
            "keep": keeps,
            "precision": round(keeps / len(above), 4),
        }
        if keeps == len(above):
            entry["lower_bound_rule_of_three"] = round(1 - 3 / len(above), 4)
            entry["lower_bound_exact"] = exact_lower_bound(len(above))
        floors.append(entry)
    return {"by_band": bands, "by_floor": floors, "population": f"{len(arxiv)} arXiv labels"}


def journal_day_breakdown(rows: list[dict], day: str) -> dict:
    """One day, rank by rank. 2026-08-11's journal p@10 is 0.30 and this is why."""
    drows = sorted(
        [r for r in rows if r["source"] == "journal" and r["date"] == day],
        key=lambda r: r.get("rank", 0),
    )
    return {
        "date": day,
        "n": len(drows),
        "sequence": [
            {"rank": r.get("rank"), "label": r["label"].replace("drop_", ""),
             "title": r.get("title", "")[:70]}
            for r in drows
        ],
    }


# The three candidate rules V1-3 left on the table, measured against the ranks
# that actually publish rather than against the whole corpus.
MATERIALS_TOPICS = {
    "openalex:T10264": "Asphalt Pavement Performance Evaluation",
    "openalex:T11606": "Infrastructure Maintenance and Monitoring",
}
MATERIALS_SUBFIELD = "2205"  # Civil and Structural Engineering


def materials_in_top_ranks(rows: list[dict], top: int = 15) -> dict:
    """How often the pavement-materials drift reaches a rank that publishes.

    V1-3 counted 36 works in 4,674 (0.77%) corpus-wide and deliberately applied
    no rule. This asks the other half — how often it reaches the ranks that
    publish — and in doing so measures the rules themselves, because two known
    cases are on record: 2026-08-11 journal ranks 9 and 10, both labelled
    `not_urban`, one on nano-TiO2 pore structure and one on asphalt ageing.

    **V1-3's keyword rule catches neither, even with the abstracts.** The
    nano-TiO2 paper contains no term the vocabulary knows; the asphalt paper
    contains exactly one and the rule requires two. That is a finding about the
    rule, and widening the keywords until they hit two papers already known
    would be fitting the rule to the answer.

    The topic pair catches one of the two. The subfield catches both and is
    blunt — V1-3 measured 77 of its 107 corpus works as not materials at all.
    All three are reported; none is applied.
    """
    from materials_drift import looks_like_materials  # type: ignore

    from pipeline import store

    def facts(row: dict) -> tuple[str, list, str]:
        item = store.load_item(row["work_key"])
        if item is None:
            return "", [], ""
        topics = [(t.id, t.label, t.subfield) for t in item.entities.topics]
        return (item.bibliography.abstract or ""), topics, (
            topics[0][2] if topics else ""
        )

    considered = [r for r in rows if r.get("rank", 99) <= top]
    by_rule: dict[str, list[dict]] = {"keyword": [], "topic_pair": [], "subfield_2205": []}
    for r in considered:
        abstract, topics, primary_subfield = facts(r)
        entry = {
            "date": r["date"],
            "source": r["source"],
            "rank": r.get("rank"),
            "label": r["label"].replace("drop_", ""),
            "title": r.get("title", "")[:70],
            "primary_topic": topics[0][1] if topics else None,
        }
        if looks_like_materials(r.get("title", ""), abstract):
            by_rule["keyword"].append(entry)
        if any(t[0] in MATERIALS_TOPICS for t in topics):
            by_rule["topic_pair"].append(entry)
        if primary_subfield == MATERIALS_SUBFIELD:
            by_rule["subfield_2205"].append(entry)

    return {
        "population": f"{len(considered)} labels at rank <= {top}, both paths, 5 days",
        "by_rule": {
            name: {
                "n_matched": len(hits),
                "by_label": dict(Counter(h["label"] for h in hits)),
                "matches": hits,
            }
            for name, hits in by_rule.items()
        },
        "known_cases": (
            "2026-08-11 journal ranks 9 and 10 are both materials papers and both "
            "labelled not_urban. keyword catches 0 of 2, topic_pair 1 of 2, "
            "subfield_2205 2 of 2."
        ),
        "applied": False,
    }


def cites_canon_on_ranked_only(rows: list[dict]) -> dict:
    """V2's binary, re-run on the ranked sample alone.

    **The probe is not pooled in.** It is 30 band-stratified labels drawn on the
    very quantity being tested, and mixing it into a rate is exactly the misuse
    `LabelSetMisuse` exists to prevent. This asks a narrower question: does the
    direction hold in the ranked sample now that it is 148 rows?
    """
    from affinity_normalisation import fisher_2x2  # type: ignore
    from journal_metrics import canon_sets, cites_canon  # type: ignore

    from pipeline.graph.citation import load_reference_base

    foundation, _ = canon_sets()
    refs = {r["work_key"]: (r.get("referenced_works") or []) for r in load_reference_base()}

    scored = [r for r in rows if refs.get(r["work_key"])]
    cites = [r for r in scored if cites_canon(refs[r["work_key"]], foundation)]
    zero = [r for r in scored if not cites_canon(refs[r["work_key"]], foundation)]

    def rates(subset: list[dict]) -> dict:
        q = sum(1 for r in subset if r["label"] == "drop_not_our_kind")
        keep = sum(1 for r in subset if r["label"] == "keep")
        return {
            "n": len(subset),
            "keep": keep,
            "keep_rate": round(keep / len(subset), 4) if subset else None,
            "not_our_kind": q,
            "not_our_kind_rate": round(q / len(subset), 4) if subset else None,
        }

    c, z = rates(cites), rates(zero)
    return {
        "population": (
            f"{len(scored)} of {len(rows)} ranked labels have a reference list; "
            f"the affinity probe is NOT included"
        ),
        "cites_canon": c,
        "no_canon": z,
        "fisher_p_keep": fisher_2x2(
            c["keep"], c["n"] - c["keep"], z["keep"], z["n"] - z["keep"]
        ),
        "fisher_p_not_our_kind": fisher_2x2(
            c["not_our_kind"], c["n"] - c["not_our_kind"],
            z["not_our_kind"], z["n"] - z["not_our_kind"],
        ),
    }


def kind_classifier_readiness(rows: list[dict]) -> dict:
    """U1 estimated a minimum `q` count. Compare it against what exists now."""
    q_ranked = sum(1 for r in rows if r["label"] == "drop_not_our_kind")
    q_probe = sum(
        1 for r in load_labels("affinity_probe") if r["label"] == "drop_not_our_kind"
    )
    return {
        "q_in_ranked_sample": q_ranked,
        "q_in_probe": q_probe,
        "q_total_if_pooled_for_training": q_ranked + q_probe,
        "u1_floor_to_fit": 30,
        "u1_floor_to_trust": 90,
        "can_fit_now": (q_ranked + q_probe) >= 30,
        "note": (
            "Pooling the two files is legitimate for *training* and never for a "
            "rate — the probe's class balance is a property of its sampling "
            "design. The fit floor is met; the trust floor is not, and 18 of the "
            "25 ranked `q` labels are on the journal path, where the error is a "
            "scope question rather than a classification one."
        ),
        "q_by_source": dict(Counter(
            r["source"] for r in rows if r["label"] == "drop_not_our_kind"
        )),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = load_labels("relevance")
    out = {
        "population": (
            f"{len(rows)} ranked labels over "
            f"{len({r['date'] for r in rows})} days (runs/labels/relevance.jsonl)"
        ),
        "q1b": q1b(rows),
        "arxiv_score_bands": score_bands(rows),
        "journal_2026_08_11": journal_day_breakdown(rows, "2026-08-11"),
        "materials_in_top_15": materials_in_top_ranks(rows),
        "cites_canon_ranked_only": cites_canon_on_ranked_only(rows),
        "kind_classifier_readiness": kind_classifier_readiness(rows),
    }
    if a.json:
        Path(a.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n",
        )

    print(out["population"])
    print("\nQ1b — precision@10, mean of daily values")
    for src, v in out["q1b"].items():
        bar = "PASS" if v["precision_at_10"] >= 0.70 else "**BELOW 0.70**"
        print(f"  {src:<8} {v['precision_at_10']:.3f}  {bar}   n={v['n_labels']}, "
              f"spread {v['spread']:.2f}")
        print(f"           per day: {v['per_day']}")
        print(f"           drops: {v['drop_reasons']}")

    print("\narXiv score bands")
    for b in out["arxiv_score_bands"]["by_band"]:
        print(f"  {b['band']:<12} n={b['n']:<3} keep={b['keep']:<3} rate={b['keep_rate']}")
    print("\narXiv floors")
    for f in out["arxiv_score_bands"]["by_floor"]:
        tail = ""
        if "lower_bound_exact" in f:
            tail = (f"   95% lower bound: exact {f['lower_bound_exact']}, "
                    f"rule-of-three {f['lower_bound_rule_of_three']}")
        print(f"  >={f['floor']:<5} n={f['n']:<3} precision={f['precision']}{tail}")

    m = out["materials_in_top_15"]
    print(f"\nmaterials at rank <= 15 ({m['population']}) — rule NOT applied")
    for name, r in m["by_rule"].items():
        print(f"   {name:<14} {r['n_matched']:>2} matched  {r['by_label']}")
        for h in r["matches"][:6]:
            print(f"       {h['date']} {h['source']:<8} rank {h['rank']:>2} "
                  f"[{h['label']}] {h['title'][:50]}")

    c = out["cites_canon_ranked_only"]
    print(f"\ncites_canon on the ranked sample only ({c['population']})")
    print(f"   cites canon: keep {c['cites_canon']['keep_rate']}, "
          f"not_our_kind {c['cites_canon']['not_our_kind_rate']} (n={c['cites_canon']['n']})")
    print(f"   no canon   : keep {c['no_canon']['keep_rate']}, "
          f"not_our_kind {c['no_canon']['not_our_kind_rate']} (n={c['no_canon']['n']})")
    print(f"   Fisher p — keep {c['fisher_p_keep']}, not_our_kind {c['fisher_p_not_our_kind']}")

    k = out["kind_classifier_readiness"]
    print(f"\nkind classifier: q={k['q_in_ranked_sample']} ranked + {k['q_in_probe']} probe "
          f"= {k['q_total_if_pooled_for_training']}; fit floor {k['u1_floor_to_fit']}, "
          f"trust floor {k['u1_floor_to_trust']}; can fit now: {k['can_fit_now']}")
    print(f"   q by source: {k['q_by_source']}")


if __name__ == "__main__":
    main()
