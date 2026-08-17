"""V2: which canon-affinity normalisation the probe labels actually support.

The probe found `hits/sqrt(refs)` ranking "Beyond the Western paradigm" (2 canon
hits out of 5 references) at 28.84, above "Multilevel SEM of walkability" (7 out
of 63) at 17.99. A paper with 63 references of which 7 are foundational is more
embedded in the field than one with 5 of which 2 are, not less; the square root
over-rewards a short reference list.

Five candidates are scored against the 30 probe labels and the 45 labelled
journal items, on two questions that are not the same:

- **keep vs drop** — does the number track the judgement at all?
- **keep vs `not_our_kind`** — does it track *this* judgement, which is the one
  a kind classifier would be asked to make?

**n = 30. Fisher's exact on the binary split gives p ~= 0.35.** Nothing here is
significant and nothing here is claimed to be. What can be argued from it is
direction and mechanism.

Usage:
    uv run python scripts/affinity_normalisation.py --json runs/affinity_normalisation.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.graph.citation import load_reference_base  # noqa: E402
from pipeline.labeling import load_labels  # noqa: E402


def variants(refs: list[str], canon: dict[str, float]) -> dict[str, float]:
    """Every normalisation under consideration, from one reference list."""
    n = len(refs)
    hits = [r for r in refs if r in canon]
    h = len(hits)
    weight = sum(canon[r] for r in hits)
    return {
        # The incumbent. 100 * h / sqrt(n) — the constant is cosmetic.
        "hits_over_sqrt_refs": round(100 * h / math.sqrt(n), 4) if n else 0.0,
        # The obvious alternative: how many foundational works it cites, full
        # stop. Immune to reference-list length because it ignores it.
        "raw_hits": float(h),
        # Length-aware without the short-list bonus: log grows slowly enough
        # that a 5-reference paper is not rewarded for being brief.
        "hits_over_log_refs": round(h / math.log(n + math.e), 4) if n else 0.0,
        # Plain share. Punishes long reference lists, which is the mirror error.
        "hits_share": round(h / n, 4) if n else 0.0,
        # Canon weights carried through: citing three central works is not the
        # same as citing three marginal ones.
        "weighted_hits": round(weight, 4),
    }


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Proportion of (positive, negative) pairs ordered correctly. Ties count half."""
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def fisher_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact. Written out because scipy is not a dependency."""
    def logfact(n: int) -> float:
        return math.lgamma(n + 1)

    def p_of(x: int, y: int, z: int, w: int) -> float:
        return math.exp(
            logfact(x + y) + logfact(z + w) + logfact(x + z) + logfact(y + w)
            - logfact(x) - logfact(y) - logfact(z) - logfact(w)
            - logfact(x + y + z + w)
        )

    observed = p_of(a, b, c, d)
    total = 0.0
    n = a + b + c + d
    row1, col1 = a + b, a + c
    for x in range(0, min(row1, col1) + 1):
        y, z, w = row1 - x, col1 - x, n - row1 - col1 + x
        if y < 0 or z < 0 or w < 0:
            continue
        p = p_of(x, y, z, w)
        if p <= observed * (1 + 1e-9):
            total += p
    return round(min(1.0, total), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from journal_metrics import canon_sets  # type: ignore

    foundation, _ = canon_sets()
    refs = {r["work_key"]: (r.get("referenced_works") or []) for r in load_reference_base()}

    rows: list[dict] = []
    for facet in ("affinity_probe", "relevance"):
        for r in load_labels(facet):
            if facet == "relevance" and r.get("source") != "journal":
                continue
            wk = r["work_key"]
            if wk not in refs or not refs[wk]:
                continue
            rows.append({
                "facet": facet,
                "work_key": wk,
                "label": r["label"],
                "title": r.get("title", "")[:80],
                "refs_total": len(refs[wk]),
                **variants(refs[wk], foundation),
            })

    names = list(variants([], {}).keys())
    keep = [r for r in rows if r["label"] == "keep"]
    drop = [r for r in rows if r["label"] != "keep"]
    notkind = [r for r in rows if r["label"] == "drop_not_our_kind"]

    table = {}
    for name in names:
        table[name] = {
            "n_keep": len(keep),
            "n_drop": len(drop),
            "auc_keep_vs_drop": auc([r[name] for r in keep], [r[name] for r in drop]),
            "n_not_our_kind": len(notkind),
            "auc_keep_vs_not_our_kind": auc(
                [r[name] for r in keep], [r[name] for r in notkind]
            ),
            "median_keep": round(statistics.median([r[name] for r in keep]), 4) if keep else None,
            "median_drop": round(statistics.median([r[name] for r in drop]), 4) if drop else None,
        }

    # The specific inversion that started this.
    pair = {}
    for r in rows:
        if r["refs_total"] <= 6 or r["refs_total"] >= 60:
            pair[r["title"][:60]] = {
                "refs_total": r["refs_total"],
                **{k: r[k] for k in names},
            }

    # The binary question, per label set. The probe is the only one *designed*
    # for it — equal draws across affinity bands — and the ranked journal sample
    # is a top-N draw whose affinity distribution is a consequence of the
    # ranking, not of any design. Pooling them produces a flat result that is an
    # artefact of two base rates cancelling, so they are reported apart.
    def binary_for(subset: list[dict]) -> dict:
        cites = [r for r in subset if r["raw_hits"] > 0]
        zero = [r for r in subset if r["raw_hits"] == 0]
        a_ = sum(1 for r in cites if r["label"] == "keep")
        c_ = sum(1 for r in zero if r["label"] == "keep")
        q_cites = sum(1 for r in cites if r["label"] == "drop_not_our_kind")
        q_zero = sum(1 for r in zero if r["label"] == "drop_not_our_kind")
        return {
            "cites_canon": {
                "n": len(cites),
                "keep": a_,
                "keep_rate": round(a_ / len(cites), 4) if cites else None,
                "not_our_kind": q_cites,
                "not_our_kind_rate": round(q_cites / len(cites), 4) if cites else None,
            },
            "no_canon": {
                "n": len(zero),
                "keep": c_,
                "keep_rate": round(c_ / len(zero), 4) if zero else None,
                "not_our_kind": q_zero,
                "not_our_kind_rate": round(q_zero / len(zero), 4) if zero else None,
            },
            "fisher_exact_p_keep": fisher_2x2(a_, len(cites) - a_, c_, len(zero) - c_),
            "fisher_exact_p_not_our_kind": fisher_2x2(
                q_cites, len(cites) - q_cites, q_zero, len(zero) - q_zero
            ),
        }

    binary = {
        "probe_only": binary_for([r for r in rows if r["facet"] == "affinity_probe"]),
        "ranked_journal_only": binary_for([r for r in rows if r["facet"] == "relevance"]),
        "pooled_do_not_read_as_a_rate": binary_for(rows),
    }

    out = {
        "population": (
            f"{len(rows)} labelled items with a reference list — "
            f"{sum(1 for r in rows if r['facet'] == 'affinity_probe')} from the "
            f"band-stratified probe, {sum(1 for r in rows if r['facet'] == 'relevance')} "
            f"journal items from the ranked sample. Counted separately because they "
            f"were sampled differently; pooled here only to fit five curves, never "
            f"to state a rate."
        ),
        "significance": (
            "n is 30 on the probe side. Fisher's exact on the binary split is "
            "reported and it does not reach significance. Direction and mechanism "
            "only."
        ),
        "by_variant": table,
        "short_and_long_reference_lists": pair,
        "binary": binary,
        "zero_is_not_only_the_paper": (
            "A zero measures two things at once: a paper outside the field, and a "
            "corner of the field our canon does not cover. Our canon is 90 days "
            "of one corpus and leans transport, so inland waterway barges, school "
            "start times, lane-change prediction, immigration and crime, and child "
            "stunting all score zero while being exactly our kind. Five of the six "
            "zero-band keeps are of that sort."
        ),
    }
    if a.json:
        Path(a.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n",
        )

    print(out["population"])
    print(f"\n{'variant':<22} {'AUC keep/drop':>14} {'AUC keep/q':>12} "
          f"{'med keep':>10} {'med drop':>10}")
    for name, v in table.items():
        print(f"{name:<22} {str(v['auc_keep_vs_drop']):>14} "
              f"{str(v['auc_keep_vs_not_our_kind']):>12} "
              f"{str(v['median_keep']):>10} {str(v['median_drop']):>10}")
    print(f"\n(keep n={table[names[0]]['n_keep']}, drop n={table[names[0]]['n_drop']}, "
          f"not_our_kind n={table[names[0]]['n_not_our_kind']})")

    print("\nthe inversion, on short and long reference lists:")
    for title, v in list(pair.items())[:8]:
        print(f"   refs={v['refs_total']:<3} sqrt={v['hits_over_sqrt_refs']:>7} "
              f"raw={v['raw_hits']:>4} log={v['hits_over_log_refs']:>6} "
              f"weighted={v['weighted_hits']:>7}  {title[:52]}")

    for label, b in out["binary"].items():
        c, z = b["cites_canon"], b["no_canon"]
        print(f"\n{label}")
        print(f"   cites canon (n={c['n']:>2}): keep {c['keep_rate']}, "
              f"not_our_kind {c['not_our_kind']} ({c['not_our_kind_rate']})")
        print(f"   no canon    (n={z['n']:>2}): keep {z['keep_rate']}, "
              f"not_our_kind {z['not_our_kind']} ({z['not_our_kind_rate']})")
        print(f"   Fisher p — keep {b['fisher_exact_p_keep']}, "
              f"not_our_kind {b['fisher_exact_p_not_our_kind']}")


if __name__ == "__main__":
    main()
