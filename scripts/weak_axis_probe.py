"""Is weak work visible in an abstract? (phase 0Q, R1 — measurement only)

`scripts/weak_method_feature.py` asked whether the **subfield** gate could act on
`drop_weak_method`, and answered no: it catches 1 of 6 and half the weak papers
share a subfield with a keep. That answer stands.

This asks the different question the rename opened. Both weak labels are now
claimed to be abstract-visible:

  `drop_weak_method`      HOW the work was done   (10 rows)
  `drop_weak_arguments`   WHAT it claims          (5 rows)

so the axis to test is the **abstract text itself**, not the metadata.

## What is measured, and why these features

Chosen before looking, and deliberately cheap and interpretable rather than an
embedding — with 15 positives an embedding distance would be unfalsifiable, and
the point here is direction, not a model.

  `abstract_chars`      length. A thin claim often has a short abstract.
  `n_quantities`        numbers with units/counts — what was actually measured.
  `has_sample_size`     an explicit n, already extracted as a rule signal.
  `has_temporal`        a study period, likewise.
  `n_hedges`            "may", "could", "suggests", "potential", "promising" —
                        the register a narrow claim is written in.
  `n_comparatives`      "outperforms", "compared to", "baseline", "state of the
                        art" — the register of a claim with something to beat.

## What this must not do

**15 positives. No conclusion.** The report is direction and the number of labels
that would make it decidable, nothing else. Nothing is adopted, and no default
moves. Fitting a rule to 15 known cases is what 0i refused the materials-keyword
expansion for.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from journal_gate import _index_stage_items  # noqa: E402
from pipeline import paths, store  # noqa: E402
from pipeline.labeling import load_labels, superseded  # noqa: E402
from pipeline.signals import _N_EQUALS, _SAMPLE, _TEMPORAL_LOOSE, _TEMPORAL_RANGE  # noqa: E402

HEDGE = re.compile(
    r"\b(may|might|could|suggests?|potential(ly)?|promising|preliminary|"
    r"explorator(y|ily)|indicat(e|es)\s+that|appears?\s+to)\b", re.I
)
COMPARATIVE = re.compile(
    r"\b(outperform(s|ed)?|compared\s+(to|with)|baselines?|state[- ]of[- ]the[- ]art|"
    r"benchmark(s|ed)?|ablation|improv(es|ed|ement)\s+(of|by)|versus|vs\.)\b", re.I
)
QUANTITY = re.compile(
    r"\b\d[\d,.]*\s*(%|percent|km|m|km2|ha|years?|months?|days?|hours?|"
    r"cities|stations|households|respondents|trips|observations|samples)\b", re.I
)

WEAK = {"drop_weak_method", "drop_weak_arguments"}


def features(abstract: str) -> dict:
    a = abstract or ""
    return {
        "abstract_chars": len(a),
        "n_quantities": len(QUANTITY.findall(a)),
        "has_sample_size": int(bool(_SAMPLE.search(a) or _N_EQUALS.search(a))),
        "has_temporal": int(bool(_TEMPORAL_RANGE.search(a) or _TEMPORAL_LOOSE.search(a))),
        "n_hedges": len(HEDGE.findall(a)),
        "n_comparatives": len(COMPARATIVE.findall(a)),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    index = _index_stage_items()
    groups: dict[str, list[dict]] = {"keep": [], "drop_weak_method": [],
                                     "drop_weak_arguments": []}
    no_abstract = Counter()
    for r in superseded(load_labels("relevance")):
        label = r["label"]
        if label not in groups:
            continue
        # Both stores. The run stages hold candidates that were never published
        # and `content/items/` holds the ones that were; looking in only one of
        # them reports a coverage gap that is an artefact of the lookup. The
        # first draft of this script did exactly that and made it look as though
        # 80 of 99 labelled papers had no abstract.
        item = index.get(r["work_key"]) or store.load_item(r["work_key"])
        abstract = (item.bibliography.abstract or "") if item else ""
        if not abstract:
            no_abstract[label] += 1
            continue
        groups[label].append({**features(abstract), "title": r.get("title", "")})

    print("population — relevance labels with an abstract on disk")
    for label, rows in groups.items():
        print(f"  {label:<22} {len(rows):>3}  (no abstract: {no_abstract[label]})")
    print("\n**15 positives at most. This is a direction, not a finding.**\n")

    keys = list(features("").keys())
    print(f"{'feature':<18}{'keep':>16}{'weak_method':>16}{'weak_arguments':>18}")
    seps = {}
    for k in keys:
        cells = []
        for label in ("keep", "drop_weak_method", "drop_weak_arguments"):
            vals = [r[k] for r in groups[label]]
            cells.append(statistics.fmean(vals) if vals else float("nan"))
        print(f"{k:<18}{cells[0]:>16.2f}{cells[1]:>16.2f}{cells[2]:>18.2f}")
        seps[k] = cells

    # Overlap is the honest summary at this n: how much of the weak group falls
    # inside the keeps' interquartile range. A feature that separates would put
    # most of the weak group outside it.
    print("\nshare of each weak group inside the keeps' interquartile range")
    print("(1.00 = indistinguishable on this feature; low = some separation)")
    print(f"{'feature':<18}{'weak_method':>14}{'weak_arguments':>16}")
    for k in keys:
        kv = sorted(r[k] for r in groups["keep"])
        if len(kv) < 4:
            continue
        q1 = statistics.quantiles(kv, n=4)[0]
        q3 = statistics.quantiles(kv, n=4)[2]
        row = []
        for label in ("drop_weak_method", "drop_weak_arguments"):
            vals = [r[k] for r in groups[label]]
            row.append(sum(1 for v in vals if q1 <= v <= q3) / len(vals) if vals else float("nan"))
        print(f"{k:<18}{row[0]:>14.2f}{row[1]:>16.2f}")

    # --- how many labels would make this decidable? ----------------------
    #
    # Reported because "not enough data" is only useful with a number attached.
    # Treat each feature as binary (present / absent), take the observed rates
    # as the effect size, and use the standard two-proportion sample size at
    # 80% power and alpha 0.05. That is optimistic — the rates ARE the noise at
    # this n — so it is a floor on what would be needed, not an estimate.
    import math

    print("\nlabels per group that would settle it (binary form of each feature,")
    print("80% power, alpha 0.05, observed rates as effect size — a FLOOR)")
    print(f"{'feature':<18}{'keep rate':>11}{'weak rate':>11}{'n per group':>13}")
    needed = {}
    weak_rows = groups["drop_weak_method"] + groups["drop_weak_arguments"]
    for k in keys:
        p1 = sum(1 for r in groups["keep"] if r[k] > 0) / max(1, len(groups["keep"]))
        p2 = sum(1 for r in weak_rows if r[k] > 0) / max(1, len(weak_rows))
        if p1 == p2:
            print(f"{k:<18}{p1:>11.2f}{p2:>11.2f}{'no difference':>13}")
            needed[k] = None
            continue
        pbar = (p1 + p2) / 2
        num = (1.96 * math.sqrt(2 * pbar * (1 - pbar))
               + 0.84 * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        n = math.ceil(num / (p1 - p2) ** 2)
        needed[k] = n
        print(f"{k:<18}{p1:>11.2f}{p2:>11.2f}{n:>13}")
    live = [v for v in needed.values() if v]
    if live:
        print(f"\ncheapest feature needs {min(live)} per group; "
              f"we have {len(weak_rows)} weak.")

    out = paths.RUNS / "weak_axis_probe.json"
    out.write_text(json.dumps({
        "note": "measurement only (0Q R1); nothing adopted, no default changed",
        "n": {k: len(v) for k, v in groups.items()},
        "no_abstract": dict(no_abstract),
        "means": {k: {"keep": v[0], "weak_method": v[1], "weak_arguments": v[2]}
                  for k, v in seps.items()},
        "labels_needed_per_group_floor": needed,
    }, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
