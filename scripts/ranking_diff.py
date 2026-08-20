"""T3 verification: do the new journal signals agree with YJUN's labels? (0g)

Neither signal is applied — `selection.journal_ranking` stays `placeholder`.
YJUN labelled 30 items and `runs/labels/relevance.jsonl` records the `rank` each
was shown at, so changing the ranking now would leave those labels attributable
to no ranking at all. This measures and diffs.

**n = 15 on the journal side, of which 2 are drops.** That cannot settle
anything. It can show direction, and a signal that puts both drops at the top
would be refuted by it, which is worth knowing before building further.

Usage:
    uv run python scripts/ranking_diff.py --json runs/ranking_diff.json
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
from pipeline.config import cfg  # noqa: E402
from pipeline.graph.citation import load_work_index  # noqa: E402

SIGNALS = ("foundation_only", "with_instruments", "hits_sqrt", "linear", "raw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    metrics = json.loads((ROOT / "runs/journal_metrics.json").read_text(encoding="utf-8"))
    affinity = {r["work_key"]: r for r in metrics["affinity_top"]}
    # `affinity_top` is truncated; recompute the full map from the same inputs.
    from pipeline.graph.citation import load_reference_base
    from scripts.journal_metrics import canon_affinity, canon_sets  # type: ignore

    foundation, instrument = canon_sets()
    both = {**instrument, **foundation}
    affinity = {}
    for record in load_reference_base():
        refs = record.get("referenced_works") or []
        affinity[record["work_key"]] = {
            "references": len(refs),
            "foundation_only": canon_affinity(refs, foundation),
            "with_instruments": canon_affinity(refs, both),
            "linear": canon_affinity(refs, foundation, "linear"),
            "raw": canon_affinity(refs, foundation, "none"),
            "hits_sqrt": round(
                sum(1 for r in refs if r in foundation) / (len(refs) ** 0.5), 6
            ) if refs else 0.0,
            "foundation_hits": sum(1 for r in refs if r in foundation),
        }

    prior = {
        s["id"]: s.get("prestige_pct_in_subfield")
        for s in metrics["sources"]
    }

    labels = [
        json.loads(line)
        for line in (ROOT / "runs/labels/relevance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    journal_labels = [r for r in labels if r.get("source") == "journal"]

    index = load_work_index()
    rows = []
    for row in journal_labels:
        wk = row["work_key"]
        item = store.load_item(wk)
        venue = item.bibliography.primary_location.source_id if item else None
        aff = affinity.get(wk, {})
        rows.append({
            "work_key": wk,
            "title": (index.get(wk, {}).get("title") or row.get("title") or "")[:60],
            "label": row["label"],
            "kept": row["label"] == "keep",
            "venue_prior": prior.get(venue),
            "references": aff.get("references", 0),
            "foundation_hits": aff.get("foundation_hits", 0),
            **{k: aff.get(k) for k in SIGNALS},
        })

    def split(key: str):
        keep = [r[key] for r in rows if r["kept"] and r.get(key) is not None]
        drop = [r[key] for r in rows if not r["kept"] and r.get(key) is not None]
        return keep, drop

    comparison = {}
    for key in ("venue_prior",) + SIGNALS:
        keep, drop = split(key)
        if not keep or not drop:
            comparison[key] = {"note": "one side empty"}
            continue
        ranked = sorted(rows, key=lambda r: -(r.get(key) or 0))
        drop_positions = [i + 1 for i, r in enumerate(ranked) if not r["kept"]]
        comparison[key] = {
            "keep_median": round(statistics.median(keep), 4),
            "drop_median": round(statistics.median(drop), 4),
            "separates": statistics.median(keep) > statistics.median(drop),
            "drop_ranks_of_15": drop_positions,
        }

    result = {
        "population": "journal-side labels in runs/labels/relevance.jsonl",
        "n": len(rows),
        "keeps": sum(1 for r in rows if r["kept"]),
        "drops": sum(1 for r in rows if not r["kept"]),
        "caveat": (
            "n=15 with 2 drops. Direction only; no signal is accepted or "
            "rejected on this, and nothing is applied."
        ),
        "applied": str(cfg("selection.journal_ranking", "placeholder")),
        "comparison": comparison,
        "rows": sorted(rows, key=lambda r: -(r.get("foundation_only") or 0)),
    }

    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )

    print(f"journal labels: {result['n']} ({result['keeps']} keep / {result['drops']} drop)")
    print(f"applied ranking: {result['applied']}\n")
    print(f"{'signal':<20} {'keep med':>10} {'drop med':>10}  separates  drop ranks")
    for key, c in comparison.items():
        if "note" in c:
            continue
        print(f"{key:<20} {c['keep_median']:>10.4f} {c['drop_median']:>10.4f}  "
              f"{str(c['separates']):>9}  {c['drop_ranks_of_15']}")
    print("\nthe two `not_our_kind` items, by foundation_only rank:")
    for i, r in enumerate(result["rows"], 1):
        if not r["kept"]:
            print(f"   rank {i}/15  {r['foundation_only']:>8.3f}  "
                  f"hits={r['foundation_hits']}/{r['references']}  {r['title']}")


if __name__ == "__main__":
    main()
