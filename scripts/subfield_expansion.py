"""S5: which OpenAlex subfields should the whitelist builder look at? (phase 0f)

D10 built the whitelist from three subfields — 3322 Urban Studies, 3305
Geography/Planning/Development, 3313 Transportation. One of the three is
Transportation, so a list half made of transport journals is not a bias that
crept in, it is the rule working as written. Phase 0e measured the consequence:
the whitelist is 49.2% transport and the canon it produced was 45.3%.

Rather than argue about which subfields *ought* to count, this asks what our
corpus actually cites. Every resolved reference carries a subfield, and the
canon's recency weighting applies here unchanged: a subfield we leaned on this
month counts for more than one we leaned on in May.

Reports only. `config/pipeline.yaml`'s `whitelist_subfields` is not touched.

Usage:
    uv run python scripts/subfield_expansion.py --json runs/subfield_expansion.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import cfg, journals_vocab  # noqa: E402
from pipeline.graph.citation import load_reference_base  # noqa: E402
from pipeline.graph.daily_canon import load_resolved  # noqa: E402

CURRENT = {"3322", "3305", "3313"}


def _parse(value: str):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    half_life = float(cfg("citation.canon_half_life_days", 180))
    records = load_reference_base()
    dates = [d for d in (_parse(r.get("date", "")) for r in records) if d]
    today = max(dates) if dates else date.today()

    weighted: dict[str, float] = defaultdict(float)
    raw: dict[str, int] = defaultdict(int)
    for record in records:
        d = _parse(record.get("date", ""))
        if d is None:
            continue
        w = math.pow(0.5, max((today - d).days, 0) / half_life)
        for ref in record.get("referenced_works") or []:
            weighted[ref] += w
            raw[ref] += 1

    resolved = load_resolved()
    by_sub_w: dict[str, float] = defaultdict(float)
    by_sub_n: dict[str, int] = defaultdict(int)
    sub_name: dict[str, str] = {}
    venues_by_sub: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    venue_name: dict[str, str] = {}
    unresolved = 0

    for ref, w in weighted.items():
        row = resolved.get(ref)
        if not row or not row.get("subfield_id"):
            unresolved += 1
            continue
        sid = row["subfield_id"]
        sub_name[sid] = row.get("subfield") or sid
        by_sub_w[sid] += w
        by_sub_n[sid] += raw[ref]
        if row.get("venue_id"):
            venues_by_sub[sid][row["venue_id"]] += raw[ref]
            venue_name[row["venue_id"]] = row.get("venue") or row["venue_id"]

    total_w = sum(by_sub_w.values()) or 1.0
    covered = sum(w for s, w in by_sub_w.items() if s in CURRENT)

    whitelist_ids = {s["id"] for s in (journals_vocab().get("sources") or []) if s.get("id")}

    ranked = sorted(by_sub_w, key=lambda s: -by_sub_w[s])
    rows = []
    for sid in ranked[: a.top]:
        venues = sorted(venues_by_sub[sid], key=lambda v: -venues_by_sub[sid][v])
        new_venues = [v for v in venues if v not in whitelist_ids]
        rows.append({
            "subfield_id": sid,
            "name": sub_name.get(sid, sid),
            "in_current_set": sid in CURRENT,
            "share_weighted": round(by_sub_w[sid] / total_w, 4),
            "our_citations": by_sub_n[sid],
            "distinct_venues": len(venues),
            "venues_not_on_whitelist": len(new_venues),
            "top_venues": [
                {"id": v, "name": venue_name.get(v), "cited": venues_by_sub[sid][v],
                 "on_whitelist": v in whitelist_ids}
                for v in venues[:3]
            ],
        })

    # Where does Urban Design live? The Journal of Urban Design was the case that
    # started this, and if its subfield is outside the three the builder looks
    # at, that is a structural answer rather than a threshold one.
    urban_design = [
        {"subfield_id": sid, "name": sub_name.get(sid), "venue": venue_name.get(vid),
         "cited": n, "on_whitelist": vid in whitelist_ids}
        for sid, vs in venues_by_sub.items()
        for vid, n in vs.items()
        if "Urban Design" in (venue_name.get(vid) or "")
    ]

    result = {
        "population": "resolved references in content/graph/references.jsonl",
        "references_weighted": len(weighted),
        "resolved_with_subfield": len(weighted) - unresolved,
        "unresolved": unresolved,
        "current_subfields": sorted(CURRENT),
        "coverage_of_current_set": round(covered / total_w, 4),
        "subfields": rows,
        "urban_design_venues": urban_design,
    }

    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )

    print(f"resolved references with a subfield: {result['resolved_with_subfield']} "
          f"(unresolved {unresolved})")
    print(f"the three current subfields cover {result['coverage_of_current_set']:.1%} "
          f"of what our corpus cites\n")
    print(f"{'':>2} {'share':>7} {'cited':>7} {'venues':>7} {'new':>5}  subfield")
    for r in rows:
        mark = "*" if r["in_current_set"] else " "
        print(f"{mark:>2} {r['share_weighted']:>7.2%} {r['our_citations']:>7} "
              f"{r['distinct_venues']:>7} {r['venues_not_on_whitelist']:>5}  "
              f"{r['subfield_id']} {r['name'][:40]}")
    print("\n(* = already in the builder's subfield set)")
    if urban_design:
        print("\nUrban Design venues:")
        for u in urban_design:
            print(f"  {u['subfield_id']} {str(u['name'])[:34]:<34} {str(u['venue'])[:32]:<32} "
                  f"cited {u['cited']:>3}  whitelist={u['on_whitelist']}")


if __name__ == "__main__":
    main()
