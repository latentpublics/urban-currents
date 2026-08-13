"""R0 verification: does the new canon scope rule sort the known cases right?

Ten works were dropped by the old venue rule. Six of them were dropped wrongly —
our own corpus cited them, and the only thing against them was appearing in a
journal we do not poll daily. Four were dropped rightly: general research
instruments every field cites.

A rule change that cannot be checked against cases whose answer is already known
is a guess. This runs both rules over the same works and prints the split.

Usage:
    uv run python scripts/canon_scope_check.py --json runs/canon_scope_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import journals_vocab  # noqa: E402
from pipeline.graph.canon import _ids, _in_scope, accumulate  # noqa: E402

# The ten from the phase 0d report, with the verdict a human already reached.
# `expected_in` is the claim this check tests, not something it derives.
KNOWN = {
    "Introducing the": ("15-Minute City", True),
    "Built Environment Correlates of Walking": ("Saelens & Handy", True),
    "Measuring the Unmeasurable": ("Ewing & Handy", True),
    "Congested traffic states": ("Treiber/Helbing", True),
    "Rethinking Informality": ("Roy", True),
    "Mapping global urban boundaries": ("GAIA urban boundaries", True),
    "Using thematic analysis in psychology": ("Braun & Clarke", False),
    "Cutoff criteria for fit indexes": ("Hu & Bentler", False),
    "Evaluating Structural Equation Models": ("Fornell & Larcker", False),
    "A new criterion for assessing": ("Henseler", False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from pipeline.collectors.openalex import configure_pyalex

    pyalex = configure_pyalex()
    rows = {r["openalex_id"]: r for r in accumulate()}
    eligible = [r for r in rows.values() if r["archive_citations"] >= 2]
    ids = [r["openalex_id"].split(":", 1)[1] for r in eligible]

    fetched = {}
    cost = 0.0
    for i in range(0, len(ids), 50):
        res, meta = (
            pyalex.Works().filter(openalex_id="|".join(ids[i : i + 50]))
            .get(per_page=50, return_meta=True)
        )
        cost += float((meta or {}).get("cost_usd") or 0.0)
        for w in res:
            fetched[(w.get("id") or "").rsplit("/", 1)[-1]] = w

    whitelist = {
        s["id"] for s in (journals_vocab().get("sources") or [])
        if s.get("id") and s.get("include", True)
    }

    out = []
    for work in fetched.values():
        title = work.get("display_name") or ""
        match = next((k for k in KNOWN if title.startswith(k)), None)
        if not match:
            continue
        label, expected_in = KNOWN[match]
        topic, subfield, _ = _ids(work)
        venue_ok = _in_scope(work, whitelist, "venue")
        subfield_ok = _in_scope(work, whitelist, "subfield")
        out.append({
            "label": label,
            "title": title,
            "venue": ((work.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "topic": (work.get("primary_topic") or {}).get("display_name"),
            "topic_id": topic,
            "subfield_id": subfield,
            "expected_in_scope": expected_in,
            "old_rule_venue": venue_ok,
            "new_rule_subfield": subfield_ok,
            "correct": subfield_ok == expected_in,
        })

    out.sort(key=lambda r: (not r["expected_in_scope"], r["label"]))
    result = {
        "checked": len(out),
        "correct": sum(1 for r in out if r["correct"]),
        "wrong": [r["label"] for r in out if not r["correct"]],
        "openalex_cost_usd": round(cost, 6),
        "rows": out,
    }
    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )

    print(f"{'case':<26} {'expect':>7} {'old':>6} {'new':>6}   topic")
    for r in out:
        flag = "" if r["correct"] else "  <-- WRONG"
        print(
            f"{r['label'][:26]:<26} {'IN' if r['expected_in_scope'] else 'OUT':>7} "
            f"{'in' if r['old_rule_venue'] else 'out':>6} "
            f"{'in' if r['new_rule_subfield'] else 'out':>6}   "
            f"{str(r['topic'])[:34]}{flag}"
        )
    print(f"\n{result['correct']}/{result['checked']} correct; cost ${result['openalex_cost_usd']}")


if __name__ == "__main__":
    main()
