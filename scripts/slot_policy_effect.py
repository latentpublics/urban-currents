"""V1-1: what the elastic arXiv policy does to 90 days of publishing (phase 0i).

Replays the backfill's scored candidates through the old fixed-slot rule and the
new floor-based one, and reports the daily publish count and composition under
each. Nothing is published and no score is recomputed — this reads
`runs/backfill/scores.jsonl`, which the backfill already produced.

The comparison that matters is not "does the total drop" (it does, by design)
but **which items stop publishing**: the arXiv 0.35-0.80 mass the labels found
to be a third to a half keeps.

Usage:
    uv run python scripts/slot_policy_effect.py --json runs/slot_policy_effect.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.calibrate import load_scores  # noqa: E402
from pipeline.config import cfg  # noqa: E402


def old_rule(day: list[dict], threshold: float, journal_slots=12, arxiv_slots=12):
    """Fixed 12/12 with two-way lending — the rule up to phase 0h."""
    journal = sorted(
        (r for r in day if r["source"] == "journal"), key=lambda r: -r["headline"]
    )
    arxiv = sorted(
        (r for r in day if r["source"] == "arxiv" and r["relevance"] >= threshold),
        key=lambda r: -r["relevance"],
    )
    j, a = journal[:journal_slots], arxiv[:arxiv_slots]
    spare = (journal_slots - len(j)) + (arxiv_slots - len(a))
    if spare:
        j = journal[: len(j) + spare]
        spare = journal_slots + arxiv_slots - len(j) - len(a)
        if spare:
            a = arxiv[: len(a) + spare]
    return j, a


def new_rule(day: list[dict], threshold: float, floor: float, arxiv_max: int,
             journal_base: int, journal_max: int):
    journal = sorted(
        (r for r in day if r["source"] == "journal"), key=lambda r: -r["headline"]
    )
    arxiv = sorted(
        (r for r in day if r["source"] == "arxiv" and r["relevance"] >= max(threshold, floor)),
        key=lambda r: -r["relevance"],
    )
    a = arxiv[:arxiv_max]
    j = journal[: min(journal_max, journal_base + max(0, arxiv_max - len(a)))]
    return j, a


def describe(counts: list[int]) -> dict:
    s = sorted(counts)
    return {
        "days": len(s),
        "median": statistics.median(s) if s else 0,
        "mean": round(statistics.fmean(s), 2) if s else 0,
        "p25": s[len(s) // 4] if s else 0,
        "p75": s[(3 * len(s)) // 4] if s else 0,
        "min": s[0] if s else 0,
        "max": s[-1] if s else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = load_scores()
    if not rows:
        raise SystemExit("no runs/backfill/scores.jsonl — run `uc backfill --days 90` first")

    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("date"):
            by_day[r["date"]].append(r)

    threshold = float(cfg("classifier.threshold", 0.35))
    floor = float(cfg("selection.arxiv.floor", 0.80))
    arxiv_max = int(cfg("selection.arxiv.max", 12))
    journal_base = int(cfg("selection.slots.journal", 12))
    journal_max = int(cfg("selection.journal.max", 15))

    old_tot, new_tot = [], []
    old_a, new_a, old_j, new_j = [], [], [], []
    dropped_arxiv_scores: list[float] = []
    added_journal = 0
    for _day, items in sorted(by_day.items()):
        oj, oa = old_rule(items, threshold)
        nj, na = new_rule(items, threshold, floor, arxiv_max, journal_base, journal_max)
        old_tot.append(len(oj) + len(oa))
        new_tot.append(len(nj) + len(na))
        old_a.append(len(oa))
        new_a.append(len(na))
        old_j.append(len(oj))
        new_j.append(len(nj))
        kept_keys = {r["work_key"] for r in na}
        dropped_arxiv_scores += [
            r["relevance"] for r in oa if r["work_key"] not in kept_keys
        ]
        added_journal += max(0, len(nj) - len(oj))

    bands = {
        "<0.50": sum(1 for v in dropped_arxiv_scores if v < 0.50),
        "0.50-0.70": sum(1 for v in dropped_arxiv_scores if 0.50 <= v < 0.70),
        "0.70-0.80": sum(1 for v in dropped_arxiv_scores if 0.70 <= v < 0.80),
        ">=0.80": sum(1 for v in dropped_arxiv_scores if v >= 0.80),
    }
    # Labelled keep rate per band (90 relevance labels, arXiv side, 3 days).
    # Multiplying gives an estimate of what the change stops publishing, and it
    # is an estimate — the bands were measured on 45 arXiv labels, not on these
    # 90 days.
    keep_rate = {"<0.50": 0.33, "0.50-0.70": 0.33, "0.70-0.80": 0.44, ">=0.80": 1.0}
    est_keeps_lost = round(sum(bands[b] * keep_rate[b] for b in bands), 1)

    out = {
        "population": "90-day backfill candidates (runs/backfill/scores.jsonl)",
        "policy": {
            "threshold_unchanged": threshold,
            "arxiv_floor": floor,
            "arxiv_max": arxiv_max,
            "journal_base": journal_base,
            "journal_max": journal_max,
        },
        "published_per_day": {"old": describe(old_tot), "new": describe(new_tot)},
        "arxiv_per_day": {"old": describe(old_a), "new": describe(new_a)},
        "journal_per_day": {"old": describe(old_j), "new": describe(new_j)},
        "arxiv_items_no_longer_published": {
            "total": len(dropped_arxiv_scores),
            "by_score_band": bands,
            "estimated_keeps_among_them": est_keeps_lost,
            "estimated_drops_among_them": len(dropped_arxiv_scores) - est_keeps_lost,
            "note": (
                "keep rates come from 45 labelled arXiv items over 3 days, applied "
                "to 90 days of candidates — an estimate, not a count"
            ),
        },
        "journal_slots_added": added_journal,
    }
    if a.json:
        Path(a.json).write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

    print(f"population: {out['population']}  ({len(by_day)} days)")
    for name, key in (("published/day", "published_per_day"),
                      ("arxiv/day", "arxiv_per_day"),
                      ("journal/day", "journal_per_day")):
        o, n = out[key]["old"], out[key]["new"]
        print(f"  {name:<15} old median {o['median']:>5} (p25 {o['p25']}, p75 {o['p75']}, "
              f"max {o['max']})   new median {n['median']:>5} (p25 {n['p25']}, "
              f"p75 {n['p75']}, max {n['max']})")
    d = out["arxiv_items_no_longer_published"]
    print(f"\narXiv items no longer published: {d['total']}")
    for band, n in d["by_score_band"].items():
        print(f"   {band:<10} {n:>5}   labelled keep rate {keep_rate[band]}")
    print(f"   estimated keeps lost {d['estimated_keeps_among_them']}, "
          f"drops avoided {d['estimated_drops_among_them']}")
    print(f"journal slots added across the range: {out['journal_slots_added']}")


if __name__ == "__main__":
    main()
