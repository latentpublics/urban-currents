"""Verify that the 90-day backfill is actually complete (roadmap §2.5).

Q2's "median 28 selected items/day" and Q3's quiet-day threshold are both
computed on the backfill. D15 lets a failed collection window be logged and
skipped rather than aborting the run — and on the first attempt a window really
did fail with a 429. So "the backfill finished" is not the same claim as "the
backfill is complete", and the difference matters to two headline numbers.

This script answers three questions:

1. Does every date in the range carry items, and are any suspiciously thin?
2. Do our stored per-window counts match what arXiv reports for the same query?
3. Is the measured intake (~364/day) real, or an artefact of missing windows?

Question 2 is the one that cannot be answered from our own files, so it asks
arXiv for ``totalResults`` on a sample of windows and compares.

Usage:
    uv run python scripts/check_backfill_coverage.py [--verify-windows 3]
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import io
import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import cfg, contact_email  # noqa: E402
from pipeline.paths import RUNS  # noqa: E402

LOW_DAY_FRACTION = 0.25  # a day below this share of the median is worth a look


def load_daily_counts() -> collections.Counter:
    """Total intake per date — both sides of the gate, since the gate is not
    part of what "did we collect it" means."""
    days: collections.Counter = collections.Counter()
    sources = [RUNS / "backfill" / "scores.jsonl"] + [
        Path(p) for p in glob.glob(str(RUNS / "backfill_*" / "gate_rejected.jsonl"))
    ]
    for p in sources:
        if not p.exists():
            continue
        for line in io.open(p, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line).get("date") or ""
            if d:
                days[d] += 1
    return days


def arxiv_total_for(start: datetime.date, end: datetime.date) -> int:
    """``totalResults`` arXiv reports for our exact query over a window."""
    cats = list(cfg("arxiv.categories", []) or [])
    query = (
        "(" + " OR ".join(f"cat:{c}" for c in cats) + ")"
        f" AND submittedDate:[{start.strftime('%Y%m%d')}0000 TO {end.strftime('%Y%m%d')}2359]"
    )
    r = httpx.get(
        cfg("arxiv.api_url", "http://export.arxiv.org/api/query"),
        params={"search_query": query, "start": 0, "max_results": 1},
        headers={"User-Agent": f"urban-currents/0.2 (mailto:{contact_email()})"},
        timeout=120.0,
        follow_redirects=True,
    )
    r.raise_for_status()
    m = re.search(r"totalResults>(\d+)<", r.text)
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-windows", type=int, default=3,
                    help="How many windows to cross-check against arXiv (0 to skip)")
    args = ap.parse_args()

    meta_path = RUNS / "backfill" / "backfill.meta.json"
    if not meta_path.exists():
        print("no backfill found; run `uc backfill --days 90` first")
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    start = datetime.date.fromisoformat(meta["start"])
    end = datetime.date.fromisoformat(meta["end"])

    days = load_daily_counts()
    all_days = [
        start + datetime.timedelta(days=i) for i in range((end - start).days + 1)
    ]
    counts = {d: days.get(str(d), 0) for d in all_days}
    values = sorted(counts.values())
    median = values[len(values) // 2] if values else 0

    zero_days = [d for d, n in counts.items() if n == 0]
    low_days = [d for d, n in counts.items() if 0 < n < median * LOW_DAY_FRACTION]

    # Windows the collector actually wrote raw responses for.
    raw = sorted(
        {
            re.sub(r"_p\d+\.xml$", "", Path(p).name)
            for p in glob.glob(str(RUNS / "backfill_*" / "raw" / "arxiv" / "*.xml"))
        }
    )
    window_days = int(cfg("arxiv.window_days", 7))
    expected_windows = -(-len(all_days) // window_days)

    errors = []
    for p in glob.glob(str(RUNS / "backfill_*" / "metrics.json")):
        errors += json.loads(Path(p).read_text(encoding="utf-8")).get("errors", [])
    skipped = [e for e in errors if "window" in e.lower() and "failed" in e.lower()]

    report: dict = {
        "range": [str(start), str(end)],
        "days_expected": len(all_days),
        "days_with_items": sum(1 for n in counts.values() if n > 0),
        "zero_item_days": [str(d) for d in zero_days],
        "thin_days": [f"{d} ({counts[d]})" for d in low_days],
        "per_day": {
            "median": median,
            "min": values[0] if values else 0,
            "max": values[-1] if values else 0,
            "mean": round(sum(values) / len(values), 1) if values else 0,
        },
        "total_items": sum(values),
        "raw_windows_written": len(raw),
        "raw_windows_expected": expected_windows,
        "skipped_window_errors": skipped,
    }

    if args.verify_windows:
        checks = []
        step = max(1, len(raw) // args.verify_windows)
        for name in raw[::step][: args.verify_windows]:
            w_start, w_end = (datetime.date.fromisoformat(s) for s in name.split("_"))
            stored = sum(
                counts.get(w_start + datetime.timedelta(days=i), 0)
                for i in range((w_end - w_start).days + 1)
            )
            upstream = arxiv_total_for(w_start, w_end)
            checks.append(
                {
                    "window": name,
                    "stored": stored,
                    "arxiv_total_results": upstream,
                    # Stored counts are de-duplicated by work_key; arXiv's total
                    # is raw hits, so stored <= upstream is the healthy direction.
                    "delta": upstream - stored,
                    "ok": upstream >= 0 and abs(upstream - stored) <= max(5, upstream * 0.02),
                }
            )
            time.sleep(float(cfg("arxiv.request_interval_s", 5.0)))
        report["window_cross_check"] = checks

    complete = (
        not zero_days
        and not skipped
        and len(raw) == expected_windows
        and all(c["ok"] for c in report.get("window_cross_check", []))
    )
    report["verdict"] = "COMPLETE" if complete else "GAPS_FOUND"

    out = RUNS / "backfill" / "coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))
    print(f"\nwritten to {out.relative_to(ROOT)}")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
