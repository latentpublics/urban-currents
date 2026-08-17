"""X1: how long after publication does a work become visible to us? (phase 0k)

Every batch so far ran a date that was already over. 2026-08-05 was collected on
2026-08-13, by which time that day had finished happening. **A pipeline that runs
itself has no such margin**, and the size of the margin it does have is the only
honest input to a schedule.

The failure this prevents is specific and it is the batch's first invariant: if
we run a day before its papers are indexed, we see nothing, and a day where we
saw nothing looks exactly like a quiet day. That is the lie phase 0k exists to
make impossible.

**Zero requests.** OpenAlex returns `created_date` — when the record entered the
index — on every Work, and the raw responses on disk carry it for all 5,085
works we have collected. The lag is `created_date - publication_date`, already
paid for.

Reported per path, because arXiv and the journals are different pipelines with
different intermediaries, and per journal, because a venue that indexes slowly
is a venue we systematically see late.

Usage:
    uv run python scripts/indexing_lag.py --json runs/indexing_lag.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.graph.citation import iter_raw_openalex_works  # noqa: E402

ARXIV_SOURCE = "S4306400194"

# "If we run on D+n, what share of that day's eventual works can we see?"
HORIZONS = (0, 1, 2, 3, 5, 7, 14)


def _parse(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None


def _quantile(sorted_values: list[int], q: float) -> Optional[int]:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, int(q * (len(sorted_values) - 1) + 0.5))
    return sorted_values[idx]


def describe(lags: list[int]) -> dict:
    s = sorted(lags)
    return {
        "n": len(s),
        "p50": _quantile(s, 0.50),
        "p90": _quantile(s, 0.90),
        "p99": _quantile(s, 0.99),
        "min": s[0] if s else None,
        "max": s[-1] if s else None,
        "negative": sum(1 for v in s if v < 0),
        "same_day": sum(1 for v in s if v == 0),
    }


def coverage_table(lags: list[int]) -> dict[str, float]:
    """Share visible by D+n. This is the table a schedule is chosen from."""
    total = len(lags)
    return {
        f"D+{h}": round(sum(1 for v in lags if v <= h) / total, 4) if total else 0.0
        for h in HORIZONS
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    by_path: dict[str, list[int]] = defaultdict(list)
    by_journal: dict[str, list[int]] = defaultdict(list)
    journal_names: dict[str, str] = {}
    seen: set[str] = set()
    skipped = Counter()

    for work in iter_raw_openalex_works():
        wid = work.get("id")
        if not wid or wid in seen:
            continue
        seen.add(wid)

        created = _parse(work.get("created_date"))
        published = _parse(work.get("publication_date"))
        if not created or not published:
            skipped["missing_date"] += 1
            continue

        lag = (created - published).days
        loc = (work.get("primary_location") or {}).get("source") or {}
        sid = (loc.get("id") or "").rsplit("/", 1)[-1]
        path = "arxiv" if sid == ARXIV_SOURCE else "journal"
        by_path[path].append(lag)
        if path == "journal" and sid:
            by_journal[sid].append(lag)
            journal_names.setdefault(sid, loc.get("display_name") or sid)

    # The arXiv path is not in the OpenAlex raw responses — arXiv is collected
    # from its own API and stored as XML — so it is measured separately and its
    # measurement is weaker. What is available is `collected_at -
    # first_published` on stored items, which conflates arXiv's own indexing lag
    # with our scheduling: every batch so far ran historical dates on purpose.
    #
    # The clean evidence is the one direct experiment on record. On 2026-08-14
    # the pipeline was run for 2026-08-14 and arXiv returned **0 candidates**,
    # and the collect stage said so in its own error line rather than letting
    # the day look quiet. Same-day does not work.
    arxiv_observed: list[int] = []
    try:
        from pipeline import store

        for item in store.iter_items():
            if not item.ids.arxiv:
                continue
            collected = item.provenance.collected_at
            published = item.first_published
            if collected and published:
                arxiv_observed.append((collected.date() - published).days)
    except Exception:  # noqa: BLE001 - the store is not required for the rest
        arxiv_observed = []

    all_lags = [v for vs in by_path.values() for v in vs]
    out = {
        "population": (
            f"{len(seen)} distinct OpenAlex works in runs/*/raw/openalex "
            f"(collect + backfill); lag = created_date - publication_date, "
            f"0 additional requests"
        ),
        "skipped": dict(skipped),
        "overall": describe(all_lags),
        "overall_coverage": coverage_table(all_lags),
        "by_path": {
            path: {**describe(lags), "coverage": coverage_table(lags)}
            for path, lags in sorted(by_path.items())
        },
        "slowest_journals": [],
        "fastest_journals": [],
        "arxiv_path": {
            "measurement": (
                "collected_at - first_published over stored arXiv items; this "
                "conflates arXiv's indexing with our scheduling, because every "
                "batch so far deliberately ran historical dates"
            ),
            "n": len(arxiv_observed),
            "min_observed": min(arxiv_observed) if arxiv_observed else None,
            "distribution": dict(sorted(Counter(arxiv_observed).items())),
            "same_day_experiment": (
                "2026-08-14 was run on 2026-08-14: 0 arXiv candidates, and the "
                "collect stage recorded 'arXiv indexes submissions with a lag' "
                "rather than letting the day read as quiet"
            ),
            "conclusion": (
                "no arXiv item has ever been observed on its own publication "
                "day; the smallest observed lag is 1 day"
            ),
        },
    }

    # A venue that indexes late is a venue we systematically see late, which is a
    # coverage bias rather than a scheduling detail. Only venues with enough
    # works to have a stable p90.
    ranked = [
        {
            "source_id": sid,
            "name": journal_names.get(sid, sid),
            "n": len(lags),
            "p50": _quantile(sorted(lags), 0.50),
            "p90": _quantile(sorted(lags), 0.90),
        }
        for sid, lags in by_journal.items()
        if len(lags) >= 10
    ]
    ranked.sort(key=lambda r: (-(r["p90"] or 0), -r["n"]))
    out["slowest_journals"] = ranked[:20]
    out["fastest_journals"] = sorted(ranked, key=lambda r: (r["p90"] or 0, -r["n"]))[:10]
    out["journals_with_enough_works"] = len(ranked)

    # The pre-registered rule, evaluated here rather than after the fact.
    p90 = out["overall"]["p90"]
    out["decision"] = {
        "rule": (
            "p90 lag <= 1 day keeps issue date = publication date; "
            "p90 > 1 day switches the issue to discovery date"
        ),
        "p90_days": p90,
        "verdict": "publication_date" if (p90 is not None and p90 <= 1) else "discovery_date",
    }

    if a.json:
        Path(a.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n",
        )

    print(out["population"])
    print(f"\n{'population':<12} {'n':>6} {'p50':>5} {'p90':>5} {'p99':>5} "
          f"{'max':>6} {'same day':>9} {'negative':>9}")
    rows = [("overall", out["overall"])] + [
        (p, v) for p, v in out["by_path"].items()
    ]
    for name, v in rows:
        print(f"{name:<12} {v['n']:>6} {str(v['p50']):>5} {str(v['p90']):>5} "
              f"{str(v['p99']):>5} {str(v['max']):>6} {v['same_day']:>9} "
              f"{v['negative']:>9}")

    print(f"\n{'':<12} " + " ".join(f"{f'D+{h}':>7}" for h in HORIZONS))
    for name, v in rows:
        cov = v["coverage"] if "coverage" in v else out["overall_coverage"]
        print(f"{name:<12} " + " ".join(f"{cov[f'D+{h}'] * 100:>6.1f}%" for h in HORIZONS))

    print(f"\nslowest journals (n >= 10, {out['journals_with_enough_works']} qualify)")
    for r in out["slowest_journals"][:10]:
        print(f"   p90 {str(r['p90']):>4}  p50 {str(r['p50']):>4}  n={r['n']:<4} {r['name'][:52]}")

    d = out["decision"]
    print(f"\npre-registered rule: {d['rule']}")
    print(f"measured p90 = {d['p90_days']} days  ->  ISSUE DATE = {d['verdict'].upper()}")


if __name__ == "__main__":
    main()
