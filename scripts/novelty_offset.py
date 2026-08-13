"""Measure the gap between backfill novelty and live novelty (roadmap §2.5, D16).

The quiet-day threshold is calibrated on the 90-day backfill, and the backfill
does not summarise — so its overlay tags come only from the rule-based
vocabulary scan, while a live day also gets LLM-extracted tags. The ``novelty``
component of the headline score is computed from exactly those tags.

If LLM extraction finds more tags than the scan, live novelty runs higher than
the distribution the threshold was drawn from, and the headline rate drifts
above the 30-50% target. This script measures that offset on the days that do
have summaries, so the report can state a number instead of a worry. If it is
negligible, saying so is a result.

**And it measures the larger gap, which is not about tags at all.** Novelty is
measured against the archive, and the archive the threshold was calibrated on
was 90 days deep by the end of the replay, while `content/` currently holds a
handful of days. A young archive makes almost every tag fresh. That shift is
several times the LLM one, it decays on its own as days accumulate, and without
a number next to it the first weeks of a 100% headline rate look like a bug in
the threshold rather than a property of a new archive.

Usage:
    uv run python scripts/novelty_offset.py --dates 2026-08-05 2026-08-06 …
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import store  # noqa: E402
from pipeline.linking.vocab_match import Vocabulary, scan_text  # noqa: E402
from pipeline.metrics import Run  # noqa: E402
from pipeline.paths import RUNS  # noqa: E402
from pipeline.score.headline import novelty  # noqa: E402
from pipeline.stages import read_stage  # noqa: E402

FACETS = ("methods", "data", "tools")


def rule_only_tags(item, vocabs) -> list[str]:
    text = f"{item.bibliography.title} {item.bibliography.abstract or ''}"
    ids: list[str] = []
    for facet in FACETS:
        ids += [r.id for r in scan_text(text, facet, vocabs[facet])]
    return ids


def _archive_maturity(
    rows: list[dict], dates: list[date], seen: set[str], weight: float, n_measured: int
) -> dict:
    """Live novelty against the same days in the 90-day replay.

    Same days, same items, same scoring code — the only difference is how much
    archive stood behind them. The replay had ~2,000 published items by August
    and `content/` has a few dozen, so this isolates the maturity effect from
    the LLM one measured above.
    """
    scores = RUNS / "backfill" / "scores.jsonl"
    if not scores.exists():
        return {"status": "NO_BACKFILL", "hint": "run `uc backfill --days 90` first"}

    wanted = {str(d) for d in dates}
    replay, earlier_published = [], 0
    first_day = min(wanted)
    for line in scores.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("published"):
            continue
        if r.get("date") in wanted:
            replay.append(float((r.get("components") or {}).get("novelty", 0.0)))
        elif str(r.get("date", "")) < first_day:
            earlier_published += 1
    if not replay:
        return {"status": "DATES_NOT_IN_BACKFILL", "dates": sorted(wanted)}

    live_mean = statistics.fmean(r["novelty_live"] for r in rows)
    replay_mean = statistics.fmean(replay)
    delta = live_mean - replay_mean
    return {
        "status": "OK",
        # Every item on the measured days, not just the ones that carried a tag:
        # the rest are still part of the archive those days were scored against.
        "archive_items_live": len(list(store.iter_items())) - n_measured,
        "archive_items_replay": earlier_published,
        "distinct_tags_live": len(seen),
        "novelty_live_mean": round(live_mean, 4),
        "novelty_replay_mean": round(replay_mean, 4),
        "mean_delta": round(delta, 4),
        "headline_mean_shift": round(delta * weight, 4),
        "note": (
            "Decays on its own as content/ fills. Until it does, the live "
            "headline rate runs above the calibrated 30-50% band and that is "
            "the archive being young, not the threshold being wrong."
        ),
    }


def measure(dates: list[date]) -> dict:
    vocabs = {f: Vocabulary.load(f) for f in FACETS}

    # The days being measured are already published, so their own tags are in
    # content/. Counting them as "previously seen" forces live novelty to zero —
    # the same self-reference that broke idempotency in stage_score. Exclude the
    # measured items, exactly as scoring does.
    measured_keys: set[str] = set()
    staged: dict[date, list] = {}
    for d in dates:
        run = Run.for_date(d)
        items = read_stage(run, "summarize") or read_stage(run, "select")
        staged[d] = items
        measured_keys |= {it.work_key for it in items}

    seen = {
        e.id
        for it in store.iter_items()
        if it.work_key not in measured_keys
        for e in it.entities.methods + it.entities.data + it.entities.tools
    }

    rows = []
    for d in dates:
        items = staged[d]
        for item in items:
            live_ids = [
                e.id
                for e in item.entities.methods + item.entities.data + item.entities.tools
            ]
            if not live_ids:
                continue
            rule_ids = rule_only_tags(item, vocabs)

            live_novelty = novelty(item, seen)
            fresh_rule = sum(1 for i in set(rule_ids) if i not in seen)
            rule_novelty = round(fresh_rule / len(set(rule_ids)), 4) if rule_ids else 0.0

            rows.append(
                {
                    "date": str(d),
                    "work_key": item.work_key,
                    "n_tags_live": len(set(live_ids)),
                    "n_tags_rule_only": len(set(rule_ids)),
                    "novelty_live": live_novelty,
                    "novelty_rule_only": rule_novelty,
                    "delta_novelty": round(live_novelty - rule_novelty, 4),
                    "has_summary": bool(item.summary.en and item.summary.en.what),
                }
            )

    if not rows:
        return {"status": "NO_DATA", "hint": "run the pipeline with summaries first"}

    deltas = [r["delta_novelty"] for r in rows]
    tag_delta = [r["n_tags_live"] - r["n_tags_rule_only"] for r in rows]
    # novelty carries a 0.20 weight in the headline score (config/scoring.yaml).
    weight = 0.20
    mean_delta = statistics.fmean(deltas)

    result = {
        "status": "OK",
        "dates": [str(d) for d in dates],
        "items": len(rows),
        "with_summary": sum(1 for r in rows if r["has_summary"]),
        "tags_per_item": {
            "live_mean": round(statistics.fmean(r["n_tags_live"] for r in rows), 2),
            "rule_only_mean": round(statistics.fmean(r["n_tags_rule_only"] for r in rows), 2),
            "mean_extra_from_llm": round(statistics.fmean(tag_delta), 2),
        },
        "novelty": {
            "live_mean": round(statistics.fmean(r["novelty_live"] for r in rows), 4),
            "rule_only_mean": round(statistics.fmean(r["novelty_rule_only"] for r in rows), 4),
            "mean_delta": round(mean_delta, 4),
            "median_delta": round(statistics.median(deltas), 4),
            "max_abs_delta": round(max(abs(x) for x in deltas), 4),
        },
        "headline_score_offset": {
            "novelty_weight": weight,
            "mean_shift": round(mean_delta * weight, 4),
            "note": (
                "How much of the live-vs-backfill headline gap the LLM tags "
                "explain. Compare with archive_maturity below, which is larger."
            ),
        },
        "archive_maturity": _archive_maturity(
            rows, dates, seen, weight, len(measured_keys)
        ),
        "rows": rows[:50],
    }
    out = RUNS / "backfill" / "novelty_offset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True)
    a = ap.parse_args()
    result = measure([date.fromisoformat(s) for s in a.dates])
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
