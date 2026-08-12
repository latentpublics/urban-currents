"""90-day backfill and quiet-day threshold calibration (PRD §5.6, Q2, Q3).

Two weeks of live operation gives 70-100 items. Picking a quantile from that is
statistically meaningless, so the threshold comes from a 90-day backfill instead
— thousands of scored items, from which we take the quantile that puts the
headline rate in the 30-50% band.

**The backfill does not summarise.** That is the single largest cost risk in
Phase 0 (PRD §10), and it is unnecessary: the headline score needs relevance,
badges, cluster size and overlay tags, all of which are available without an LLM
(the overlay comes from the rule-based vocabulary scan).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from . import paths, store
from .config import cfg, scoring_config
from .metrics import Run
from .models import Item
from .score.headline import score_item


def backfill_dir() -> Path:
    """Resolved per call, not captured at import — otherwise a test with a
    temporary UC_ROOT would write into the real repository."""
    return paths.RUNS / "backfill"


def _scores_path() -> Path:
    return backfill_dir() / "scores.jsonl"


def run_backfill(
    end: date,
    days: int = 90,
    sources: str = "arxiv",
    max_pages: Optional[int] = None,
) -> dict[str, Any]:
    """Collect, gate, classify and score a date range without summarising."""
    from .filters.classifier import score_items
    from .filters.gate import Gate, apply_gate
    from .dedup.merge import merge_candidates
    from .linking.vocab_match import Vocabulary, scan_text
    from .signals import apply_badges, apply_rule_signals

    start = end - timedelta(days=days - 1)
    run = Run(f"backfill_{start}_{end}", end)

    candidates: list[Item] = []
    if sources in ("all", "arxiv"):
        from .collectors.arxiv import ArxivCollector

        collector = ArxivCollector(run)
        candidates.extend(collector.collect(end, backfill_from=start, max_pages=max_pages))
    if sources in ("all", "openalex"):
        from .collectors.openalex import OpenAlexCollector

        try:
            oc = OpenAlexCollector(run)
            candidates.extend(oc.collect_journals(end, backfill_from=start, max_pages=200))
        except Exception as e:  # noqa: BLE001
            run.error(f"backfill openalex: {type(e).__name__}: {e}")

    run.count("arxiv_fetched", len(candidates))
    merged = merge_candidates(candidates, run_date=end).items
    kept, dropped = apply_gate(merged, Gate.from_vocab())
    run.count("after_dedup", len(merged))
    run.count("after_gate", len(kept))

    # Rejected items are kept on disk: the gate recall measurement (PRD §5.3)
    # samples from exactly this set.
    with (run.dir / "gate_rejected.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for it, reason in dropped:
            fh.write(
                json.dumps(
                    {
                        "work_key": it.work_key,
                        "reason": reason,
                        "title": it.bibliography.title,
                        "abstract": it.bibliography.abstract or "",
                        "categories": it.bibliography.categories,
                        "date": str(it.first_published or ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    pred = score_items(kept)
    vocabs = {f: Vocabulary.load(f) for f in ("methods", "data", "tools")}
    seen: set[str] = {
        e.id
        for item in store.iter_items()
        for e in item.entities.methods + item.entities.data + item.entities.tools
    }

    threshold = float(cfg("classifier.threshold", 0.5))
    backfill_dir().mkdir(parents=True, exist_ok=True)
    rows = []
    for item in kept:
        apply_rule_signals(item)
        apply_badges(item)
        text = f"{item.bibliography.title} {item.bibliography.abstract or ''}"
        for facet in ("methods", "data", "tools"):
            refs = scan_text(text, facet, vocabs[facet])
            if refs:
                setattr(item.entities, facet, refs)
        score_item(item, seen)
        rows.append(
            {
                "work_key": item.work_key,
                "date": str(item.first_published or ""),
                "relevance": item.scores.relevance,
                "headline": item.scores.headline,
                "components": item.scores.components.model_dump(),
                "categories": item.bibliography.categories,
                "source": "arxiv" if item.ids.arxiv else "journal",
                "selected": item.scores.relevance >= threshold,
            }
        )

    with _scores_path().open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    meta = {
        "start": str(start),
        "end": str(end),
        "days": days,
        "candidates": len(candidates),
        "after_dedup": len(merged),
        "after_gate": len(kept),
        "gate_rejected": len(dropped),
        "classifier_version": pred.version,
        "selection_threshold": threshold,
        "selected": sum(1 for r in rows if r["selected"]),
        "openalex_cost_usd": round(run.metrics.cost.openalex_usd, 6),
    }
    (backfill_dir() / "backfill.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    run.stage("backfill", "OK")
    run.save()
    return meta


def load_scores() -> list[dict]:
    p = _scores_path()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def daily_distribution(rows: list[dict], threshold: float) -> dict[str, Any]:
    """Q2: per-day count of items that clear the selection threshold."""
    per_day = Counter(r["date"] for r in rows if r["relevance"] >= threshold)
    all_days = sorted({r["date"] for r in rows if r["date"]})
    counts = sorted(per_day.get(d, 0) for d in all_days)
    return {
        "days_observed": len(all_days),
        "median_per_day": _quantile(counts, 0.5) if counts else 0,
        "p25_per_day": _quantile(counts, 0.25) if counts else 0,
        "p75_per_day": _quantile(counts, 0.75) if counts else 0,
        "min_per_day": counts[0] if counts else 0,
        "max_per_day": counts[-1] if counts else 0,
        "total_selected": sum(counts),
        "per_day": {d: per_day.get(d, 0) for d in all_days},
    }


def calibrate_threshold(
    target_low: float = 0.30, target_high: float = 0.50, apply: bool = False
) -> dict[str, Any]:
    """Q3: pick the headline threshold that lands the headline rate in-band.

    The rate is computed **per day** — the fraction of days that would carry a
    headline — not the fraction of items, because that is the thing the reader
    experiences.
    """
    rows = load_scores()
    if not rows:
        return {"status": "NO_DATA", "hint": "run `uc backfill --days 90` first"}

    threshold_sel = float(cfg("classifier.threshold", 0.5))
    selected = [r for r in rows if r["relevance"] >= threshold_sel]
    if not selected:
        return {"status": "NO_SELECTED_ITEMS", "n_scored": len(rows)}

    by_day: dict[str, float] = {}
    for r in selected:
        d = r["date"]
        if d:
            by_day[d] = max(by_day.get(d, 0.0), r["headline"])
    day_tops = sorted(by_day.values())
    if not day_tops:
        return {"status": "NO_DATED_ITEMS", "n_scored": len(rows)}

    target_mid = (target_low + target_high) / 2
    # A headline appears on a day when that day's top score clears the threshold,
    # so the threshold that yields rate R is the (1-R) quantile of daily tops.
    chosen = _quantile(day_tops, 1 - target_mid)
    rate = sum(1 for v in day_tops if v >= chosen) / len(day_tops)

    scores = sorted(r["headline"] for r in selected)
    hist_edges = [round(i / 20, 2) for i in range(21)]
    hist = Counter()
    for s in scores:
        bucket = min(19, int(s * 20))
        hist[round(bucket / 20, 2)] += 1

    result = {
        "status": "OK",
        "n_scored": len(rows),
        "n_selected": len(selected),
        "n_days": len(day_tops),
        "quantile": round(1 - target_mid, 4),
        "headline_threshold": round(chosen, 4),
        "headline_rate": round(rate, 4),
        "target_band": [target_low, target_high],
        "in_band": target_low <= rate <= target_high,
        "score_quantiles": {
            str(q): round(_quantile(scores, q), 4)
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "histogram": {str(e): hist.get(e, 0) for e in hist_edges[:-1]},
        "daily_distribution": daily_distribution(rows, threshold_sel),
    }

    if apply:
        _write_threshold(result)
        result["applied_to"] = str(paths.CONFIG / "scoring.yaml")

    backfill_dir().mkdir(parents=True, exist_ok=True)
    (backfill_dir() / "calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def _write_threshold(result: dict[str, Any]) -> None:
    """Rewrite the two calibration blocks in config/scoring.yaml in place.

    Line editing rather than a YAML round-trip: the comments in that file explain
    the weights, and re-serialising would throw them away.
    """
    path = paths.CONFIG / "scoring.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_calibration = False
    for line in lines:
        if line.startswith("headline_threshold:"):
            out.append(f"headline_threshold: {result['headline_threshold']}")
            continue
        if line.startswith("calibration:"):
            in_calibration = True
            out.append("calibration:")
            out.append(f"  quantile: {result['quantile']}")
            out.append(f"  headline_rate: {result['headline_rate']}")
            out.append(f"  n_scored: {result['n_selected']}")
            out.append(f"  n_days: {result['n_days']}")
            out.append(f"  calibrated_at: {date.today().isoformat()}")
            out.append('  source: "backfill"')
            continue
        if in_calibration:
            if line.startswith("  ") or not line.strip():
                continue
            in_calibration = False
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    scoring_config.cache_clear()
