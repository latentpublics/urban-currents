"""Measure what the keyword gate throws away (PRD §5.3).

The gate exists to keep cs.LG / cs.CV / cs.AI volume manageable, and it is
allowed to be crude — but only because its recall gets measured. Draw 200 items
at random from the rejected set, run the classifier over them, and count how
many clear the selection threshold. **More than 3 means the gate is too narrow**
and the keyword list has to widen.

An unmeasured gate is not a filter, it is a silent loss.
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path
from typing import Any, Optional

from . import paths
from .config import cfg
from .models import Bibliography, Item

MAX_ACCEPTABLE_MISSES = 3


def _rejected_files() -> list[Path]:
    """Rejected-item logs. The backfill log comes first — it is far larger, and
    the daily logs only carry titles, not the abstracts the classifier needs."""
    return sorted(paths.RUNS.glob("backfill_*/gate_rejected.jsonl")) + sorted(
        paths.RUNS.glob("run_*/gate_dropped.jsonl")
    )


def load_rejected() -> list[dict]:
    rows: list[dict] = []
    for p in _rejected_files():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _to_item(row: dict) -> Optional[Item]:
    title = row.get("title")
    if not title:
        return None
    try:
        return Item(
            work_key=row.get("work_key") or "arxiv:0000.00000",
            bibliography=Bibliography(
                title=title,
                abstract=row.get("abstract") or "",
                categories=row.get("categories") or [],
            ),
        )
    except Exception:
        return None


def measure_gate_recall(
    sample: int = 200, run_date: Optional[date] = None, seed: int = 42
) -> dict[str, Any]:
    from .filters.classifier import score_items

    rows = load_rejected()
    if not rows:
        return {
            "status": "NO_DATA",
            "hint": "run `uc backfill` or a daily run first — the gate log is written there",
        }

    rng = random.Random(seed)
    drawn = rng.sample(rows, min(sample, len(rows)))
    items = [it for it in (_to_item(r) for r in drawn) if it is not None]
    pred = score_items(items)

    threshold = float(cfg("classifier.threshold", 0.5))
    misses = [
        {
            "work_key": it.work_key,
            "score": it.scores.relevance,
            "title": it.bibliography.title[:110],
        }
        for it in items
        if it.scores.relevance >= threshold
    ]
    misses.sort(key=lambda m: -m["score"])

    result = {
        "status": "OK",
        "rejected_pool": len(rows),
        "sampled": len(items),
        "threshold": threshold,
        "classifier_version": pred.version,
        "above_threshold": len(misses),
        "max_acceptable": MAX_ACCEPTABLE_MISSES,
        "verdict": "GATE_OK" if len(misses) <= MAX_ACCEPTABLE_MISSES else "GATE_TOO_NARROW",
        "estimated_daily_loss": (
            round(len(misses) / len(items) * len(rows) / max(1, _distinct_days(rows)), 2)
            if items
            else 0
        ),
        "misses": misses[:20],
    }
    out = paths.RUNS / "gate_recall.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def _distinct_days(rows: list[dict]) -> int:
    return len({r.get("date") for r in rows if r.get("date")}) or 1
