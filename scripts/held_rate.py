"""How much would the suspicion rules hold back? (phase 0L, M2)

A held queue is only useful in a narrow band. Hold too little and it is theatre;
hold too much and it is not a filter but a different editorial policy adopted by
accident. The addendum sets the line at **30% of what a day would otherwise
publish** — past that, the rules are too wide and should be reported as such
rather than quietly shipped.

This runs the rules over the days that already have prepared candidates and
reports what each rule catches, without changing anything. Nothing is written to
`content/`; the rules are evaluated against stored stage output.

Usage:
    uv run python scripts/held_rate.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import held, paths  # noqa: E402
from pipeline.config import cfg  # noqa: E402
from pipeline.metrics import Run  # noqa: E402
from pipeline.models import Item  # noqa: E402


def _stage_items(run: Run, name: str) -> list[Item]:
    path = run.dir / "stages" / f"{name}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(Item.model_validate_json(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def days_with_candidates() -> list[date]:
    out = []
    for d in sorted((paths.RUNS).glob("run_*")):
        stem = d.name.replace("run_", "")
        try:
            day = date.fromisoformat(stem)
        except ValueError:
            continue
        if (d / "stages" / "select.jsonl").exists():
            out.append(day)
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    floor = float(cfg("selection.arxiv_floor", 0.80))
    rows = []
    totals = Counter()

    for day in days_with_candidates():
        run = Run.for_date(day)
        selected = _stage_items(run, "select")
        pool = _stage_items(run, "classify")
        if not selected:
            continue

        selected_keys = {it.work_key for it in selected}
        suspicions = []
        for it in selected:
            source = "arxiv" if it.work_key.startswith("arxiv:") else "journal"
            s = held.inspect(it, source, selected=True, floor=floor)
            if s:
                suspicions.append(s)
        for it in pool:
            if it.work_key in selected_keys or not it.work_key.startswith("arxiv:"):
                continue
            s = held.inspect(it, "arxiv", selected=False, floor=floor)
            if s:
                suspicions.append(s)

        withheld = [s for s in suspicions if s.kind == held.WITHHELD]
        near = [s for s in suspicions if s.kind == held.NEAR_MISS]
        would_publish = len(selected)
        rate = len(withheld) / would_publish if would_publish else None

        for s in suspicions:
            totals[s.rule] += 1
        totals["published"] += would_publish - len(withheld)
        totals["withheld"] += len(withheld)
        totals["near_miss"] += len(near)

        rows.append({
            "date": str(day),
            "selected": would_publish,
            "withheld": len(withheld),
            "publishes": would_publish - len(withheld),
            "near_miss": len(near),
            "withheld_rate": round(rate, 4) if rate is not None else None,
            "by_rule": dict(Counter(s.rule for s in suspicions)),
            "examples": [
                {"rule": s.rule, "score": s.score, "title": s.title[:70]}
                for s in withheld[:4]
            ],
        })

    overall = (
        totals["withheld"] / (totals["withheld"] + totals["published"])
        if (totals["withheld"] + totals["published"])
        else None
    )

    print(f"{'date':12} {'sel':>4} {'held':>5} {'pub':>4} {'near':>5}  rate")
    for r in rows:
        print(
            f"{r['date']:12} {r['selected']:>4} {r['withheld']:>5} "
            f"{r['publishes']:>4} {r['near_miss']:>5}  {r['withheld_rate']}"
        )
    print()
    print(f"overall withheld rate: {overall:.4f}" if overall is not None else "no days")
    print(f"by rule: {dict(totals)}")
    if overall is not None and overall > 0.30:
        print(
            "\n*** OVER 30% — the rules are too wide. Report this rather than "
            "shipping it: at this rate the queue is not a filter, it is a "
            "different editorial policy."
        )

    out = {
        "days": rows,
        "totals": dict(totals),
        "overall_withheld_rate": round(overall, 4) if overall is not None else None,
        "over_the_line": bool(overall is not None and overall > 0.30),
        "line": 0.30,
    }
    target = paths.RUNS / "held_rate.json"
    target.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\n→ {target}")


if __name__ == "__main__":
    main()
