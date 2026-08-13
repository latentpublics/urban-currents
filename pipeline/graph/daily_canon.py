"""Accumulate the canon a day at a time, inside the OpenAlex budget (R3).

The 90-day harvest was a one-off: it read responses already on disk. From here
the base grows by a day at a time, and the only recurring cost is resolving the
metadata of newly seen reference IDs.

Two properties this needs and the batch harvest did not:

**A budget that cannot starve the pipeline.** Resolution takes at most
`canon.daily_budget_fraction` of the day's OpenAlex allowance — a fifth by
default. The collectors need the rest, and a canon that crowds out collection
has its priorities backwards.

**A queue that admits it is behind.** What the budget could not resolve stays in
`runs/state/canon_pending.jsonl` and is tried first tomorrow. The queue length
goes into metrics, because a queue that grows every day means the budget is too
small and nobody would notice otherwise.

Idempotent: running a date twice adds nothing, because the reference base is
keyed by work_key and rebuilt from source rather than appended to blindly.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

from .. import paths, store
from ..config import cfg
from ..metrics import Run
from .citation import build_reference_base, load_reference_base

PENDING_FILE = "canon_pending.jsonl"
RESOLVED_FILE = "canon_resolved.jsonl"


def _pending_path() -> Path:
    return paths.STATE / PENDING_FILE


def _resolved_path() -> Path:
    return paths.STATE / RESOLVED_FILE


def load_resolved() -> dict[str, dict]:
    """Unparseable lines are skipped, not fatal.

    A store that one bad line makes unreadable loses everything to a single
    interrupted write — which is exactly what happened when a title containing
    U+2028 split a record in two (see `store.jsonl_line`).
    """
    bad: list = []
    rows = store.read_jsonl(_resolved_path(), on_error=bad)
    if bad:
        print(f"canon_resolved.jsonl: skipped {len(bad)} unparseable line(s)")
    return {r["openalex_id"]: r for r in rows if r.get("openalex_id")}


def load_pending() -> list[str]:
    p = _pending_path()
    if not p.exists():
        return []
    return [
        r["openalex_id"] for r in store.read_jsonl(p) if r.get("openalex_id")
    ]


def _write_pending(ids: list[str]) -> None:
    _pending_path().parent.mkdir(parents=True, exist_ok=True)
    store.write_text_atomic(
        _pending_path(),
        "\n".join(store.jsonl_line({"openalex_id": i}) for i in sorted(set(ids)))
        + ("\n" if ids else ""),
    )


def _append_resolved(rows: list[dict]) -> None:
    if not rows:
        return
    existing = load_resolved()
    for row in rows:
        existing[row["openalex_id"]] = row
    store.write_text_atomic(
        _resolved_path(),
        "\n".join(store.jsonl_line(existing[k]) for k in sorted(existing)) + "\n",
    )


def accumulate_day(
    d: date, run: Optional[Run] = None, max_ids: Optional[int] = None
) -> dict[str, Any]:
    """Fold the day into the reference base and resolve what the budget allows."""
    run = run or Run.for_date(d)

    stats = build_reference_base()
    records = load_reference_base()

    known = set(load_resolved())
    # How often our corpus cites each reference. The queue is drained in that
    # order, because a work cited once may never matter and a work cited forty
    # times is the canon — resolving in arbitrary order would spend days on the
    # tail before touching the head. Measured here: 139,540 distinct references
    # of which 23,362 are cited more than once.
    demand: dict[str, int] = {}
    for record in records:
        for ref in record.get("referenced_works") or []:
            demand[ref] = demand.get(ref, 0) + 1

    wanted = set(demand)
    queue = [i for i in load_pending() if i not in known]
    queue += sorted(wanted - known - set(queue))
    queue.sort(key=lambda i: (-demand.get(i, 0), i))

    daily = float(cfg("openalex.daily_budget_usd", 1.0))
    fraction = float(cfg("canon.daily_budget_fraction", 0.2))
    budget = daily * fraction
    # Measured in phase 0d: 200 ids cost $0.0004, about $0.002 per thousand. On
    # cost alone a fifth of the day budget buys 50,000 ids — which is 1,000
    # requests and the better part of an hour. Cost is not the binding
    # constraint here, wall clock is, so the cap is the lower of the two and the
    # request ceiling is what actually bites. A day produces a few thousand new
    # references, so the queue still drains; it just drains over several days
    # from a standing start, which is what the pending queue is for.
    per_id = 0.002 / 1000
    by_cost = int(budget / per_id / 2)
    by_requests = int(cfg("canon.max_requests_per_run", 40)) * 50
    cap = max_ids if max_ids is not None else min(by_cost, by_requests)

    batch = queue[:cap]
    resolved, cost = _resolve(batch)
    _append_resolved(resolved)

    still_pending = [i for i in queue if i not in {r["openalex_id"] for r in resolved}]
    _write_pending(still_pending)

    run.count("canon_references", stats["reference_mentions"])
    run.count("canon_resolved_total", len(known) + len(resolved))
    # The number that matters over time: if this only ever grows, the budget is
    # too small and the canon is falling behind its own input.
    run.count("canon_pending", len(still_pending))
    run.add_cost("openalex_usd", cost)
    run.stage("canon", "OK")
    run.save()

    return {
        "date": str(d),
        "status": "OK",
        "reference_base": stats,
        "already_resolved": len(known),
        "queue_before": len(queue),
        "attempted": len(batch),
        "resolved_now": len(resolved),
        "pending_after": len(still_pending),
        "budget_usd": round(budget, 4),
        "id_cap": cap,
        "openalex_cost_usd": round(cost, 6),
    }


def _resolve(ids: list[str], batch: int = 50) -> tuple[list[dict], float]:
    if not ids:
        return [], 0.0
    from ..collectors.openalex import configure_pyalex

    pyalex = configure_pyalex()
    if pyalex is None:
        return [], 0.0

    bare = [i.split(":", 1)[1] if ":" in i else i for i in ids]
    out: list[dict] = []
    cost = 0.0
    for start in range(0, len(bare), batch):
        chunk = bare[start : start + batch]
        results, meta = (
            pyalex.Works().filter(openalex_id="|".join(chunk))
            .get(per_page=batch, return_meta=True)
        )
        cost += float((meta or {}).get("cost_usd") or 0.0)
        for w in results:
            wid = (w.get("id") or "").rsplit("/", 1)[-1]
            pt = w.get("primary_topic") or {}
            loc = (w.get("primary_location") or {}).get("source") or {}
            out.append({
                "openalex_id": f"openalex:{wid}",
                "title": w.get("display_name"),
                "year": w.get("publication_year"),
                "publication_date": w.get("publication_date"),
                "venue": loc.get("display_name"),
                "venue_id": (loc.get("id") or "").rsplit("/", 1)[-1] or None,
                "topic": pt.get("display_name"),
                "topic_id": (pt.get("id") or "").rsplit("/", 1)[-1] or None,
                "subfield": (pt.get("subfield") or {}).get("display_name"),
                "subfield_id": ((pt.get("subfield") or {}).get("id") or "").rsplit("/", 1)[-1] or None,
                "authors": [
                    (a.get("author") or {}).get("display_name")
                    for a in (w.get("authorships") or [])[:5]
                ],
                "cited_by_count": w.get("cited_by_count"),
            })
    return out, cost
