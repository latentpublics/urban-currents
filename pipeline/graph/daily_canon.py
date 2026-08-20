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
UNRESOLVABLE_FILE = "canon_unresolvable.jsonl"

# How many times an id may come back empty before it is parked.
#
# **Measured in 0T**: a batch of 3,000 returned 2,548 works — 452 ids, 15%, that
# OpenAlex has no record for. Merged works, withdrawn records, ids that only
# ever existed in somebody's reference list. Without this they return to the
# head of the queue every day forever, because the queue is ordered by how often
# our corpus cites them and a much-cited dead id sorts first.
#
# Three, not one: an empty response can also mean a bad minute at the API, and
# parking a live id on one miss would quietly shrink the canon.
MAX_ATTEMPTS = 3

# OpenAlex resolves 50 ids in one request. Measured in 0T: every request came
# back with all 50, so batching is real and the queue costs requests/50.
BATCH = 50

# Time kept back from the day's deadline. The chore stops with this much left so
# that a run which publishes fine is never recorded as interrupted because a
# background task was still going when the clock ran out.
SAFETY_MARGIN_S = 120.0


def _seconds_left(deadline: Any) -> float:
    """How long the day's clock has, or infinity if there is no clock.

    `Deadline.remaining` is a **property**, not a method — reading it through a
    `callable()` test silently returned infinity, which would have let this
    chore run past the deadline it exists to respect. Both shapes are handled,
    and an unrecognised object means "no clock" rather than a crash.

    A SIGTERM already received counts as no time left. The platform is about to
    kill us and the right move is to write the queue and stop, not to start
    another request.
    """
    if deadline is None:
        return float("inf")
    if getattr(deadline, "signalled", False):
        return 0.0
    value = getattr(deadline, "remaining", None)
    if value is None:
        value = getattr(deadline, "seconds_left", None)
    if callable(value):
        try:
            value = value()
        except Exception:  # noqa: BLE001
            return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _openalex_spent(run: Run) -> float:
    """What this run has already spent at OpenAlex, before the chore starts.

    `metrics.cost` is a model with attributes, not a dict — reading it with
    `.get()` returned 0.0 for every run, which would have handed this chore the
    full share on a day collection had already spent most of it.
    """
    cost = getattr(run.metrics, "cost", None)
    if cost is None:
        return 0.0
    value = getattr(cost, "openalex_usd", None)
    if value is None and isinstance(cost, dict):
        value = cost.get("openalex_usd")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


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


def load_attempts() -> dict[str, int]:
    """How many times each pending id has been asked for and not answered."""
    p = _pending_path()
    if not p.exists():
        return {}
    return {
        r["openalex_id"]: int(r.get("attempts") or 0)
        for r in store.read_jsonl(p)
        if r.get("openalex_id")
    }


def load_unresolvable() -> set[str]:
    p = paths.STATE / UNRESOLVABLE_FILE
    if not p.exists():
        return set()
    return {
        r["openalex_id"] for r in store.read_jsonl(p) if r.get("openalex_id")
    }


def _write_pending(ids: list[str], attempts: Optional[dict[str, int]] = None) -> None:
    attempts = attempts or {}
    _pending_path().parent.mkdir(parents=True, exist_ok=True)
    store.write_text_atomic(
        _pending_path(),
        "\n".join(
            store.jsonl_line(
                {"openalex_id": i, "attempts": attempts[i]} if attempts.get(i)
                else {"openalex_id": i}
            )
            for i in sorted(set(ids))
        )
        + ("\n" if ids else ""),
    )


def _park(ids: set[str], attempts: dict[str, int]) -> None:
    """Ids OpenAlex will not answer for. Recorded, not deleted.

    They stay countable so "the queue is draining" can be told apart from "the
    queue has stopped shrinking because what is left is unanswerable" — two
    very different reports from the same falling number.
    """
    if not ids:
        return
    p = paths.STATE / UNRESOLVABLE_FILE
    existing = load_unresolvable()
    rows = sorted(existing | ids)
    p.parent.mkdir(parents=True, exist_ok=True)
    store.write_text_atomic(
        p,
        "\n".join(
            store.jsonl_line({"openalex_id": i, "attempts": attempts.get(i, MAX_ATTEMPTS)})
            for i in rows
        )
        + "\n",
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
    d: date,
    run: Optional[Run] = None,
    max_ids: Optional[int] = None,
    deadline: Any = None,
) -> dict[str, Any]:
    """Fold the day into the reference base and resolve what the budget allows.

    `deadline` is the daily run's own clock, when there is one. This stops on
    **whichever of three limits arrives first** — money, its own wall clock, or
    the day's remaining time — and leaves the rest in the queue. Stopping is a
    normal ending here; the queue is the mechanism for being behind.
    """
    run = run or Run.for_date(d)

    stats = build_reference_base()
    records = load_reference_base()

    known = set(load_resolved())
    parked = load_unresolvable()
    attempts = load_attempts()
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
    queue = [i for i in load_pending() if i not in known and i not in parked]
    queue += sorted(wanted - known - parked - set(queue))
    queue.sort(key=lambda i: (-demand.get(i, 0), i))

    daily = float(cfg("openalex.daily_budget_usd", 1.0))
    fraction = float(cfg("canon.daily_budget_fraction", 0.5))
    spent_today = _openalex_spent(run)
    # **What is left of the share, not the whole share.** Collection and
    # enrichment run first and their spend is already on the clock; taking the
    # fraction of the gross budget would let this quietly exceed it on a day
    # when collection was expensive.
    budget = max(0.0, daily * fraction - spent_today)

    # Measured in 0T against the live API: **50 ids per request, $0.0001 per
    # request, 1.70s per request** — so $0.000002 an id and 34ms an id.
    #
    # At that price the whole 116,630-id queue costs **$0.23**, well inside one
    # day's $1.00. **Cost is not the constraint; wall clock is.** The same queue
    # is 2,333 requests and 66 minutes, which is more than the entire
    # `daily.max_minutes` budget of 30 — so it cannot be done in one run however
    # much money is left, and the per-run limit has to be time.
    per_request_usd = 0.0001
    per_request_s = 1.7
    by_cost = int(budget / per_request_usd) * BATCH

    seconds = float(cfg("canon.max_seconds_per_run", 600))
    if deadline is not None:
        # Never spend the day's last minutes on a chore. A run killed here would
        # be killed *after* publishing, but it would still be recorded as
        # interrupted, and an issue that went out fine deserves better than a
        # failure row for a background task.
        left = _seconds_left(deadline) - SAFETY_MARGIN_S
        seconds = min(seconds, max(0.0, left))
    by_time = int(seconds / per_request_s) * BATCH

    cap = max_ids if max_ids is not None else max(0, min(by_cost, by_time))
    limit = "explicit" if max_ids is not None else (
        "cost" if by_cost <= by_time else "time"
    )

    batch = queue[:cap]
    resolved, cost, stopped = _resolve(batch, deadline=deadline, budget_usd=budget)
    _append_resolved(resolved)

    # **Built once.** This was a set comprehension inside the condition, which
    # Python re-evaluates for every element: 116,630 ids each rebuilding a
    # 17,000-entry set is around two billion operations, and the first live run
    # sat in it long enough to look hung. The resolved ids never change during
    # the loop, so the set is hoisted.
    done = {r["openalex_id"] for r in resolved}
    # Only ids we actually asked about this run can have failed this run.
    missed = [i for i in batch if i not in done]
    for i in missed:
        attempts[i] = attempts.get(i, 0) + 1
    give_up = {i for i in missed if attempts[i] >= MAX_ATTEMPTS}
    _park(give_up, attempts)

    still_pending = [i for i in queue if i not in done and i not in give_up]
    _write_pending(still_pending, attempts)

    run.count("canon_references", stats["reference_mentions"])
    run.count("canon_resolved_total", len(known) + len(resolved))
    # The number that matters over time: if this only ever grows, the budget is
    # too small and the canon is falling behind its own input.
    run.count("canon_pending", len(still_pending))
    run.count("canon_unresolvable", len(parked | give_up))
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
        "missed_this_run": len(missed),
        "parked_now": len(give_up),
        "unresolvable_total": len(parked | give_up),
        "budget_usd": round(budget, 4),
        "id_cap": cap,
        "cap_set_by": limit,
        # Which limit actually ended the run, as opposed to which one set the
        # cap. `budget` and `deadline` are both **normal endings**.
        "stopped_by": stopped,
        "openalex_cost_usd": round(cost, 6),
    }


def _resolve(
    ids: list[str],
    batch: int = BATCH,
    deadline: Any = None,
    budget_usd: Optional[float] = None,
) -> tuple[list[dict], float, str]:
    """Resolve ids in batches, stopping on time, money or a rate limit.

    Returns what was resolved **and why it stopped**. Every stopping reason
    here is a normal ending: the caller writes the remainder back to the queue
    and tomorrow starts from it. Nothing in this function is an alert.
    """
    if not ids:
        return [], 0.0, "nothing to do"

    # Checked **before** the client is built. There is no point configuring an
    # API client for work there is no time or money left to do, and doing it in
    # this order also keeps the two "normal ending" paths together.
    if deadline is not None and _seconds_left(deadline) <= SAFETY_MARGIN_S:
        return [], 0.0, "deadline"
    if budget_usd is not None and budget_usd <= 0:
        return [], 0.0, "budget"

    from ..collectors.openalex import (
        OpenAlexBudgetExhausted,
        OpenAlexUnavailable,
        configure_pyalex,
    )

    try:
        pyalex = configure_pyalex()
    except OpenAlexUnavailable:
        # No key. **Not a failure**: this chore is optional and a run without
        # OpenAlex credentials is a normal state — CI has none. Recording it as
        # FAILED every day would train everyone to ignore the field.
        return [], 0.0, "no OpenAlex client"
    if pyalex is None:
        return [], 0.0, "no OpenAlex client"

    bare = [i.split(":", 1)[1] if ":" in i else i for i in ids]
    out: list[dict] = []
    cost = 0.0
    stopped = "queue drained"
    for start in range(0, len(bare), batch):
        if budget_usd is not None and cost >= budget_usd:
            stopped = "budget"
            break
        if deadline is not None and _seconds_left(deadline) <= SAFETY_MARGIN_S:
            stopped = "deadline"
            break
        chunk = bare[start : start + batch]
        try:
            results, meta = (
                pyalex.Works().filter(openalex_id="|".join(chunk))
                .get(per_page=batch, return_meta=True)
            )
        except OpenAlexBudgetExhausted:
            # 0N (P0) made a 429 surface immediately instead of sleeping on a
            # 12-hour Retry-After. Here that is simply the end of today's share.
            stopped = "rate limited"
            break
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
    return out, cost, stopped
