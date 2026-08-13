"""Promote `unreadable` items that have since gained an abstract (phase 0d, Q5).

An item published under `Also published today` is a real record with no summary,
because no source had its abstract on the day it appeared. That can change: a
publisher deposits late, or a key arrives that opens a source we could not reach.
Until now the only way back was for the same paper to turn up in a later
collection window, which for a journal article means never.

**Published issues are immutable.** A past issue's `unreadable` list is not
edited — it recorded what was true that day and it stays true. A promoted item
enters the candidate pool from the promotion date, exactly as a preprint that
becomes a journal article does, and the change is recorded as a `status_change`
on the day it happens.

The first real use is the Springer key: 9 of the 97 currently unreadable items
are Springer DOIs whose abstracts that API still serves.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from . import store
from .collectors.abstracts import enrich_abstracts, needs_abstract, publisher_of
from .metrics import Run
from .models import Item


def unreadable_items(since: Optional[date] = None) -> list[Item]:
    """Items an issue published as unreadable and that still have no abstract.

    Deliberately re-checks the item rather than trusting the issue: one already
    promoted is no longer a candidate, and re-summarising it would cost a call
    to learn what the file already says.
    """
    keys: dict[str, date] = {}
    for issue in store.iter_issues():
        if since and issue.date < since:
            continue
        for wk in issue.unreadable:
            keys.setdefault(wk, issue.date)

    out = []
    for wk in sorted(keys):
        item = store.load_item(wk)
        if item is not None and needs_abstract(item):
            out.append(item)
    return out


def promote(
    d: date,
    since: Optional[date] = None,
    use_llm: bool = True,
    enricher=None,
    sources: tuple[str, ...] = ("crossref", "springer"),
) -> dict[str, Any]:
    """Re-enrich unreadable items; summarise and update the ones that now have one.

    Returns counts. Items that gain nothing are left exactly as they were, so
    running this daily against a source that never opens costs requests and
    changes no bytes.
    """
    run = Run.for_date(d)
    candidates = unreadable_items(since)
    if not candidates:
        return {"date": str(d), "status": "NO_CANDIDATES", "candidates": 0}

    counts = enrich_abstracts(candidates, run, enricher=enricher, sources=sources)
    recovered = [it for it in candidates if not needs_abstract(it)]

    summarised = 0
    if recovered and use_llm:
        from .linking.pipeline import link_items
        from .run_stages import reconcile_places_status
        from .signals import apply_badges, apply_rule_signals
        from .summarize.run import summarize_items

        link_items(recovered, run, use_llm=True)
        stats = summarize_items(recovered, run, use_llm=True)
        summarised = stats.get("summarized", 0)
        for item in recovered:
            apply_rule_signals(item)
            apply_badges(item)
        reconcile_places_status(recovered)

    for item in recovered:
        store.save_item(item, today=d)

    by_publisher: dict[str, int] = {}
    for item in recovered:
        name = publisher_of(item)
        by_publisher[name] = by_publisher.get(name, 0) + 1

    run.count("promoted", len(recovered))
    run.stage("promote", "OK")
    run.save()

    return {
        "date": str(d),
        "status": "OK",
        "candidates": len(candidates),
        "recovered": len(recovered),
        "summarised": summarised,
        "by_source": {
            k: v for k, v in counts.items() if k in ("crossref", "springer_api", "none")
        },
        "by_publisher": dict(sorted(by_publisher.items())),
        "work_keys": sorted(it.work_key for it in recovered),
        # The promoted items join the candidate pool from here; past issues are
        # untouched, and `stage_select` picks them up on the next run.
        "note": (
            "Promoted items enter the next run's candidate pool. Published "
            "issues are immutable and their unreadable lists are unchanged."
        ),
    }
