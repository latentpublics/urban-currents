"""How far behind the citation base is (phase 0T, V2).

0k wrote that "a queue that only grows means the budget is too small and nobody
would notice otherwise". Nobody could: the number went into a run's metrics and
nowhere a person looks. It goes in `uc status` now — the command for coming back
after being away — so "is it draining" is answerable at a glance rather than by
reading a week of metrics files.
"""

from __future__ import annotations

from typing import Any


def counts() -> dict[str, Any]:
    from .graph.daily_canon import load_pending, load_resolved, load_unresolvable

    pending = len(load_pending())
    resolved = len(load_resolved())
    unresolvable = len(load_unresolvable())
    total = pending + resolved + unresolvable
    return {
        "resolved": resolved,
        "pending": pending,
        # Ids OpenAlex has no record for — merged, withdrawn, or never real.
        # Counted apart from `pending` so that a queue which stops shrinking
        # because what is left is unanswerable does not read as one that
        # stalled. Those are different problems with the same flat line.
        "unresolvable": unresolvable,
        "share_resolved": round(resolved / total, 4) if total else None,
    }
