"""S2: resolve the part of the citation backlog that can become canon.

The daily stage caps itself at 40 requests so it cannot crowd out collection,
which leaves 137,584 references and about 68 days to drain. But draining all of
it was never the goal: of 139,540 distinct references only 23,362 are cited more
than once by our corpus, and the canon entry rule is exactly that. A work cited
once is one paper's reading list.

So this is a one-off that resolves the useful backlog and stops. It shares the
daily stage's resolved store and pending queue, so running it does not create a
second source of truth — it just fills the same file faster, and the daily stage
picks up from wherever this left off.

Resumable by construction: everything already in `canon_resolved.jsonl` is
skipped, so an interrupted run costs only the batch it was in the middle of.

Cost ceiling is a hard stop, not a warning. At the measured rate ($0.002 per
thousand ids) 23,362 references should cost about $0.05; anything near the
ceiling means the rate assumption is wrong and the run should end so a human can
find out why.

Usage:
    uv run python scripts/resolve_canon_backlog.py --min-citations 2 --max-cost 0.20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.graph.citation import load_reference_base  # noqa: E402
from pipeline.graph.daily_canon import (  # noqa: E402
    _append_resolved,
    _resolve,
    _write_pending,
    load_pending,
    load_resolved,
)

BATCH = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-citations", type=int, default=2)
    ap.add_argument("--max-cost", type=float, default=0.20)
    ap.add_argument("--max-seconds", type=int, default=3000)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    demand: dict[str, int] = {}
    for record in load_reference_base():
        for ref in record.get("referenced_works") or []:
            demand[ref] = demand.get(ref, 0) + 1

    known = set(load_resolved())
    wanted = [k for k, n in demand.items() if n >= a.min_citations and k not in known]
    # Most-cited first: an interrupted run should have resolved the works that
    # matter, not an arbitrary slice.
    wanted.sort(key=lambda k: (-demand[k], k))

    print(f"distinct references: {len(demand)}")
    print(f"cited >= {a.min_citations}: {sum(1 for n in demand.values() if n >= a.min_citations)}")
    print(f"already resolved: {len(known)}")
    print(f"to resolve now: {len(wanted)}")

    cost = 0.0
    done = 0
    started = time.monotonic()
    buffer: list[dict] = []

    for start in range(0, len(wanted), BATCH):
        if cost >= a.max_cost:
            print(f"\nSTOP: cost ${cost:.4f} reached the ${a.max_cost} ceiling")
            break
        if time.monotonic() - started > a.max_seconds:
            print(f"\nSTOP: {a.max_seconds}s elapsed; rerun to continue")
            break

        rows, batch_cost = _resolve(wanted[start : start + BATCH], batch=BATCH)
        cost += batch_cost
        buffer.extend(rows)
        done += len(rows)

        # Flushed periodically rather than at the end: an interrupted run should
        # keep what it paid for.
        if len(buffer) >= 500:
            _append_resolved(buffer)
            buffer = []
            print(f"  {done}/{len(wanted)} resolved, ${cost:.4f}")

    _append_resolved(buffer)

    resolved_now = set(load_resolved())
    still = [k for k in load_pending() if k not in resolved_now]
    still += [k for k in demand if k not in resolved_now and k not in set(still)]
    _write_pending(sorted(set(still), key=lambda k: (-demand.get(k, 0), k)))

    print(f"\nresolved this run: {done}")
    print(f"resolved total:    {len(resolved_now)}")
    print(f"pending after:     {len(still)}")
    print(f"openalex cost:     ${cost:.4f}  (ceiling ${a.max_cost})")


if __name__ == "__main__":
    main()
