"""Why does `tag shift` print nothing? (phase 0N, P3)

It prints nothing on the days anyone looks at, and the reason is not the
deviation arithmetic — **it is that those days have no archive behind them.**

Walking every issue and printing its baseline length beside its result:

    2026-06-19 .. 2026-07-01   baseline 7-19 days   status OK    6 deviations
    2026-08-05 .. 2026-08-11   baseline 0-6  days   NO_BASELINE  0

The backfill filled 2026-06-12 to 07-01. The five original issues are in
August. Between them sits a **33-day gap**, so 2026-08-05's 30-day lookback
window — 07-06 to 08-05 — contains **zero** archive days. The section reports
`NO_BASELINE` and that is the correct answer to the question it was asked.

Twenty consecutive days do exist. They are simply a month away from the days
whose synthesis anyone reads, and a baseline cannot reach across a hole.

So: not a bug, and the remedy is not a code change but the rest of the backfill,
which closes the gap by walking forward from 07-02.

The count on days that *do* have a baseline is **6 deviations over 13 days** —
low, and expected: overlay tags run about 1.2 per item, so clearing "at least 3
today and at least 3x the daily average" is genuinely rare. The docstring on
`deviations` says exactly this — the section is bounded by the vocabulary, not
by the method — and this measures how much the pending vocabulary curation is
worth.

Usage:
    uv run python scripts/tag_shift_trace.py
"""

import sys
import json
from collections import Counter
from datetime import date
sys.path.insert(0, r"C:/Users/jour/Documents/GitHub/urban-currents")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import store  # noqa: E402  - sys.path is set two lines up
from pipeline.synthesis import deviations, _tags_of  # noqa: E402

issues = sorted((store.paths.CONTENT / "issues").glob("*.json"))
print(f"{'date':12} {'items':>5} {'base':>5} {'status':13} {'found':>5} {'distinct':>8} {'max_today':>9}")
for p in issues:
    d = json.loads(p.read_text(encoding="utf-8"))
    day = date.fromisoformat(d["date"])
    items = [it for it in (store.load_item(k) for k in d["items"]) if it]
    dev = deviations(day, items)
    today = Counter(t for it in items for t in _tags_of(it))
    mx = today.most_common(1)[0][1] if today else 0
    print(f"{d['date']:12} {len(items):>5} {dev['baseline_days']:>5} {dev['status']:13} "
          f"{len(dev['found']):>5} {dev['distinct_tags_today']:>8} {mx:>9}")
