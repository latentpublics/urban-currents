"""How long after submission does arXiv's API admit a paper exists? (phase 0k, X6)

X1 measured the journal side from stored responses and could only say of arXiv
that the lag is "at least 1 day" — the one direct experiment on record was a
single day returning zero. That was enough to move the issue date, and not
enough to size the daily window.

It became worth measuring properly when the first real `uc daily --dry-run`
covered 2026-08-15..17 and fetched **zero** arXiv items while the journal side
worked. Two explanations fit that observation and they call for opposite
responses:

- arXiv genuinely had nothing in our categories over a weekend, or
- the window ends before arXiv's `submittedDate` index reaches it, in which case
  a scheduled daily would publish journal-only issues every morning and report
  `collect.arxiv: OK` while doing it.

The second would be the outcome model's blind spot: every stage green, a real
issue published, and a whole source silently missing. So this asks the API for
one day at a time, walking backwards, and reports where the counts start.

One request per day, at the collector's own rate limit. `totalResults` only —
no entries are parsed and nothing is stored, because the question is how many
exist and not what they are.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import paths  # noqa: E402
from pipeline.config import cfg  # noqa: E402

API = "https://export.arxiv.org/api/query"
TOTAL = re.compile(r"<opensearch:totalResults[^>]*>(\d+)<")
UPDATED = re.compile(r"<updated>([^<]+)</updated>")


def categories() -> list[str]:
    return list(cfg("gate.ungated_categories", [])) + list(cfg("gate.gated_categories", []))


def count_for(day: date, cats: list[str]) -> tuple[int, str]:
    """`totalResults` for one submission day, plus the feed's own timestamp.

    The feed timestamp is the useful second number: it says when arXiv last
    rebuilt the index that is answering, which is what a zero has to be read
    against.
    """
    cat_query = " OR ".join(f"cat:{c}" for c in cats)
    stamp = day.strftime("%Y%m%d")
    query = f'({cat_query}) AND submittedDate:"{stamp}0000 TO {stamp}2359"'
    url = f"{API}?{urllib.parse.urlencode({'search_query': query, 'start': 0, 'max_results': 1})}"

    req = urllib.request.Request(url, headers={"User-Agent": "urban-currents/0.2 (phase0k)"})
    with urllib.request.urlopen(req, timeout=60) as response:
        body = response.read().decode("utf-8", "replace")

    total = TOTAL.search(body)
    updated = UPDATED.search(body)
    return (int(total.group(1)) if total else -1, updated.group(1) if updated else "")


def main() -> None:
    today = date.today()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cats = categories()

    rows = []
    for i in range(1, days + 1):
        day = today - timedelta(days=i)
        total, feed_updated = count_for(day, cats)
        rows.append({
            "date": str(day),
            "days_ago": i,
            "total_results": total,
            "weekday": day.strftime("%a"),
            "feed_updated": feed_updated,
        })
        print(f"  {day} ({day.strftime('%a')})  D-{i:<2}  {total:>5}")
        time.sleep(3.0)  # the collector's own rate limit

    # The first horizon at which every day at or beyond it is non-empty. A single
    # empty Sunday inside a run of populated days is a weekend, not a lag.
    lag = None
    for i, row in enumerate(rows):
        if all(r["total_results"] > 0 for r in rows[i:]):
            lag = row["days_ago"]
            break

    weekdays = [r for r in rows if r["weekday"] not in ("Sat", "Sun")]
    empty_weekdays = [r["date"] for r in weekdays if r["total_results"] == 0]

    out = {
        "measured_at": str(today),
        "categories": cats,
        "days": rows,
        "first_reliable_horizon_days": lag,
        "empty_weekdays": empty_weekdays,
        "note": (
            "total_results is arXiv's own count for that submission day, asked "
            "today. A day that is empty now and populated tomorrow is indexing "
            "lag; a day that is empty and stays empty is a weekend."
        ),
    }
    target = paths.RUNS / "arxiv_visibility.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\nfirst horizon with no empty day beyond it: D-{lag}")
    print(f"empty weekdays: {empty_weekdays or 'none'}")
    print(f"→ {target}")


if __name__ == "__main__":
    main()
