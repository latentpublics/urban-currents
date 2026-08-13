"""Q1-3: are the works our items reference actually identifiable? (phase 0d)

`referenced_works` is a list of OpenAlex Work IDs and nothing else. A canon
candidate needs a title, a year, a journal and authors, so the layer only works
if those IDs resolve — and the date precision decides whether an anniversary
card is even possible, because OpenAlex stores `publication_date` at whatever
precision the publisher deposited.

Costs OpenAlex budget: works are fetched 50 to a page with an `id:` filter, and
every response's `meta.cost_usd` is accumulated exactly as the collector does.

Reports only. Nothing here writes to `content/`.

Usage:
    uv run python scripts/citation_resolve.py --sample 200 --json runs/citation_resolve.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import store  # noqa: E402
from pipeline.collectors.openalex import configure_pyalex  # noqa: E402
from pipeline.config import cfg  # noqa: E402

PAGE = 50


def date_precision(work: dict) -> str:
    """day / month / year, read from what OpenAlex actually deposited.

    `publication_date` is always an ISO date string, but a year-only deposit
    surfaces as January 1st and a month-only one as the 1st of that month. That
    is a heuristic and it is stated as one: a paper genuinely published on
    1 January is counted as year-precision here. `from_publication_date` and
    `publication_year` agreeing is the corroboration used.
    """
    d = (work.get("publication_date") or "").strip()
    if not d or len(d) < 10:
        return "unknown"
    _, month, day = d.split("-")
    if month == "01" and day == "01":
        return "year"
    if day == "01":
        return "month"
    return "day"


def sample_reference_ids(n: int, seed: int = 42) -> tuple[list[str], dict]:
    all_refs: list[str] = []
    per_item: list[int] = []
    for item in store.iter_items():
        refs = item.graph.referenced_works
        per_item.append(len(refs))
        all_refs.extend(refs)
    unique = sorted(set(all_refs))
    rng = random.Random(seed)
    picked = rng.sample(unique, min(n, len(unique)))
    return picked, {
        "reference_mentions": len(all_refs),
        "distinct_reference_ids": len(unique),
        "items_scanned": len(per_item),
        "sampled": len(picked),
    }


def resolve(ids: list[str]) -> dict:
    pyalex = configure_pyalex()
    if pyalex is None:
        return {"status": "NO_KEY", "why": "OPENALEX_KEY is not set"}

    bare = [i.split(":", 1)[1] if ":" in i else i for i in ids]
    found: dict[str, dict] = {}
    cost = 0.0
    pages = 0

    for start in range(0, len(bare), PAGE):
        chunk = bare[start : start + PAGE]
        query = pyalex.Works().filter(openalex_id="|".join(chunk))
        results, meta = query.get(per_page=PAGE, return_meta=True)
        pages += 1
        cost += float((meta or {}).get("cost_usd") or 0.0)
        for w in results:
            wid = (w.get("id") or "").rsplit("/", 1)[-1]
            found[wid] = w

    missing = [b for b in bare if b not in found]
    precision = Counter(date_precision(w) for w in found.values())
    have_title = sum(1 for w in found.values() if w.get("display_name"))
    have_venue = sum(
        1 for w in found.values()
        if ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
    )
    have_author = sum(1 for w in found.values() if w.get("authorships"))
    years = [w.get("publication_year") for w in found.values() if w.get("publication_year")]

    return {
        "status": "OK",
        "requested": len(bare),
        "resolved": len(found),
        "resolve_rate": round(len(found) / len(bare), 4) if bare else None,
        "missing_examples": missing[:5],
        "pages_fetched": pages,
        "openalex_cost_usd": round(cost, 6),
        "cost_per_1000_ids": round(cost / len(bare) * 1000, 6) if bare else None,
        "metadata_completeness": {
            "title": round(have_title / len(found), 4) if found else None,
            "venue": round(have_venue / len(found), 4) if found else None,
            "authors": round(have_author / len(found), 4) if found else None,
        },
        "date_precision": {
            k: round(v / len(found), 4) for k, v in sorted(precision.items())
        },
        "date_precision_counts": dict(sorted(precision.items())),
        "year_range": [min(years), max(years)] if years else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ids, stats = sample_reference_ids(a.sample)
    result = {"sampling": stats, "resolution": resolve(ids), "seed": 42}
    result["daily_budget_usd"] = float(cfg("openalex.daily_budget_usd", 1.0))

    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
