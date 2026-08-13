"""Q1-2: is a reference list that exists also complete? (phase 0d)

A non-zero `referenced_works` can still be truncated, and a canon built on a
truncated base would be quietly biased toward whatever survived the truncation.

Crossref is the independent check: publishers deposit a `reference-count`
alongside the references themselves, so comparing OpenAlex's list length against
that count is an API-to-API comparison with no scraping and no estimation. It
does not cover arXiv preprints, which have no Crossref reference deposit — that
gap is reported rather than filled with a guess, since counting LaTeX
bibliographies would mean fetching source tarballs this pipeline does not fetch.

Reports only.

Usage:
    uv run python scripts/citation_completeness.py --sample 30 --json out.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from pipeline import store  # noqa: E402
from pipeline.collectors.abstracts import CROSSREF_API, publisher_of  # noqa: E402
from pipeline.config import contact_email  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    candidates = [
        it for it in store.iter_items()
        if it.ids.doi and not it.ids.arxiv and it.graph.referenced_works
    ]
    rng = random.Random(42)
    picked = rng.sample(candidates, min(a.sample, len(candidates)))

    client = httpx.Client(
        headers={"User-Agent": (
            f"UrbanCurrents/0.1 (+https://github.com/youngjour/urban-currents; "
            f"mailto:{contact_email()})"
        )},
        timeout=30.0,
        follow_redirects=True,
    )

    rows = []
    for item in picked:
        try:
            r = client.get(
                f"{CROSSREF_API}{item.ids.doi}", params={"mailto": contact_email()}
            )
        except Exception as e:  # noqa: BLE001
            rows.append({"work_key": item.work_key, "error": type(e).__name__})
            continue
        if r.status_code != 200:
            rows.append({"work_key": item.work_key, "error": f"HTTP {r.status_code}"})
            continue
        msg = r.json().get("message") or {}
        deposited = msg.get("reference-count")
        ours = len(item.graph.referenced_works)
        rows.append({
            "work_key": item.work_key,
            "publisher": publisher_of(item),
            "openalex_references": ours,
            "crossref_reference_count": deposited,
            "ratio": round(ours / deposited, 4) if deposited else None,
        })

    usable = [r for r in rows if r.get("ratio") is not None]
    ratios = [r["ratio"] for r in usable]
    result = {
        "population": "content/items with a DOI and a non-empty referenced_works",
        "eligible": len(candidates),
        "sampled": len(picked),
        "compared": len(usable),
        "not_compared": len(rows) - len(usable),
        "ratio_openalex_over_crossref": {
            "median": round(statistics.median(ratios), 4) if ratios else None,
            "mean": round(statistics.fmean(ratios), 4) if ratios else None,
            "min": round(min(ratios), 4) if ratios else None,
            "max": round(max(ratios), 4) if ratios else None,
            "at_least_90pct": sum(1 for x in ratios if x >= 0.9),
            "below_50pct": sum(1 for x in ratios if x < 0.5),
        },
        "arxiv_note": (
            "arXiv preprints are excluded: they have no Crossref reference "
            "deposit, and counting LaTeX bibliographies would need source "
            "tarballs this pipeline does not fetch. Not estimated."
        ),
        "rows": rows,
    }

    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
