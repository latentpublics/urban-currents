"""V1-2: how many book reviews the detector finds, and what it wrongly catches.

Runs over three populations, kept apart because they answer different questions:
the labelled sample (does it find the three we know about?), the five prepared
days (what does it change about today?), and every raw OpenAlex response on disk
(how often does it fire, and on what?).

Every match is printed. A detector nobody can audit is a detector nobody should
trust, and the false-positive question cannot be answered by a count.

Usage:
    uv run python scripts/book_review_scan.py --json runs/book_review_scan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.filters.book_review import is_book_review, signals  # noqa: E402
from pipeline.graph.citation import iter_raw_openalex_works  # noqa: E402
from pipeline.labeling import load_labels  # noqa: E402
from pipeline.metrics import Run  # noqa: E402
from pipeline.stages import read_stage  # noqa: E402

DAYS = ["2026-08-05", "2026-08-06", "2026-08-07", "2026-08-10", "2026-08-11"]


def scan_labels() -> dict:
    rows = [r for r in load_labels("relevance") if r.get("source") == "journal"]
    hits = [r for r in rows if is_book_review(r["title"])]
    by_label = Counter(r["label"] for r in hits)
    return {
        "population": f"{len(rows)} labelled journal items",
        "detected": len(hits),
        "detected_labels": dict(by_label),
        # A detected item that YJUN kept would be a false positive with a
        # verdict attached, which is the strongest evidence available.
        "detected_but_kept": [r["title"][:90] for r in hits if r["label"] == "keep"],
        "titles": [f"[{r['label']}] {r['title'][:90]}" for r in hits],
    }


def scan_days() -> dict:
    out = {"population": "classify-stage journal candidates, 5 prepared days", "by_day": {}}
    total = 0
    titles = []
    for d in DAYS:
        items = read_stage(Run.for_date(date.fromisoformat(d)), "classify") or []
        hits = [it for it in items if is_book_review(it.bibliography.title)]
        out["by_day"][d] = {"candidates": len(items), "detected": len(hits)}
        total += len(hits)
        titles += [it.bibliography.title[:100] for it in hits]
    out["detected"] = total
    out["titles"] = titles
    return out


def scan_raw() -> dict:
    """Every OpenAlex work the pipeline has collected, with `type` available."""
    seen: set[str] = set()
    hits: list[dict] = []
    by_signal: Counter = Counter()
    n = 0
    type_counts: Counter = Counter()
    for work in iter_raw_openalex_works():
        wid = work.get("id")
        if not wid or wid in seen:
            continue
        seen.add(wid)
        n += 1
        title = work.get("display_name") or ""
        wtype = work.get("type")
        type_counts[wtype] += 1
        if is_book_review(title, wtype):
            s = signals(title, wtype)
            by_signal[tuple(sorted(k for k, v in s.items() if v))] += 1
            hits.append({
                "title": title[:110],
                "type": wtype,
                "venue": ((work.get("primary_location") or {}).get("source") or {}).get(
                    "display_name"
                ),
                "signals": [k for k, v in s.items() if v],
                "pages": (work.get("biblio") or {}).get("first_page"),
            })
    return {
        "population": f"{n} distinct works in runs/*/raw/openalex (collect + backfill)",
        "detected": len(hits),
        "rate": round(len(hits) / n, 5) if n else 0,
        "by_signal_combination": {" + ".join(k): v for k, v in by_signal.most_common()},
        "openalex_types_present": dict(type_counts.most_common(8)),
        "matches": hits,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out = {"labels": scan_labels(), "days": scan_days(), "raw": scan_raw()}
    if a.json:
        Path(a.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n",
        )

    print(f"labelled journal items: {out['labels']['population']}")
    print(f"   detected {out['labels']['detected']} — {out['labels']['detected_labels']}")
    for t in out["labels"]["titles"]:
        print(f"     {t}")
    if out["labels"]["detected_but_kept"]:
        print(f"   FALSE POSITIVES (YJUN kept these): {out['labels']['detected_but_kept']}")

    print(f"\n5 prepared days: detected {out['days']['detected']}")
    for d, v in out["days"]["by_day"].items():
        print(f"   {d}  {v['detected']} of {v['candidates']} candidates")

    r = out["raw"]
    print(f"\n{r['population']}")
    print(f"   detected {r['detected']} ({r['rate'] * 100:.3f}%)")
    print(f"   by signal: {r['by_signal_combination']}")
    print(f"   openalex types seen: {r['openalex_types_present']}")
    for m in r["matches"][:25]:
        print(f"     [{m['type']}] {m['title']}  ({', '.join(m['signals'])})")
    if len(r["matches"]) > 25:
        print(f"     … and {len(r['matches']) - 25} more in the JSON")


if __name__ == "__main__":
    main()
