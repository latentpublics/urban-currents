"""V1-3: how much civil-materials work arrives through the transport journals.

Two of the four `drop_not_our_kind` items in the affinity probe are pavement
materials papers — reclaimed asphalt binder consistency, quarry by-products in
Otta seal surfacing. They are not misclassified: a transport journal published
them and the whitelist is a membership test, so they enter correctly and are
still not what this digest is about.

**This counts. It does not exclude.** Where the boundary of "urban data science"
sits is a coverage decision, and a coverage decision made by a script is a
coverage decision nobody made. What is produced here is the size of the
population and a candidate rule for identifying it.

Usage:
    uv run python scripts/materials_drift.py --json runs/materials_drift.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.graph.citation import iter_raw_openalex_works  # noqa: E402

# Terms from the two labelled examples plus the vocabulary that surrounds them.
# Deliberately narrow: "pavement" and "asphalt" are the materials themselves,
# not the transport phenomena, and a paper about traffic on a pavement does not
# use "binder" or "aggregate gradation".
MATERIALS_TERMS = (
    "asphalt", "bitumen", "bituminous", "binder content", "reclaimed asphalt",
    " rap ", "aggregate", "quarry", "seal surfacing", "chip seal", "subgrade",
    "pavement", "concrete mix", "compressive strength", "rutting", "fatigue cracking",
    "marshall stability", "superpave", "gradation",
)
STRONG = ("asphalt", "bitumen", "bituminous", "quarry", "superpave", "seal surfacing")


def hits(text: str) -> list[str]:
    low = f" {(text or '').lower()} "
    return [t for t in MATERIALS_TERMS if t.strip() in low]


def looks_like_materials(title: str, abstract: str) -> bool:
    """Two terms in the title, or a strong term anywhere plus one more.

    A candidate rule, offered for judgement. It is not applied anywhere.
    """
    t, a = hits(title), hits(f"{title} {abstract}")
    if len(t) >= 2:
        return True
    return any(s in " ".join(a) for s in STRONG) and len(a) >= 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from pipeline.collectors.base import invert_abstract

    seen: set[str] = set()
    total = 0
    matches: list[dict] = []
    topics: Counter = Counter()
    subfields: Counter = Counter()
    venues: Counter = Counter()
    all_subfields: Counter = Counter()

    for work in iter_raw_openalex_works():
        wid = work.get("id")
        if not wid or wid in seen:
            continue
        seen.add(wid)
        total += 1
        title = work.get("display_name") or ""
        abstract = invert_abstract(work.get("abstract_inverted_index")) or ""
        pt = work.get("primary_topic") or {}
        sf = (pt.get("subfield") or {}).get("display_name") or "—"
        all_subfields[sf] += 1
        if not looks_like_materials(title, abstract):
            continue
        topics[pt.get("display_name") or "—"] += 1
        subfields[sf] += 1
        venues[
            ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "—"
        ] += 1
        matches.append({
            "title": title[:110],
            "topic": pt.get("display_name"),
            "subfield": sf,
            "venue": ((work.get("primary_location") or {}).get("source") or {}).get(
                "display_name"
            ),
        })

    # The keyword rule was the way in; the topic assignment is the better rule
    # and the scan is what found it. IDs read from the resolved records, never
    # guessed (D92) — the last time identifiers were written from memory, four
    # of six were wrong and the rule silently did nothing.
    candidate_rule = {
        "primary_topic.id": {
            "https://openalex.org/T10264": "Asphalt Pavement Performance Evaluation (27 in corpus)",
            "https://openalex.org/T11606": "Infrastructure Maintenance and Monitoring (20 in corpus)",
        },
        "primary_topic.subfield.id": {
            "https://openalex.org/subfields/2205":
                "Civil and Structural Engineering (107 in corpus, 30 of them matched)",
        },
        "note": (
            "the subfield is wider than the drift — 77 of its 107 works are not "
            "materials papers — so the topic pair is the tighter rule and the "
            "subfield is the blunt one"
        ),
    }

    out = {
        "candidate_rule": candidate_rule,
        "population": f"{total} distinct works in runs/*/raw/openalex (collect + backfill)",
        "matched": len(matches),
        "share": round(len(matches) / total, 5) if total else 0,
        "rule": (
            "two materials terms in the title, or one strong term (asphalt, "
            "bitumen, quarry, superpave, seal surfacing) plus another anywhere"
        ),
        "applied": False,
        "top_topics": dict(topics.most_common(12)),
        "top_subfields": dict(subfields.most_common(8)),
        "top_venues": dict(venues.most_common(10)),
        "subfield_share_of_corpus": {
            k: {"matched": subfields.get(k, 0), "in_corpus": all_subfields.get(k, 0)}
            for k in list(subfields)[:8]
        },
        "matches": matches,
    }
    if a.json:
        Path(a.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n",
        )

    print(out["population"])
    print(f"matched {out['matched']} ({out['share'] * 100:.2f}%) — rule NOT applied anywhere")
    print("\ntop topics:")
    for k, v in out["top_topics"].items():
        print(f"   {v:>4}  {k}")
    print("\ntop subfields (matched / in corpus):")
    for k, v in out["subfield_share_of_corpus"].items():
        print(f"   {v['matched']:>4} / {v['in_corpus']:<5}  {k}")
    print("\ntop venues:")
    for k, v in out["top_venues"].items():
        print(f"   {v:>4}  {k}")


if __name__ == "__main__":
    main()
