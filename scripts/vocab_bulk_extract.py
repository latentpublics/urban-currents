"""S3: grow the vocabulary from a large stratified sample (phase 0f).

Phase 0e diagnosed the overlay bottleneck as a vocabulary problem — the LLM
proposed 5.4 tags per item that nothing caught, against 1.2 it did — but the
sample was five days, and a frequency floor of 2 over 123 items produced only 22
candidates. Most of the 636 unmatched terms appeared exactly once, so the floor
could not tell a real term from a one-paper coinage. **The floor was not wrong;
the sample was too small for it to work.**

So this runs the extraction prompt over 300 items drawn from the 90 days, and
raises the floor to 3 now that repeats mean something.

**Stratified, not random.** Phase 0e measured our collection at 49% transport,
so a random draw grows the transport vocabulary and leaves the rest where it is.
The sample is spread across OpenAlex `primary_topic.subfield` with a per-stratum
cap, and the composition is reported rather than assumed.

Extraction only. No summaries: the output wanted here is vocabulary, not cards,
and summarising 300 backfill items would cost several times as much for nothing.

Usage:
    uv run python scripts/vocab_bulk_extract.py --sample 300 --json runs/vocab_bulk.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.collectors.base import clean_text, invert_abstract, normalize_doi  # noqa: E402
from pipeline.collectors.base import normalize_openalex_id  # noqa: E402
from pipeline.graph.citation import iter_raw_openalex_works  # noqa: E402
from pipeline.linking.extract import extract_overlay  # noqa: E402
from pipeline.linking.vocab_match import Vocabulary, match_facet  # noqa: E402
from pipeline.llm import LLMClient  # noqa: E402
from pipeline.metrics import Run  # noqa: E402
from pipeline.models import Bibliography, Ids, Item  # noqa: E402

FACETS = ("methods", "data", "tools")


def candidate_items() -> list[tuple[str, Item]]:
    """Backfill works that have an abstract, paired with their subfield."""
    out: list[tuple[str, Item]] = []
    seen: set[str] = set()
    for work in iter_raw_openalex_works():
        abstract = invert_abstract(work.get("abstract_inverted_index"))
        title = clean_text(work.get("display_name"))
        if not abstract or not title:
            continue
        oa = normalize_openalex_id(work.get("id"))
        doi = normalize_doi(work.get("doi"))
        key = f"doi:{doi}" if doi else f"openalex:{oa}"
        if key in seen:
            continue
        seen.add(key)
        subfield = (
            ((work.get("primary_topic") or {}).get("subfield") or {}).get("id") or ""
        ).rsplit("/", 1)[-1] or "unknown"
        out.append((
            subfield,
            Item(
                work_key=key,
                first_published=date(2026, 8, 11),
                ids=Ids(openalex=oa, doi=doi),
                bibliography=Bibliography(title=title, abstract=abstract),
            ),
        ))
    return out


def stratify(pool: list[tuple[str, Item]], n: int, seed: int = 42) -> dict[str, list[Item]]:
    """Spread the sample across subfields with a per-stratum cap.

    Round-robin over strata rather than a fixed quota each: strata are very
    unequal, and a fixed quota either starves the big ones or cannot fill from
    the small ones. This takes one from each in turn until the sample is full,
    which gives the smaller subfields their whole population and caps the large
    ones without any parameter to tune.
    """
    rng = random.Random(seed)
    by_sub: dict[str, list[Item]] = defaultdict(list)
    for subfield, item in pool:
        by_sub[subfield].append(item)
    for items in by_sub.values():
        rng.shuffle(items)

    picked: dict[str, list[Item]] = defaultdict(list)
    order = sorted(by_sub, key=lambda s: -len(by_sub[s]))
    total = 0
    while total < n:
        progressed = False
        for subfield in order:
            if total >= n:
                break
            bucket = by_sub[subfield]
            if len(picked[subfield]) < len(bucket):
                picked[subfield].append(bucket[len(picked[subfield])])
                total += 1
                progressed = True
        if not progressed:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pool = candidate_items()
    print(f"backfill works with an abstract: {len(pool)}")

    picked = stratify(pool, a.sample)
    items = [it for group in picked.values() for it in group]
    composition = sorted(
        ({"subfield": s, "sampled": len(v)} for s, v in picked.items()),
        key=lambda r: -r["sampled"],
    )
    print(f"sampled: {len(items)} across {len(picked)} subfields")
    for row in composition[:12]:
        print(f"   {row['subfield']:>8}: {row['sampled']}")

    run = Run.for_date(date(2026, 8, 11))
    client = LLMClient(task="extract")
    if not client.available():
        print("no LLM key: nothing to do")
        return

    stash, stats = extract_overlay(items, run, client=client)
    print(f"\nextraction: {json.dumps(stats)}")

    vocabs = {f: Vocabulary.load(f) for f in FACETS}
    unmatched: Counter = Counter()
    example: dict[tuple[str, str], str] = {}
    matched = 0
    by_item = {it.work_key: it for it in items}

    for work_key, payload in stash.items():
        for facet in FACETS:
            result = match_facet(payload.get(facet) or [], facet, vocabs[facet])
            matched += len(result.refs)
            for raw in result.unmatched:
                term = (raw or "").strip().lower()
                if not term:
                    continue
                unmatched[(facet, term)] += 1
                example.setdefault(
                    (facet, term), by_item[work_key].bibliography.title
                )

    result = {
        "population": "90-day backfill works with an abstract",
        "pool": len(pool),
        "sampled": len(items),
        "composition": composition,
        "extraction": stats,
        "matched_tags": matched,
        "distinct_unmatched_terms": len(unmatched),
        "unmatched_total": sum(unmatched.values()),
        "top_unmatched": [
            {
                "facet": facet, "term": term, "count": n,
                "example_title": example.get((facet, term)),
            }
            for (facet, term), n in unmatched.most_common(150)
        ],
    }

    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )

    print(f"\nmatched: {matched} | unmatched: {sum(unmatched.values())} "
          f"across {len(unmatched)} distinct terms")
    print(f"\ntop 30 unmatched:")
    for row in result["top_unmatched"][:30]:
        print(f"  {row['count']:>3}  {row['facet']:<8} {row['term'][:52]}")


if __name__ == "__main__":
    main()
