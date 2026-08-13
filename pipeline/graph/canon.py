"""Canon candidates: the works this archive keeps citing (phase 0d, Q3).

Selection only. Nothing here publishes, renders, or enters an issue — the output
is a list and the evidence for judging whether the list is any good.

**Recency weighting.** A raw citation count answers "what has this field always
cited", which is a library question. Weighting each citation by how recent the
*citing* item is answers "what is this field leaning on now", which is the one a
daily product asks. A 2007 paper cited by twelve items this month is alive; the
same paper cited once in 2019 is furniture. The weight is an exponential decay
on the citing item's publication date with a half-life from
`citation.canon_half_life_days`, so a citation from a year ago counts about a
quarter of a fresh one.

**Scope.** Restricted to works published in a whitelist journal or on arXiv.
Urban research cites statistics texts, machine-learning papers and epidemiology;
without the restriction the top of the list is whatever the whole world cites
most, and the count of what this drops is reported rather than hidden.

**Known bias**, measured in Q1-2: OpenAlex's reference lists are a median 0.75
of what publishers deposit, because it keeps the references it can resolve to a
Work. Books, reports and non-DOI sources fall out, so this ranking
under-represents exactly the kind of source a planning literature cites most.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Optional

from .. import paths, store
from ..config import cfg
from .citation import load_reference_base

CANDIDATES_FILE = "candidates.json"


def _parse(value: str) -> Optional[date]:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def accumulate(
    records: Optional[list[dict]] = None,
    today: Optional[date] = None,
    half_life_days: Optional[int] = None,
) -> list[dict]:
    """Weighted and raw citation counts per referenced Work, most cited first."""
    records = load_reference_base() if records is None else records
    half_life = float(
        cfg("citation.canon_half_life_days", 180) if half_life_days is None else half_life_days
    )
    dates = [d for d in (_parse(r.get("date", "")) for r in records) if d]
    today = today or (max(dates) if dates else date.today())

    raw: dict[str, int] = defaultdict(int)
    weighted: dict[str, float] = defaultdict(float)
    recent12: dict[str, int] = defaultdict(int)
    citers: dict[str, set[str]] = defaultdict(set)

    for record in records:
        citing_date = _parse(record.get("date", ""))
        if citing_date is None:
            continue
        age = max((today - citing_date).days, 0)
        weight = math.pow(0.5, age / half_life) if half_life > 0 else 1.0
        is_recent = age <= 365
        for ref in record.get("referenced_works") or []:
            raw[ref] += 1
            weighted[ref] += weight
            citers[ref].add(record["work_key"])
            if is_recent:
                recent12[ref] += 1

    rows = [
        {
            "openalex_id": ref,
            "archive_citations": raw[ref],
            "archive_citations_last_12m": recent12[ref],
            "weighted_score": round(weighted[ref], 6),
            "citing_items": sorted(citers[ref]),
        }
        for ref in raw
    ]
    rows.sort(key=lambda r: (-r["weighted_score"], -r["archive_citations"], r["openalex_id"]))
    return rows


def _date_precision(work: dict) -> str:
    d = (work.get("publication_date") or "").strip()
    if not d or len(d) < 10:
        return "unknown"
    _, month, day = d.split("-")
    if month == "01" and day == "01":
        return "year"
    if day == "01":
        return "month"
    return "day"


def exclusions() -> dict[str, set[str]]:
    """Topic and subfield IDs to drop, from `vocab/canon_exclude_subfields.yaml`."""
    from ..config import vocab_file

    doc = vocab_file("canon_exclude_subfields.yaml") or {}
    return {
        "topics": {t["id"] for t in (doc.get("topics") or []) if t.get("id")},
        "subfields": {s["id"] for s in (doc.get("subfields") or []) if s.get("id")},
    }


def _ids(work: dict) -> tuple[str, str, str]:
    pt = work.get("primary_topic") or {}
    topic = (pt.get("id") or "").rsplit("/", 1)[-1]
    subfield = ((pt.get("subfield") or {}).get("id") or "").rsplit("/", 1)[-1]
    source = ((work.get("primary_location") or {}).get("source") or {}).get("id") or ""
    return topic, subfield, source.rsplit("/", 1)[-1]


def _in_scope(work: dict, whitelist_ids: set[str], mode: str = "subfield") -> bool:
    """Whether a cited work belongs in the canon list.

    `venue` is the original rule and is kept so the change is reversible and
    measurable: published in a whitelist journal or on arXiv. It was wrong
    because `journals.yaml` answers "what do we poll daily", and using it here
    let a 159-entry operational list overrule our own corpus — three of our
    papers cited Ewing & Handy's urban design qualities and it was dropped for
    appearing in a journal we do not poll.

    `subfield` is the rule now in force, and despite the name it excludes at
    **topic** level: entry is being cited twice by our corpus, and the only way
    out is being a generic research instrument, named in
    `vocab/canon_exclude_subfields.yaml` with the work each entry removes.
    Subfield granularity was measured and rejected — see that file.
    """
    topic, subfield, source_id = _ids(work)

    if mode in ("venue", "both"):
        in_venue = bool(source_id) and (
            source_id in whitelist_ids
            or source_id == cfg("openalex.arxiv_source_id", "S4306400194")
        )
        if mode == "venue":
            return in_venue
        if not in_venue:
            return False

    ex = exclusions()
    return topic not in ex["topics"] and subfield not in ex["subfields"]


def resolve_candidates(
    rows: list[dict],
    top_n: Optional[int] = None,
    batch: int = 50,
    mode: Optional[str] = None,
    min_citations: Optional[int] = None,
) -> dict[str, Any]:
    """Fetch metadata for the highest-scoring works and keep the in-scope ones."""
    from ..collectors.openalex import configure_pyalex
    from ..config import journals_vocab

    pyalex = configure_pyalex()
    if pyalex is None:
        return {"status": "NO_KEY", "why": "OPENALEX_KEY is not set"}

    top_n = int(cfg("citation.canon_top_n", 300) if top_n is None else top_n)
    mode = mode or str(cfg("citation.canon_scope_mode", "subfield"))
    # Entry is our own corpus citing it more than once. One citation is one
    # paper's reading list; two is the beginning of a shared reference.
    min_citations = int(
        cfg("citation.canon_min_citations", 2) if min_citations is None else min_citations
    )
    eligible = [r for r in rows if r["archive_citations"] >= min_citations]
    # Over-fetch: the scope filter drops works from outside the field, and the
    # aim is `top_n` in-scope candidates rather than `top_n` fetched.
    pool = eligible[: top_n * 3]
    whitelist = {
        s["id"] for s in (journals_vocab().get("sources") or [])
        if s.get("id") and s.get("include", True)
    }

    by_id = {r["openalex_id"]: r for r in pool}
    bare = [k.split(":", 1)[1] if ":" in k else k for k in by_id]
    cost = 0.0
    fetched: dict[str, dict] = {}
    for start in range(0, len(bare), batch):
        chunk = bare[start : start + batch]
        results, meta = (
            pyalex.Works().filter(openalex_id="|".join(chunk)).get(
                per_page=batch, return_meta=True
            )
        )
        cost += float((meta or {}).get("cost_usd") or 0.0)
        for w in results:
            fetched[(w.get("id") or "").rsplit("/", 1)[-1]] = w

    candidates, out_of_scope = [], []
    for key, row in by_id.items():
        work = fetched.get(key.split(":", 1)[1] if ":" in key else key)
        if work is None:
            continue
        loc = (work.get("primary_location") or {}).get("source") or {}
        topic, subfield, source_id = _ids(work)
        pt = work.get("primary_topic") or {}
        world = work.get("cited_by_count") or 0
        entry = {
            **{k: v for k, v in row.items() if k != "citing_items"},
            "title": work.get("display_name"),
            "year": work.get("publication_year"),
            "publication_date": work.get("publication_date"),
            "date_precision": _date_precision(work),
            "venue": loc.get("display_name"),
            "venue_id": source_id or None,
            "topic": pt.get("display_name"),
            "topic_id": topic or None,
            "subfield": (pt.get("subfield") or {}).get("display_name"),
            "subfield_id": subfield or None,
            "authors": [
                (a.get("author") or {}).get("display_name")
                for a in (work.get("authorships") or [])[:5]
            ],
            "openalex_cited_by_count": world,
            # A general research instrument is cited enormously by everyone and
            # a little by us. Reported as a column so a human skimming the list
            # can recognise one; never used to exclude, because the boundary is
            # arbitrary and heavily cited genuine canon exists.
            "world_to_archive_ratio": (
                round(world / row["archive_citations"], 1)
                if row["archive_citations"] else None
            ),
            "citing_items": row["citing_items"],
        }
        (candidates if _in_scope(work, whitelist, mode) else out_of_scope).append(entry)

    candidates.sort(key=lambda r: (-r["weighted_score"], r["openalex_id"]))
    out_of_scope.sort(key=lambda r: -r["weighted_score"])
    return {
        "status": "OK",
        "generated_from": "content/graph/references.jsonl",
        "scope_mode": mode,
        "min_archive_citations": min_citations,
        "distinct_referenced_works": len(rows),
        "eligible_by_citation_count": len(eligible),
        "considered": len(pool),
        "resolved": len(fetched),
        "in_scope": len(candidates),
        "out_of_scope": len(out_of_scope),
        "out_of_scope_examples": [
            {
                "title": e["title"], "venue": e["venue"], "topic": e["topic"],
                "archive_citations": e["archive_citations"],
                "openalex_cited_by_count": e["openalex_cited_by_count"],
            }
            for e in out_of_scope[:10]
        ],
        "half_life_days": float(cfg("citation.canon_half_life_days", 180)),
        "openalex_cost_usd": round(cost, 6),
        "candidates": candidates[:top_n],
    }


def build_candidates(out: Optional[Path] = None, **kwargs) -> dict[str, Any]:
    result = resolve_candidates(accumulate(), **kwargs)
    target = out or (paths.CONTENT / "canon" / CANDIDATES_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    store.write_text_atomic(
        target, json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return result
