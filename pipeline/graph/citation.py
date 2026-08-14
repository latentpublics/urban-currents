"""Citation layer: reference base, internal citations, bibliographic coupling.

Everything here is a **build output**. It is derived from Items and from the
run stage files, it is regenerated rather than edited, and deleting any of it
costs a recompute and nothing else.

Why coupling rather than co-citation: two papers are bibliographically coupled
when they cite the same earlier work, and that is knowable the day both appear.
Co-citation — two papers later cited together — needs the field to have reacted,
which takes years. A daily product can only use the first.

What Q1 measured, and what it means for everything below:

- Journal items carry references 86% of the time; arXiv items 5%. This layer
  describes the journal half of the archive and says so.
- OpenAlex's reference lists are a median 0.75 of what publishers deposited to
  Crossref, because OpenAlex keeps the references it can resolve to a Work.
  Books, reports and non-DOI sources fall out, so anything ranked here is
  biased toward what OpenAlex already indexes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .. import paths, store
from ..config import cfg
from ..models import Item

REFERENCES_FILE = "references.jsonl"
COUPLING_FILE = "coupling.jsonl"
WORK_INDEX_FILE = "work_index.jsonl"


# --------------------------------------------------------------------------
# Reference base
# --------------------------------------------------------------------------


def iter_run_candidates() -> Iterator[Item]:
    """Every above-threshold candidate the pipeline has ever scored.

    A third of the archive's references sit on items that were scored and never
    published — 124 of 345 candidates over the five prepared days, carrying
    2,819 of 8,435 references. Reading them from the run stage files keeps that
    base without changing what `content/items/` means: that directory is what
    was *published*, and it stays that way.

    Runs are not committed, so this is best-effort by design. What it finds is
    written to `content/graph/references.jsonl`, which is.
    """
    from ..stages import read_stage

    seen: set[str] = set()
    for run_dir in sorted(paths.RUNS.glob("run_*")):
        if not (run_dir / "stages").exists():
            continue

        class _R:  # `read_stage` only needs `.dir`
            dir = run_dir

        for stage in ("enrich", "classify"):
            items = read_stage(_R, stage)
            if not items:
                continue
            for item in items:
                if item.work_key in seen or not item.graph.referenced_works:
                    continue
                seen.add(item.work_key)
                yield item
            break


# Which runs the reference base is allowed to read. The canon's claim is "what
# our corpus stands on", and our corpus is what the pipeline collected — not
# every response any script happened to receive while measuring something.
CITATION_ORIGINS = frozenset({"collect", "backfill"})


def run_origin(run_dir: Path) -> str:
    """What a run on disk was for: recorded if known, inferred if it predates the field.

    Runs written before `Metrics.origin` existed carry no tag, and re-running
    them to acquire one would cost real OpenAlex requests for nothing. The
    inference uses the stages the run actually recorded, which is the same fact
    the tag would have stored.
    """
    import json as _json

    metrics = run_dir / "metrics.json"
    if not metrics.exists():
        return "unattributed"
    try:
        data = _json.loads(metrics.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable run is not a usable one
        return "unattributed"
    recorded = data.get("origin")
    if recorded:
        return str(recorded)
    stages = data.get("stages") or {}
    if "collect" in stages:
        return "collect"
    if "backfill" in stages:
        return "backfill"
    return "unattributed"


def iter_raw_openalex_works(origins: frozenset = CITATION_ORIGINS) -> Iterator[dict]:
    """Every OpenAlex Work written by a **collection** run.

    The 90-day backfill collected and scored 37,390 candidates and kept the
    responses verbatim, so the references for those days are already on disk.
    Re-fetching them would be paying for what we have — measured, 88% of the
    backfill's journal works carry `referenced_works` in the stored response.

    Reading every `runs/*/raw/openalex` was the earlier behaviour and it was
    not, in fact, pulling in foreign responses — the trainset builder writes no
    raw files at all, so nothing on disk today comes from anywhere but a collect
    or a backfill. The filter exists because that was true by accident: any
    future script that calls a collector to measure something would have had its
    responses join the archive silently, and "our archive" would have changed
    meaning without anything recording that it had.
    """
    import json as _json

    for raw_dir in sorted(paths.RUNS.glob("*/raw/openalex")):
        if run_origin(raw_dir.parent.parent) not in origins:
            continue
        for path in sorted(raw_dir.glob("*.json")):
            try:
                payload = _json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a truncated response is not fatal
                continue
            works = payload if isinstance(payload, list) else (payload.get("results") or [])
            for work in works:
                if isinstance(work, dict) and work.get("id"):
                    yield work


def _work_reference_record(work: dict) -> Optional[dict]:
    """A reference-base row from a raw OpenAlex Work, or None if unusable."""
    from ..collectors.base import normalize_doi, normalize_openalex_id
    from ..collectors.openalex import _arxiv_id_from_work

    refs = [
        f"openalex:{normalize_openalex_id(w)}"
        for w in (work.get("referenced_works") or [])
        if normalize_openalex_id(w)
    ]
    if not refs:
        return None

    arxiv_id = _arxiv_id_from_work(work)
    doi = normalize_doi(work.get("doi"))
    oa_id = normalize_openalex_id(work.get("id"))
    # Same work_key priority as the collector (PRD §5.2), so a row harvested
    # here and the same paper collected live are one record, not two.
    if arxiv_id:
        work_key = f"arxiv:{arxiv_id}"
    elif doi:
        work_key = f"doi:{doi}"
    elif oa_id:
        work_key = f"openalex:{oa_id}"
    else:
        return None

    return {
        "work_key": work_key,
        "date": str(work.get("publication_date") or ""),
        "published": False,
        "referenced_works": sorted(set(refs)),
        # Not written to the file: used only to recognise a paper we already
        # hold under a different key.
        "identifiers": {v.lower() for v in (arxiv_id, doi, oa_id) if v},
    }


def build_reference_base(out: Optional[Path] = None) -> dict[str, int]:
    """Write `content/graph/references.jsonl` — work_key to referenced Work IDs.

    Published items first, then scored-but-unpublished candidates found in the
    run directories. Sorted and de-duplicated so the file is byte-stable.
    """
    rows: dict[str, dict] = {}
    for item in store.iter_items():
        if item.graph.referenced_works:
            rows[item.work_key] = {
                "work_key": item.work_key,
                "date": str(item.first_published or item.updated or ""),
                "published": True,
                "referenced_works": sorted(set(item.graph.referenced_works)),
            }
    published = len(rows)

    for item in iter_run_candidates():
        if item.work_key in rows:
            continue
        rows[item.work_key] = {
            "work_key": item.work_key,
            "date": str(item.first_published or ""),
            "published": False,
            "referenced_works": sorted(set(item.graph.referenced_works)),
        }
    from_runs = len(rows) - published

    # Every identifier already claimed by a live record. A raw response has not
    # been through dedup, so the journal version of a paper we already hold as a
    # preprint arrives under its DOI and becomes a second record — which showed
    # up as coupling pairs with a Jaccard of exactly 1.0, two work_keys with
    # byte-identical reference lists. Matching on identifiers catches what the
    # work_key alone cannot.
    claimed: set[str] = set()
    for item in store.iter_items():
        for value in (item.ids.arxiv, item.ids.doi, item.ids.openalex):
            if value:
                claimed.add(value.lower())
    for item in iter_run_candidates():
        for value in (item.ids.arxiv, item.ids.doi, item.ids.openalex):
            if value:
                claimed.add(value.lower())

    # Harvested last so a live record always wins: it has been through dedup and
    # merging, and the raw response has not.
    harvested = 0
    deduped = 0
    for work in iter_raw_openalex_works():
        record = _work_reference_record(work)
        if record is None or record["work_key"] in rows:
            continue
        if record["identifiers"] & claimed:
            deduped += 1
            continue
        claimed |= record["identifiers"]
        rows[record["work_key"]] = {
            k: v for k, v in record.items() if k != "identifiers"
        }
        harvested += 1

    target = out or (paths.GRAPH / REFERENCES_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(rows[k], ensure_ascii=False, sort_keys=True) for k in sorted(rows)
    ]
    store.write_text_atomic(target, "\n".join(lines) + ("\n" if lines else ""))
    mentions = [len(r["referenced_works"]) for r in rows.values()]
    distinct = {ref for r in rows.values() for ref in r["referenced_works"]}
    return {
        "records": len(rows),
        # Populations, kept apart: what an issue carried, what was scored and
        # not published, and what the backfill's stored responses already held.
        "published": published,
        "from_run_stages": from_runs,
        "harvested_from_raw": harvested,
        "harvest_deduped": deduped,
        "reference_mentions": sum(mentions),
        "distinct_references": len(distinct),
        "median_references_per_record": (
            sorted(mentions)[len(mentions) // 2] if mentions else 0
        ),
    }


def build_work_index(out: Optional[Path] = None) -> dict[str, int]:
    """Title, authors, year and venue for every work_key we hold references for.

    The canon card needs to name the papers that cite a work, and 87 of 92
    citing items were harvested backfill records with no bibliography at all —
    the reference base stores what a paper *cites*, not what it *is*. All of it
    is already in `runs/*/raw/`, so this costs nothing.

    A separate file rather than new entries in `content/items/`: that directory
    means "published" (D69), and these records were scored and never published.
    """
    from ..collectors.base import clean_text, normalize_doi, normalize_openalex_id

    rows: dict[str, dict] = {}

    # Published items first — they carry the merged, deduped bibliography.
    for item in store.iter_items():
        rows[item.work_key] = {
            "work_key": item.work_key,
            "title": item.bibliography.title,
            "authors": [a.name for a in item.bibliography.authors[:5]],
            "year": (item.first_published.year if item.first_published else None),
            "venue": item.bibliography.primary_location.source_name,
            "doi": item.ids.doi,
            "published": True,
        }
    published = len(rows)

    for work in iter_raw_openalex_works():
        record = _work_reference_record(work)
        if record is None or record["work_key"] in rows:
            continue
        loc = (work.get("primary_location") or {}).get("source") or {}
        rows[record["work_key"]] = {
            "work_key": record["work_key"],
            "title": clean_text(work.get("display_name")),
            "authors": [
                (a.get("author") or {}).get("display_name")
                for a in (work.get("authorships") or [])[:5]
            ],
            "year": work.get("publication_year"),
            "venue": loc.get("display_name"),
            "doi": normalize_doi(work.get("doi")),
            "openalex": normalize_openalex_id(work.get("id")),
            "published": False,
        }

    target = out or (paths.GRAPH / WORK_INDEX_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [store.jsonl_line(rows[k]) for k in sorted(rows)]
    store.write_text_atomic(target, "\n".join(lines) + ("\n" if lines else ""))
    return {
        "records": len(rows),
        "from_content_items": published,
        "from_raw": len(rows) - published,
        "with_title": sum(1 for r in rows.values() if r.get("title")),
    }


def load_work_index(path: Optional[Path] = None) -> dict[str, dict]:
    p = path or (paths.GRAPH / WORK_INDEX_FILE)
    return {r["work_key"]: r for r in store.read_jsonl(p) if r.get("work_key")}


def load_reference_base(path: Optional[Path] = None) -> list[dict]:
    p = path or (paths.GRAPH / REFERENCES_FILE)
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------
# Internal citations
# --------------------------------------------------------------------------


def internal_citation_edges(items: Optional[Iterable[Item]] = None) -> list[tuple[str, str]]:
    """`cites_internal`: the cited work is itself in our archive.

    Distinguished from `cites` because it answers a different question — not
    "what does this paper build on" but "how much does this archive close on
    itself". Early on that number is near zero and saying so is the result.
    """
    items = list(items) if items is not None else list(store.iter_items())
    by_openalex = {
        f"openalex:{it.ids.openalex}": it.work_key for it in items if it.ids.openalex
    }
    out: list[tuple[str, str]] = []
    for item in items:
        for ref in item.graph.referenced_works:
            target = by_openalex.get(ref)
            if target and target != item.work_key:
                out.append((item.work_key, target))
    return sorted(set(out))


# --------------------------------------------------------------------------
# Bibliographic coupling
# --------------------------------------------------------------------------


def _parse_date(value: str) -> Optional[date]:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def compute_coupling(
    records: Optional[list[dict]] = None,
    min_shared: Optional[int] = None,
    window_days: Optional[int] = None,
) -> list[dict]:
    """Pairs sharing at least `min_shared` references, within a moving window.

    The window is not only a performance guard, though it is that — the pair
    count is quadratic and the archive grows every day. It is also the honest
    scope of the claim: "read these together" is a statement about what is
    current, and two papers ninety days apart are not each other's context.

    Both normalisations are returned. Jaccard alone hides that a paper with 80
    references has more chances to overlap than one with 12; the raw count alone
    hides that 5 shared out of 12 is a stronger signal than 5 out of 80.
    """
    records = load_reference_base() if records is None else records
    min_shared = int(cfg("citation.min_shared_references", 3) if min_shared is None else min_shared)
    window_days = int(
        cfg("citation.coupling_window_days", 90) if window_days is None else window_days
    )

    usable = []
    for r in records:
        d = _parse_date(r.get("date", ""))
        refs = set(r.get("referenced_works") or [])
        if d and len(refs) >= min_shared:
            usable.append((r["work_key"], d, refs))
    usable.sort(key=lambda t: t[1])

    # An inverted index over references keeps this near-linear in practice: only
    # pairs that share at least one reference are ever considered, instead of
    # every pair in the window.
    by_ref: dict[str, list[int]] = defaultdict(list)
    for idx, (_, _, refs) in enumerate(usable):
        for ref in refs:
            by_ref[ref].append(idx)

    shared_counts: dict[tuple[int, int], int] = defaultdict(int)
    window = timedelta(days=window_days)
    for holders in by_ref.values():
        if len(holders) < 2:
            continue
        for i, j in combinations(holders, 2):
            if abs(usable[i][1] - usable[j][1]) <= window:
                shared_counts[(i, j)] += 1

    out = []
    for (i, j), shared in shared_counts.items():
        if shared < min_shared:
            continue
        a_key, a_date, a_refs = usable[i]
        b_key, b_date, b_refs = usable[j]
        union = len(a_refs | b_refs)
        out.append({
            "a": a_key,
            "b": b_key,
            "shared": shared,
            "jaccard": round(shared / union, 6) if union else 0.0,
            "a_references": len(a_refs),
            "b_references": len(b_refs),
            "date": str(max(a_date, b_date)),
        })
    out.sort(key=lambda r: (-r["shared"], r["a"], r["b"]))
    return out


def build_coupling(out: Optional[Path] = None, **kwargs) -> dict[str, int]:
    pairs = compute_coupling(**kwargs)
    target = out or (paths.GRAPH / COUPLING_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(p, ensure_ascii=False, sort_keys=True) for p in pairs]
    store.write_text_atomic(target, "\n".join(lines) + ("\n" if lines else ""))
    return {
        "pairs": len(pairs),
        "max_shared": max((p["shared"] for p in pairs), default=0),
        "max_jaccard": max((p["jaccard"] for p in pairs), default=0.0),
    }


def top_neighbours(k: int = 3, pairs: Optional[list[dict]] = None) -> dict[str, list[dict]]:
    """The k most strongly coupled neighbours of each item.

    Ranked by shared count then Jaccard: the raw count is what a reader would
    notice, and Jaccard breaks ties in favour of the tighter pair.
    """
    pairs = compute_coupling() if pairs is None else pairs
    by_item: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_item[p["a"]].append({"other": p["b"], **p})
        by_item[p["b"]].append({"other": p["a"], **p})
    return {
        key: sorted(v, key=lambda r: (-r["shared"], -r["jaccard"], r["other"]))[:k]
        for key, v in sorted(by_item.items())
    }
