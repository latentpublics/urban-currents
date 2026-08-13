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

    target = out or (paths.GRAPH / REFERENCES_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(rows[k], ensure_ascii=False, sort_keys=True) for k in sorted(rows)
    ]
    store.write_text_atomic(target, "\n".join(lines) + ("\n" if lines else ""))
    return {
        "records": len(rows),
        "published": published,
        "unpublished": len(rows) - published,
        "reference_mentions": sum(len(r["referenced_works"]) for r in rows.values()),
    }


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
