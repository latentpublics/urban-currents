"""Metapath projections and centrality (phase 0d, Q4).

Analysis only. Nothing here publishes, renders, or enters an issue.

A metapath projection collapses a two-hop path through items into a direct edge
between the things at each end: two methods that appeared in the same paper
become adjacent, and betweenness over that projection asks which method sits on
the routes between otherwise separate clusters.

**Betweenness is reported next to degree on purpose.** If the two rankings agree,
betweenness is telling us nothing degree did not, and that is a result rather
than a disappointment — it means the graph has no brokers, only hubs.

**And stability is measured before anything is believed.** Ranks computed on a
graph this young can be an artefact of which papers happened to appear. Spearman
correlation of the top ranks across 30, 60 and 90 day windows is the check: if a
name is at the top only in one window, it is not a finding.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from itertools import combinations
from typing import Any, Iterable, Optional

from .. import store
from ..models import Item

METAPATHS = {
    # method -[in]- item -[in]- method
    "method-method": ("methods", "methods"),
    # data -[in]- item -[in]- method
    "data-method": ("data", "methods"),
}


def _items_in_window(days: Optional[int], today: Optional[date] = None) -> list[Item]:
    items = list(store.iter_items())
    if not days:
        return items
    dates = [it.first_published for it in items if it.first_published]
    if not dates:
        return items
    end = today or max(dates)
    start = end - timedelta(days=days)
    return [it for it in items if it.first_published and start <= it.first_published <= end]


def project(
    metapath: str, days: Optional[int] = None, items: Optional[Iterable[Item]] = None
) -> dict[tuple[str, str], int]:
    """Weighted edges between endpoint entities that shared an item."""
    if metapath not in METAPATHS:
        raise ValueError(f"unknown metapath {metapath!r}; try {sorted(METAPATHS)}")
    left, right = METAPATHS[metapath]
    pool = list(items) if items is not None else _items_in_window(days)

    weights: dict[tuple[str, str], int] = defaultdict(int)
    for item in pool:
        lefts = {e.label for e in getattr(item.entities, left)}
        rights = {e.label for e in getattr(item.entities, right)}
        if left == right:
            for a, b in combinations(sorted(lefts), 2):
                weights[(a, b)] += 1
        else:
            for a in sorted(lefts):
                for b in sorted(rights):
                    if a != b:
                        weights[(a, b)] += 1
    return dict(weights)


def researcher_projection(days: Optional[int] = None) -> dict[tuple[str, str], int]:
    """researcher -[cites]- work -[cited by]- researcher.

    Two researchers are linked when their items cite the same work. Depends
    entirely on reference coverage, which Q1 measured at 86% on the journal path
    and 5% on the arXiv path — so this projection describes journal authors and
    is close to blind to the preprint half of the archive.
    """
    pool = _items_in_window(days)
    by_ref: dict[str, set[str]] = defaultdict(set)
    for item in pool:
        people = [e.label for e in item.entities.people]
        for ref in item.graph.referenced_works:
            by_ref[ref].update(people)

    weights: dict[tuple[str, str], int] = defaultdict(int)
    for sharers in by_ref.values():
        for a, b in combinations(sorted(sharers), 2):
            weights[(a, b)] += 1
    return dict(weights)


def centrality(
    weights: dict[tuple[str, str], int], min_degree: int = 3
) -> dict[str, Any]:
    """Betweenness and degree over a projection, after a degree floor.

    The floor matters: a node appearing in one paper has degree 1 and sits on no
    route, but there are many of them and they dominate the ranking's tail.
    """
    import networkx as nx

    g = nx.Graph()
    for (a, b), w in weights.items():
        g.add_edge(a, b, weight=w)

    if g.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "betweenness": [], "degree": [], "note": "empty"}

    kept = [n for n, d in g.degree() if d >= min_degree]
    sub = g.subgraph(kept).copy()
    if sub.number_of_nodes() == 0:
        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "nodes_after_floor": 0,
            "betweenness": [],
            "degree": [],
            "note": f"no node reaches degree {min_degree}",
        }

    btw = nx.betweenness_centrality(sub, weight=None, normalized=True)
    deg = dict(sub.degree())
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "nodes_after_floor": sub.number_of_nodes(),
        "min_degree": min_degree,
        "betweenness": sorted(
            ({"node": n, "score": round(v, 6)} for n, v in btw.items()),
            key=lambda r: (-r["score"], r["node"]),
        ),
        "degree": sorted(
            ({"node": n, "degree": d} for n, d in deg.items()),
            key=lambda r: (-r["degree"], r["node"]),
        ),
    }


def spearman(a: list[str], b: list[str]) -> Optional[float]:
    """Rank correlation over the union of two ordered lists.

    Names missing from one side are given the rank just past its end, which
    treats "absent" as "worse than anything present" rather than dropping the
    comparison — the point is whether a top-10 survives a change of window, and
    a name vanishing entirely is the strongest possible instability.
    """
    from scipy.stats import spearmanr

    universe = sorted(set(a) | set(b))
    if len(universe) < 3:
        return None
    ra = [a.index(n) if n in a else len(a) for n in universe]
    rb = [b.index(n) if n in b else len(b) for n in universe]
    if len(set(ra)) < 2 or len(set(rb)) < 2:
        return None
    rho, _ = spearmanr(ra, rb)
    return None if rho != rho else round(float(rho), 4)  # NaN guard


def stability(metapath: str, windows=(30, 60, 90), top: int = 10, min_degree: int = 3) -> dict:
    """Top-`top` betweenness ranking across window lengths, and its correlation."""
    rankings: dict[int, list[str]] = {}
    detail: dict[int, dict] = {}
    populations: dict[int, int] = {}
    for w in windows:
        pool = _items_in_window(w)
        populations[w] = len(pool)
        weights = (
            researcher_projection(w)
            if metapath == "researcher-researcher"
            else project(metapath, days=w)
        )
        c = centrality(weights, min_degree=min_degree)
        detail[w] = {
            "items_in_window": len(pool),
            "nodes": c["nodes"],
            "edges": c["edges"],
            "nodes_after_floor": c.get("nodes_after_floor", 0),
        }
        rankings[w] = [r["node"] for r in c["betweenness"][:top]]

    pairs = {}
    for i, j in combinations(windows, 2):
        pairs[f"{i}d_vs_{j}d"] = spearman(rankings[i], rankings[j])

    # A correlation of 1.0 between two windows that selected the *same items* is
    # arithmetic, not stability. The archive spans days, not months, so every
    # window here can contain all of it — and a number that looks like a strong
    # result while measuring nothing is worse than no number.
    degenerate = len(set(populations.values())) == 1
    return {
        "windows": detail,
        "top_by_window": rankings,
        "spearman": pairs,
        "windows_are_distinct": not degenerate,
        "note": (
            None if not degenerate else
            f"All windows selected the same {next(iter(populations.values()))} items: "
            f"the archive is shorter than the shortest window, so the correlations "
            f"below compare a ranking with itself and are not evidence of stability."
        ),
    }


# --------------------------------------------------------------------------
# Brokerage events
# --------------------------------------------------------------------------


def brokerage_events(days: Optional[int] = None) -> dict[str, Any]:
    """Days on which two overlay tags met in one item for the first time.

    The publishable form of this axis is an event — "street view and
    agent-based simulation appeared together for the first time" — not a ranking.
    Before designing that, the question is whether it happens at a usable rate:
    too rare and there is no column, too often and it means nothing.
    """
    items = sorted(
        (it for it in _items_in_window(days) if it.first_published),
        key=lambda it: (it.first_published, it.work_key),
    )
    seen: set[tuple[str, str]] = set()
    per_day: dict[str, int] = defaultdict(int)
    examples: list[dict] = []

    for item in items:
        tags = sorted(
            {e.label for e in item.entities.methods}
            | {e.label for e in item.entities.data}
            | {e.label for e in item.entities.tools}
        )
        for a, b in combinations(tags, 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            day = str(item.first_published)
            per_day[day] += 1
            if len(examples) < 10:
                examples.append({"date": day, "pair": [a, b], "work_key": item.work_key})

    days_observed = sorted({str(it.first_published) for it in items})
    counts = [per_day.get(d, 0) for d in days_observed]
    return {
        "days_observed": len(days_observed),
        "first_meetings": sum(counts),
        "per_day_mean": round(sum(counts) / len(counts), 2) if counts else 0,
        "per_day_max": max(counts, default=0),
        "per_day_min": min(counts, default=0),
        "per_day": {d: per_day.get(d, 0) for d in days_observed},
        "examples": examples,
        "note": (
            "Every pair is a first meeting while the archive is young, so this "
            "rate falls as it fills. It is an upper bound, not a steady state."
        ),
    }
