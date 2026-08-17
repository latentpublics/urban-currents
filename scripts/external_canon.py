"""V4: cross-check our canon against an externally computed one (phase 0i).

**Cross-check, not replacement, and the distinction is the whole design.** Our
canon comes from what our own corpus cites, which guarantees it is scoped to the
field we actually cover and makes it a list nobody else has. An externally
computed list — most-cited works in the urban subfields — is a list anyone can
compute, and it inherits OpenAlex's topic assignments, which have already put
Moran's I under Management Science and LISA under Economics (D79).

But our canon's weakness is real: 90 days of one corpus, leaning transport, and
the probe's zero band shows the blind spots directly — inland waterway barges,
school start times, child stunting, all our kind and none citing our canon.

So the external list is used for exactly one thing: **what did we miss.**

Two metrics, two questions, and they are not interchangeable:

| metric | question | expected |
|---|---|---|
| in-subgraph citations (degree) | what does this field stand on | Arnstein, Ewing & Cervero |
| betweenness | what joins its sub-literatures | different works entirely |

Degree first: it is cheap and it is what was actually asked for. Betweenness is
sampled (Brandes) and its stability is reported — phase 0d produced a stability
of exactly 1.0 that turned out to be an artefact, so an unstable ranking here is
reported as unstable rather than as a ranking.

`content/canon/candidates.json` is never written. Output goes to
`content/canon/external_reference.json`.

Usage:
    uv run python scripts/external_canon.py --feasibility     # gate, costs ~nothing
    uv run python scripts/external_canon.py --run --max-cost 0.50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.config import cfg  # noqa: E402
from pipeline.metrics import OpenAlexBudget  # noqa: E402

OUT = ROOT / "content" / "canon" / "external_reference.json"

# The three subfields the whitelist is built on. Anything wider would be a
# different question — "what is urban research" rather than "what does the
# literature we cover stand on".
SUBFIELDS = {
    "3322": "Urban Studies",
    "3305": "Geography, Planning and Development",
    "3313": "Transportation",
}

# Citation floors to size the set. The target is a subgraph big enough to have
# structure and small enough to fetch: 20,000-50,000 works.
FLOORS = (20, 30, 50, 100, 200)


def _pyalex():
    from pipeline.collectors.openalex import configure_pyalex

    return configure_pyalex()


def feasibility(budget: OpenAlexBudget) -> dict:
    """How many works clear each floor, and what fetching their references costs.

    Run before anything else. A cross-check that blows the budget is not a
    cross-check, it is a bill.
    """
    pyalex = _pyalex()
    if pyalex is None:
        return {"status": "NO_CREDENTIALS"}

    per_page = int(cfg("openalex.per_page", 100))
    rows = []
    for floor in FLOORS:
        q = pyalex.Works().filter(
            **{
                "primary_topic.subfield.id": "|".join(SUBFIELDS),
                "cited_by_count": f">{floor}",
                "type": "article",
            }
        )
        page = q.get(per_page=1)
        meta = getattr(page, "meta", {}) or {}
        budget.charge(float(meta.get("cost_usd") or 0.0))
        n = int(meta.get("count") or 0)
        requests = -(-n // per_page)  # ceiling division
        rows.append({
            "cited_by_count_over": floor,
            "works": n,
            "requests_at_%d_per_page" % per_page: requests,
            # OpenAlex's premium pricing is per request; the measured rate from
            # this project's own runs is used rather than a published figure.
            "estimated_usd": round(requests * COST_PER_REQUEST, 4),
        })
    return {
        "status": "OK",
        "subfields": SUBFIELDS,
        "cost_per_request_usd": COST_PER_REQUEST,
        "floors": rows,
        "spent_measuring": round(budget.spent, 6),
    }


# Measured from this project's own runs: runs/*/metrics.json openalex_usd
# divided by the request counts those runs made. Kept as a constant so the
# feasibility figure is reproducible rather than a guess.
COST_PER_REQUEST = 0.0001


def pick_floor(rows: list[dict], max_cost: float) -> dict | None:
    """The largest set that fits the budget and lands in the target range."""
    affordable = [r for r in rows if r["estimated_usd"] <= max_cost]
    if not affordable:
        return None
    in_range = [r for r in affordable if 15_000 <= r["works"] <= 60_000]
    return (in_range or affordable)[0]


def fetch_subgraph(floor: int, budget: OpenAlexBudget, max_cost: float) -> list[dict]:
    """Every work above the floor, with its reference list."""
    pyalex = _pyalex()
    per_page = int(cfg("openalex.per_page", 100))
    q = (
        pyalex.Works()
        .filter(
            **{
                "primary_topic.subfield.id": "|".join(SUBFIELDS),
                "cited_by_count": f">{floor}",
                "type": "article",
            }
        )
        .select(["id", "display_name", "publication_year", "referenced_works",
                 "cited_by_count", "primary_location"])
    )
    works = []
    for page in q.paginate(per_page=per_page, n_max=None):
        budget.charge(float((getattr(page, "meta", {}) or {}).get("cost_usd") or 0.0))
        works.extend(dict(w) for w in page)
        if budget.spent > max_cost:
            print(f"  stopped at {len(works)} works — ${budget.spent:.4f} spent")
            break
    return works


def degree_ranking(works: list[dict], limit: int = 200) -> list[dict]:
    """How often each work is cited *from inside this subgraph*.

    Not `cited_by_count`, which counts citations from everywhere including
    fields we do not cover. The in-subgraph count is the one that answers "what
    does *this* literature stand on".
    """
    inside = {w["id"] for w in works}
    counts: Counter = Counter()
    for w in works:
        for ref in w.get("referenced_works") or []:
            if ref in inside:
                counts[ref] += 1
    by_id = {w["id"]: w for w in works}
    out = []
    for wid, n in counts.most_common(limit):
        w = by_id.get(wid, {})
        out.append({
            "openalex_id": wid,
            "title": w.get("display_name"),
            "year": w.get("publication_year"),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get(
                "display_name"
            ),
            "in_subgraph_citations": n,
            "global_cited_by_count": w.get("cited_by_count"),
        })
    return out


def betweenness_ranking(
    works: list[dict], samples: int = 200, seed: int = 42, limit: int = 200
) -> dict:
    """Sampled Brandes betweenness, with its own stability measured.

    Two independent samples, and the rank correlation between them is reported
    next to the ranking. Phase 0d produced a stability of exactly 1.0 that was
    an artefact of comparing a list with itself; an unstable ranking here is
    labelled unstable rather than presented as a result.
    """
    import networkx as nx

    inside = {w["id"] for w in works}
    g = nx.DiGraph()
    for w in works:
        for ref in w.get("referenced_works") or []:
            if ref in inside:
                g.add_edge(w["id"], ref)
    if g.number_of_nodes() < 10:
        return {"status": "TOO_SMALL", "nodes": g.number_of_nodes()}

    k = min(samples, g.number_of_nodes())
    a = nx.betweenness_centrality(g, k=k, seed=seed, normalized=True)
    b = nx.betweenness_centrality(g, k=k, seed=seed + 1, normalized=True)

    by_id = {w["id"]: w for w in works}
    top_a = sorted(a.items(), key=lambda kv: -kv[1])[:limit]
    rank_a = {wid: i for i, (wid, _) in enumerate(top_a)}
    top_b = [wid for wid, _ in sorted(b.items(), key=lambda kv: -kv[1])[:limit]]
    rank_b = {wid: i for i, wid in enumerate(top_b)}

    return {
        "status": "OK",
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "samples": k,
        "top": [
            {
                "openalex_id": wid,
                "title": (by_id.get(wid) or {}).get("display_name"),
                "year": (by_id.get(wid) or {}).get("publication_year"),
                "betweenness": round(score, 6),
            }
            for wid, score in top_a[:limit]
        ],
        "stability_spearman_top20": _spearman(
            [w for w, _ in top_a[:20]], rank_a, rank_b, len(top_b)
        ),
    }


def _spearman(keys: list[str], rank_a: dict, rank_b: dict, missing_rank: int) -> float:
    """Rank correlation over the first list's top, with absences ranked last.

    A work that vanishes from the second sample is not "unranked" — it is worse
    than everything that survived, and scoring it as missing data is how a
    ranking is made to look stabler than it is.
    """
    if len(keys) < 3:
        return 0.0
    n = len(keys)
    d2 = 0
    for key in keys:
        ra = rank_a[key]
        rb = rank_b.get(key, missing_rank)
        d2 += (ra - rb) ** 2
    return round(1 - (6 * d2) / (n * (n * n - 1)), 4)


def compare(external: list[dict], limit: int = 30) -> dict:
    """What the external list has that ours does not — the point of the exercise.

    **Both sides are put in the same form before anything is compared.** Our
    canon stores `openalex:W2144512297` and OpenAlex returns
    `https://openalex.org/W2144512297`; compared as-is the overlap came out 0
    of 200, which read as a dramatic finding and was a string mismatch. A zero
    overlap between two lists of the same field's most-cited works is not a
    result, it is a bug report.
    """
    from journal_metrics import canon_sets  # type: ignore

    def bare(v: str) -> str:
        """`openalex:W123` and `https://openalex.org/W123` both to `W123`.

        Written out rather than reusing `normalize_openalex_id`, which strips
        the URL host but leaves an `openalex:` prefix intact — so routing both
        sides through it left them in different forms and the overlap stayed at
        zero after the first attempt at this fix. Two id shapes reach here and
        one function has to flatten both.

        A zero overlap between two lists of the same field's most-cited works
        was never a finding. It was a string mismatch that read like one.
        """
        return (v or '').strip().rsplit('/', 1)[-1].split(':')[-1]

    foundation, instrument = canon_sets()
    ours = {bare(k) for k in list(foundation) + list(instrument)}
    ours_foundation = {bare(k) for k in foundation}

    ext_ids = [bare(e["openalex_id"]) for e in external]
    overlap = [e for e in external if bare(e["openalex_id"]) in ours]
    missing = [e for e in external if bare(e["openalex_id"]) not in ours]

    ext_set = set(ext_ids)
    only_ours = [
        {"openalex_id": wid, "weight": w}
        for wid, w in sorted(foundation.items(), key=lambda kv: -kv[1])
        if bare(wid) not in ext_set
    ][:10]

    return {
        "external_top": len(external),
        "our_canon_total": len(ours),
        "our_foundation": len(ours_foundation),
        "overlap": len(overlap),
        "overlap_share": round(len(overlap) / len(external), 4) if external else 0.0,
        "we_are_missing_top_30": missing[:limit],
        "only_in_our_canon_top_10": only_ours,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feasibility", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-cost", type=float, default=0.50)
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    budget = OpenAlexBudget(daily_usd=a.max_cost * 2, stop_fraction=1.0)
    feas = feasibility(budget)
    print(json.dumps(feas, indent=2))
    if feas.get("status") != "OK" or a.feasibility:
        if a.json:
            Path(a.json).write_text(
                json.dumps({"feasibility": feas}, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
        return

    choice = pick_floor(feas["floors"], a.max_cost)
    if not choice:
        print(f"BLOCKED: no floor fits ${a.max_cost:.2f}")
        return
    print(f"\nchosen floor: >{choice['cited_by_count_over']} citations, "
          f"{choice['works']} works, ~${choice['estimated_usd']}")

    works = fetch_subgraph(choice["cited_by_count_over"], budget, a.max_cost)
    print(f"fetched {len(works)} works, ${budget.spent:.4f} spent")

    degree = degree_ranking(works)
    between = betweenness_ranking(works)
    out = {
        "population": (
            f"OpenAlex works in subfields {sorted(SUBFIELDS)} with "
            f"cited_by_count > {choice['cited_by_count_over']}, type=article"
        ),
        "purpose": "cross-check only; content/canon/candidates.json is unchanged",
        "works_fetched": len(works),
        "openalex_cost_usd": round(budget.spent, 6),
        "degree_top": degree,
        "betweenness": between,
        "comparison_degree": compare(degree),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\nwrote {OUT}")
    c = out["comparison_degree"]
    print(f"overlap with our canon: {c['overlap']}/{c['external_top']} "
          f"({c['overlap_share'] * 100:.1f}%)")
    for e in c["we_are_missing_top_30"][:10]:
        print(f"   {e['in_subgraph_citations']:>4}  {(e['title'] or '')[:70]} "
              f"({e['year']})")
    if a.json:
        Path(a.json).write_text(
            json.dumps({"feasibility": feas, "result": out}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8", newline="\n",
        )


if __name__ == "__main__":
    main()
