"""T2/T3: journal priors and paper-level canon affinity (phase 0g).

Two ranking signals for the journal path, which currently has none — every
whitelist article scores relevance 1.0 by membership, so when the rebuilt list
takes candidates from about 51 a day to about 90 for the same 12 slots, the
twelve are chosen arbitrarily.

**Neither is applied.** `selection.journal_ranking` stays `placeholder`. YJUN has
labelled 30 items and `runs/labels/relevance.jsonl` records the `rank` each was
shown at; changing the ranking now would make those labels unattributable to any
ranking. This computes, compares, and reports the diff.

---------------------------------------------------------------------------
T2 — venue prior, normalised inside its subfield
---------------------------------------------------------------------------

OpenAlex ships the impact-factor family on the Source object —
`summary_stats.2yr_mean_citedness`, `h_index`, `i10_index` — so no subscription
is needed. But **the raw value must not be used for ranking.** Citation cultures
differ by field: transport and environmental engineering journals sit
structurally higher than planning and urban design ones, so ranking on the raw
number rebuilds precisely the transport bias the last two batches removed.

Each source is therefore scored as a **percentile within its own subfield**. The
Journal of Urban Design is judged against Building and Construction, TR Part B
against Transportation.

**And this is a prior, not a verdict.** An impact factor measures a journal's
average influence, not whether this paper deserves a card, and it is
structurally unkind to young, regional and open-access journals. It belongs as
one input among several, never as a gate.

---------------------------------------------------------------------------
T3-2 — canon affinity, at the level of the paper
---------------------------------------------------------------------------

Better than any journal metric, because it looks at the paper itself: a paper
citing Ewing and Cervero and Anselin is our kind; one citing only heritage
studies is not. We already have both halves — the foundation canon and every
item's reference list.

Usage:
    uv run python scripts/journal_metrics.py --json runs/journal_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from pipeline.graph.citation import load_reference_base  # noqa: E402
from pipeline.graph.daily_canon import load_resolved  # noqa: E402

VOCAB = ROOT / "vocab" / "sources"
CANON = ROOT / "content" / "canon" / "candidates.json"


def venue_alias_map() -> dict[str, str]:
    """Previous source ID -> current source ID, for aggregation only."""
    path = VOCAB / "venue_aliases.yaml"
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        p["previous"]: p["current"]
        for p in (doc.get("pairs") or [])
        if p.get("previous") and p.get("current")
    }


def source_ids_of_interest() -> dict[str, str]:
    """Every source in the whitelist, the rebuilds, and the gap candidates."""
    out: dict[str, str] = {}
    for name in ("journals.yaml", "journals.rebuilt.yaml", "journals.rebuilt.v2.yaml",
                 "journal_gap_candidates.yaml"):
        path = VOCAB / name
        if not path.exists():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key in ("sources", "candidates"):
            for s in doc.get(key) or []:
                if s.get("id"):
                    out.setdefault(s["id"], s.get("name") or s["id"])
    return out


def dominant_subfields() -> dict[str, str]:
    """Each source's modal subfield, from the works we have resolved.

    Read from the citation record rather than fetched: a journal's subfield is
    where its papers actually sit, and 22,910 resolved works already say so.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in load_resolved().values():
        if row.get("venue_id") and row.get("subfield_id"):
            counts[row["venue_id"]][row["subfield_id"]] += 1
    return {sid: c.most_common(1)[0][0] for sid, c in counts.items() if c}


def fetch_sources(ids: list[str], batch: int = 50) -> tuple[dict[str, dict], float]:
    from pipeline.collectors.openalex import configure_pyalex

    pyalex = configure_pyalex()
    if pyalex is None:
        return {}, 0.0
    out: dict[str, dict] = {}
    cost = 0.0
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        res = pyalex.Sources().filter(openalex_id="|".join(chunk)).get(per_page=batch)
        cost += float((getattr(res, "meta", {}) or {}).get("cost_usd") or 0.0)
        for s in res:
            out[(s.get("id") or "").rsplit("/", 1)[-1]] = dict(s)
    return out, cost


def percentile(value: float, population: list[float]) -> float:
    """Share of the population at or below `value`. Empty population -> 0.5."""
    if not population:
        return 0.5
    below = sum(1 for v in population if v <= value)
    return round(below / len(population), 4)


# --------------------------------------------------------------------------
# T3-2 canon affinity
# --------------------------------------------------------------------------


def canon_sets() -> tuple[dict[str, float], dict[str, float]]:
    """Foundation and instrument canon, each as openalex_id -> weight."""
    if not CANON.exists():
        return {}, {}
    doc = json.loads(CANON.read_text(encoding="utf-8"))
    foundation, instrument = {}, {}
    for c in doc.get("candidates") or []:
        target = foundation if c.get("class") == "foundation" else instrument
        target[c["openalex_id"]] = float(c.get("weighted_score") or 0.0)
    return foundation, instrument


def cites_canon(refs: list[str], canon: dict[str, float]) -> bool:
    """Does this paper cite the field's foundation at all — the binary question.

    The probe's finding is that this is where the information is. Across its 30
    band-stratified labels, papers citing no foundational work were
    `drop_not_our_kind` 40% of the time against 10% for those that cite at least
    one; the keep rate barely moved (0.50 against 0.65). The grades *above* zero
    did not separate the bands the way the zero line did.

    **A zero measures two things, and only one of them is about the paper.** Our
    canon is 90 days of one corpus and it leans towards transport, so a paper can
    score zero by being outside the field or by being in a corner of the field
    the canon does not reach. Five of the six zero-band keeps are the second
    kind: inland waterway barges, staggered school start times, lane-change
    prediction, immigration and crime rates, child stunting. All of them are our
    kind and none of them cites our canon. Anything built on this flag inherits
    that ambiguity, and widening the canon narrows it.

    **n = 30 and Fisher's exact gives p = 0.14.** Direction and mechanism, not
    significance.
    """
    return any(r in canon for r in refs)


def canon_affinity(
    refs: list[str], canon: dict[str, float], normalise: str = "none"
) -> float:
    """How much of this paper's bibliography is the field's foundation.

    Weighted by the canon entry's own weighted score throughout, so citing Ewing
    and Cervero counts for more than citing a work our corpus touched twice.

    **The default was `sqrt` and the labels moved it to `none`.** The theory was
    that raw overlap rewards long bibliographies, so the square root split the
    difference against dividing by the reference count. In practice it inverted
    the pair that matters: "Beyond the Western paradigm" (2 canon hits in 5
    references) scored 89.4 and "Multilevel SEM of walkability" (7 in 63) scored
    88.2, which says a paper citing seven foundational works is less embedded in
    the field than one citing two. It is not.

    Measured over 69 labelled items that have reference lists, AUC for keep
    against `not_our_kind`:

    | normalisation        | AUC   | the pair above |
    |----------------------|-------|----------------|
    | `none` (weighted sum)| 0.673 | 142.8 vs 64.5 — correct |
    | `sqrt`               | 0.644 | 88.2 vs 89.4 — inverted |
    | raw hit count        | 0.644 | 7 vs 2 — correct, ignores length |
    | hits / log(refs)     | 0.642 | correct |
    | `linear` (share)     | 0.632 | worst, punishes long lists |

    The feared bias did not appear: a longer bibliography that reaches more
    foundational works turns out to *be* more embedded, which is what the
    unnormalised sum says. The other modes are kept because the comparison is
    worth being able to re-run, not because any of them is a fallback.
    """
    if not refs:
        return 0.0
    hits = [canon[r] for r in refs if r in canon]
    if not hits:
        return 0.0
    total = sum(hits)
    n = len(refs)
    if normalise == "linear":
        return round(total / n, 6)
    if normalise == "sqrt":
        return round(total / (n ** 0.5), 6)
    if normalise == "log":
        return round(total / math.log(n + math.e), 6)
    return round(total, 6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--max-cost", type=float, default=0.10)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    aliases = venue_alias_map()
    names = source_ids_of_interest()
    subfield_of = dominant_subfields()
    ids = sorted(names)
    print(f"sources of interest: {len(ids)}")

    sources, cost = fetch_sources(ids)
    print(f"resolved {len(sources)} sources, cost ${cost:.4f} (cap ${a.max_cost})")
    if cost > a.max_cost:
        print("WARNING: cost ceiling exceeded")

    # Raw metrics.
    rows: dict[str, dict] = {}
    for sid, s in sources.items():
        stats = s.get("summary_stats") or {}
        rows[sid] = {
            "id": sid,
            "name": s.get("display_name") or names.get(sid),
            "subfield_id": subfield_of.get(sid),
            "two_year_mean_citedness": round(float(stats.get("2yr_mean_citedness") or 0.0), 4),
            "h_index": int(stats.get("h_index") or 0),
            "i10_index": int(stats.get("i10_index") or 0),
            "works_count": int(s.get("works_count") or 0),
            "is_core": bool(s.get("is_core")),
            "is_in_doaj": bool(s.get("is_in_doaj")),
            "type": s.get("type"),
        }

    # Percentile inside the subfield — the point of the exercise.
    by_subfield: dict[str, list[float]] = defaultdict(list)
    for r in rows.values():
        if r["subfield_id"]:
            by_subfield[r["subfield_id"]].append(r["two_year_mean_citedness"])
    for r in rows.values():
        pop = by_subfield.get(r["subfield_id"] or "", [])
        r["prestige_pct_in_subfield"] = percentile(r["two_year_mean_citedness"], pop)
        r["subfield_population"] = len(pop)

    # Verification: does normalising actually move the transport share?
    def transport_share(ranked: list[dict], n: int = 30) -> float:
        top = ranked[:n]
        sized = [r for r in top if r["subfield_id"]]
        if not sized:
            return 0.0
        return round(sum(1 for r in sized if r["subfield_id"] == "3313") / len(sized), 4)

    with_subfield = [r for r in rows.values() if r["subfield_id"]]
    by_raw = sorted(with_subfield, key=lambda r: -r["two_year_mean_citedness"])
    by_pct = sorted(with_subfield, key=lambda r: -r["prestige_pct_in_subfield"])

    # T3-2.
    foundation, instrument = canon_sets()
    both = {**instrument, **foundation}
    affinity_rows = []
    for record in load_reference_base():
        refs = record.get("referenced_works") or []
        affinity_rows.append({
            "work_key": record["work_key"],
            "references": len(refs),
            "foundation_only": canon_affinity(refs, foundation),
            "with_instruments": canon_affinity(refs, both),
            "foundation_hits": sum(1 for r in refs if r in foundation),
            "instrument_hits": sum(1 for r in refs if r in instrument),
            "raw": canon_affinity(refs, foundation, "none"),
            "linear": canon_affinity(refs, foundation, "linear"),
            # Unweighted: hits per sqrt(references). Weighting by the canon
            # entry's own score lets one hit on Ewing and Cervero (weight ~92)
            # outscore eighteen hits spread over lesser entries, so a
            # six-reference paper can top the list. Counting hits removes that.
            "hits_sqrt": round(
                sum(1 for r in refs if r in foundation) / (len(refs) ** 0.5), 6
            ) if refs else 0.0,
        })
    affinity_rows.sort(key=lambda r: -r["foundation_only"])

    # Does the normalisation actually control for bibliography length?
    def corr(key: str) -> float:
        xs = [r["references"] for r in affinity_rows if r["references"]]
        ys = [r[key] for r in affinity_rows if r["references"]]
        if len(xs) < 3:
            return 0.0
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        return round(num / den, 4) if den else 0.0

    result = {
        "population_sources": len(ids),
        "resolved_sources": len(sources),
        "openalex_cost_usd": round(cost, 6),
        "venue_alias_pairs": len(aliases),
        "prior_is_not_a_verdict": (
            "2yr_mean_citedness measures a journal's average influence, not "
            "whether this paper deserves a card, and is structurally unkind to "
            "young, regional and open-access journals. A prior, never a gate."
        ),
        "transport_share_top30": {
            "raw_2yr_mean_citedness": transport_share(by_raw),
            "prestige_pct_in_subfield": transport_share(by_pct),
        },
        "top_raw": [
            {k: r[k] for k in ("name", "subfield_id", "two_year_mean_citedness")}
            for r in by_raw[:30]
        ],
        "top_normalised": [
            {k: r[k] for k in ("name", "subfield_id", "prestige_pct_in_subfield",
                               "two_year_mean_citedness")}
            for r in by_pct[:30]
        ],
        "length_bias_correlation": {
            "raw": corr("raw"),
            "sqrt": corr("foundation_only"),
            "linear": corr("linear"),
        },
        "affinity_top": affinity_rows[:25],
        "affinity_bottom": [r for r in affinity_rows if r["references"] >= 20][-15:],
        "sources": list(rows.values()),
    }

    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )

    ts = result["transport_share_top30"]
    print("\ntransport share of the top 30:")
    print(f"   raw 2yr_mean_citedness      : {ts['raw_2yr_mean_citedness']:.1%}")
    print(f"   prestige_pct_in_subfield    : {ts['prestige_pct_in_subfield']:.1%}")
    print("\nbibliography-length correlation (lower is better):")
    for k, v in result["length_bias_correlation"].items():
        print(f"   {k:<8}: {v:+.4f}")


if __name__ == "__main__":
    main()
