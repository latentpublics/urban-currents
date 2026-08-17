"""Should the canon absorb the physical-activity and walkability literature? (X8)

**The rule is written here, above the code that measures it, and it was written
before anything was run.** Three signals pointed at the same hole and none of
them is enough on its own to widen the list our whole affinity signal rests on:

1. **The probe's zero band.** Five of the six items that cite no canon at all and
   were still `keep` were our kind of paper sitting in a corner the canon does
   not reach — inland waterway barges, staggered school start times, child
   stunting.
2. **The external cross-check (V4).** Overlap with an externally computed top 200
   is 14.5%, and the misses cluster in physical activity and walkability.
3. **An editorial correction.** YJUN moved a labelled item from `drop_weak` to
   `keep` and said why: public health, physical activity and walkability matter
   thematically to urban science. That is the one thing the other two cannot
   supply — a person confirming the hole is a scope question and not an error.

## The pre-registered rule

Merge `content/canon/external_reference.json`'s ranked works — **only those that
pass the existing scope filter** — into our own candidates, then measure three
things. **Adopt only if all three pass.**

| # | measurement | adopt if |
|---|---|---|
| A | separation of `cites_canon` between `keep` and `not_our_kind` over the 148 relevance labels | **widens or holds** |
| B | count of labelled items citing no canon at all (the zero band) | **falls** |
| C | the `canon` anchor lines across the five issues that have them | **does not tilt further to transport** |

**A is the veto and here is why it has to be.** Widening a list makes it easier
to hit, and a canon that everything cites has stopped being a signal and become
a background. If the separation between our kind and not-our-kind narrows, the
merge bought coverage by spending the discrimination the flag exists for. That
outcome is a rejection even if B and C both improve — which is exactly the shape
of result that tempts a person to renegotiate the rule afterwards, hence writing
it down first.

**B is directional, not sufficient.** The zero band shrinks by construction when
the canon grows; the question is whether it shrinks because real gaps got covered.
B passing means nothing on its own, and B failing would mean the merge did not
even do the thing it was for.

**C guards against the known defect of the source.** Our canon already leans
transport, and the external list is ranked by citations inside urban subfields
where transport is the largest literature. Fixing a walkability hole by adding
more transport would make the tilt worse while appearing to fix it.

## What this does not do

`content/canon/candidates.json` is **never written**. If the merge is adopted the
result goes to a separate file and `citation.canon_file` points at it, so
reverting is one line in config. Nothing here trains anything, changes a label,
or touches ranking.

Usage:
    uv run python scripts/canon_merge_test.py --fetch    # resolve external ids
    uv run python scripts/canon_merge_test.py --measure  # A, B and C
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline import paths, store  # noqa: E402
from pipeline.config import cfg, journals_vocab  # noqa: E402
from pipeline.graph.canon import _in_scope  # noqa: E402

CANON_DIR = paths.CONTENT / "canon"
OURS = CANON_DIR / "candidates.json"
EXTERNAL = CANON_DIR / "external_reference.json"
RESOLVED = paths.RUNS / "canon_merge_external_resolved.json"
MERGED = CANON_DIR / "candidates_merged.json"
REPORT = paths.RUNS / "canon_merge_test.json"

# Topics whose presence marks an anchor as transport, for measurement C. Matched
# against the OpenAlex topic name, lowercased. Deliberately broad: C asks whether
# the tilt got worse, so over-counting transport on both sides is safe and
# under-counting it is not.
TRANSPORT_MARKERS = (
    "transport",
    "travel",
    "traffic",
    "mobility",
    "commut",
    "vehicle",
    "road",
    "transit",
    "car ",
    "driving",
)


# --------------------------------------------------------------------------
# Fetching the external works' scope metadata
# --------------------------------------------------------------------------


def fetch_external() -> dict[str, Any]:
    """Resolve the external top-N so the scope filter has topics to read.

    `external_reference.json` stores rank, title, venue and citation counts but
    not topic or subfield, and the scope rule excludes at topic level. One batch
    request per 50 works.
    """
    from pipeline.collectors.openalex import configure_pyalex

    import pyalex

    configure_pyalex()

    doc = json.loads(EXTERNAL.read_text(encoding="utf-8"))
    rows = doc.get("degree_top") or []
    ids = [r["openalex_id"].rsplit("/", 1)[-1] for r in rows]

    works: dict[str, dict] = {}
    cost = 0.0
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        res = pyalex.Works().filter(openalex_id="|".join(chunk)).get(per_page=50)
        cost += float((getattr(res, "meta", {}) or {}).get("cost_usd") or 0.0)
        for w in res:
            works[w["id"]] = dict(w)
        print(f"  resolved {len(works)}/{len(ids)}")

    out = {
        "fetched": len(works),
        "requested": len(ids),
        "cost_usd": round(cost, 6),
        "works": works,
    }
    RESOLVED.parent.mkdir(parents=True, exist_ok=True)
    RESOLVED.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"→ {RESOLVED}  (${cost:.4f})")
    return out


# --------------------------------------------------------------------------
# The merge
# --------------------------------------------------------------------------


def norm_id(value: str) -> str:
    """Every OpenAlex id in the canonical `openalex:W...` form.

    Three formats meet here and only one of them is ours. `candidates.json` and
    the reference base carry the prefixed form the schema requires; the external
    file and the API both use the full URL. The first version of this script
    merged 200 URL-keyed entries into a prefix-keyed canon, and every
    measurement came back byte-identical before and after — a merge that added
    nothing, reported as three passes with no effect. The giveaway was an
    overlap of 0 where V4 had measured 29.
    """
    bare = (value or "").rsplit("/", 1)[-1]
    return f"openalex:{bare}" if bare else ""


def build_merged() -> dict[str, Any]:
    """Our candidates plus the in-scope external works, each tagged by source."""
    ours_doc = json.loads(OURS.read_text(encoding="utf-8"))
    ours = {c["openalex_id"]: dict(c) for c in ours_doc["candidates"]}
    resolved = json.loads(RESOLVED.read_text(encoding="utf-8"))["works"]
    external_doc = json.loads(EXTERNAL.read_text(encoding="utf-8"))
    ranks = {
        norm_id(r["openalex_id"]): i
        for i, r in enumerate(external_doc.get("degree_top") or [])
    }

    whitelist = {
        s.get("openalex_id") for s in journals_vocab().get("sources", []) if s.get("openalex_id")
    }
    mode = str(cfg("citation.canon_scope_mode", "subfield"))

    # External-only entries need a weight on our scale. Ours is an archive
    # citation count with recency decay; theirs is a citation count inside a
    # 48,753-work subgraph. The two are not comparable and inventing a mapping
    # would put made-up numbers into the affinity signal, so every external-only
    # entry gets the **median** of our foundation weights: it asserts "this is
    # foundational" and nothing more. Both measurements that decide adoption (A
    # and B) read `cites_canon`, which is membership and ignores weight
    # entirely — so this choice cannot flip the verdict, only the magnitudes in
    # the secondary AUC.
    foundation_weights = sorted(
        float(c.get("weighted_score") or 0.0)
        for c in ours.values()
        if c.get("class") == "foundation"
    )
    median_weight = (
        foundation_weights[len(foundation_weights) // 2] if foundation_weights else 1.0
    )

    merged = {k: {**v, "source": "ours"} for k, v in ours.items()}
    out_of_scope = []
    added = 0
    for raw_id, work in resolved.items():
        oid = norm_id(raw_id)
        if oid in merged:
            merged[oid]["source"] = "both"
            continue
        if not _in_scope(work, whitelist, mode):
            out_of_scope.append({
                "openalex_id": oid,
                "title": (work.get("title") or "")[:90],
                "topic": ((work.get("primary_topic") or {}).get("display_name")),
            })
            continue
        primary = work.get("primary_topic") or {}
        merged[oid] = {
            "openalex_id": oid,
            "title": work.get("title") or "",
            "year": work.get("publication_year"),
            "publication_date": work.get("publication_date"),
            "venue": ((work.get("primary_location") or {}).get("source") or {}).get(
                "display_name"
            ),
            "authors": [
                (a.get("author") or {}).get("display_name")
                for a in (work.get("authorships") or [])[:2]
            ],
            "topic": primary.get("display_name"),
            "topic_id": (primary.get("id") or "").rsplit("/", 1)[-1] or None,
            "subfield": ((primary.get("subfield") or {}).get("display_name")),
            "openalex_cited_by_count": work.get("cited_by_count"),
            # Not from our archive. Named so nothing mistakes it for one.
            "archive_citations": 0,
            "external_rank": ranks.get(oid),
            "weighted_score": median_weight,
            "class": "foundation",
            "class_basis": "external:degree",
            "source": "external",
        }
        added += 1

    doc = {
        "generated_from": "candidates.json + external_reference.json (phase 0k, X8)",
        "ours": len(ours),
        "external_considered": len(resolved),
        "external_added": added,
        "external_out_of_scope": len(out_of_scope),
        "overlap": sum(1 for c in merged.values() if c["source"] == "both"),
        "total": len(merged),
        "external_weight_note": (
            "external-only entries carry the median foundation weight; their own "
            "citation counts are on a different scale and are not converted"
        ),
        "out_of_scope_sample": out_of_scope[:15],
        "candidates": sorted(merged.values(), key=lambda c: -float(c.get("weighted_score") or 0)),
    }
    return doc


# --------------------------------------------------------------------------
# Statistics, written out (no scipy)
# --------------------------------------------------------------------------


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p for [[a, b], [c, d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0

    def hyper(x: int) -> float:
        return (
            math.comb(a + b, x)
            * math.comb(c + d, a + c - x)
            / math.comb(n, a + c)
        )

    observed = hyper(a)
    lo = max(0, a + c - (c + d))
    hi = min(a + b, a + c)
    total = 0.0
    for x in range(lo, hi + 1):
        p = hyper(x)
        if p <= observed * (1 + 1e-9):
            total += p
    return round(min(1.0, total), 6)


def auc(positive: list[float], negative: list[float]) -> Optional[float]:
    """Probability a random positive outranks a random negative. Ties count half."""
    if not positive or not negative:
        return None
    wins = 0.0
    for p in positive:
        for q in negative:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return round(wins / (len(positive) * len(negative)), 4)


# --------------------------------------------------------------------------
# The three measurements
# --------------------------------------------------------------------------


def _labelled_with_refs() -> list[dict]:
    """Relevance labels joined to their reference lists.

    `relevance.jsonl` only — the probe is band-stratified over the very quantity
    under test and pooling the two would be the misuse the code raises on.
    """
    from pipeline.labeling import load_labels
    from pipeline.graph.citation import load_reference_base

    refs = {
        r["work_key"]: (r.get("referenced_works") or []) for r in load_reference_base()
    }
    out = []
    for row in load_labels("relevance"):
        key = row.get("work_key")
        if not key:
            continue
        out.append({
            "work_key": key,
            "label": row.get("label"),
            "refs": refs.get(key) or [],
            "source": row.get("source"),
        })
    return out


def _canon_ids(doc: dict) -> set[str]:
    return {
        c["openalex_id"] for c in doc["candidates"] if c.get("class") == "foundation"
    }


def _weights(doc: dict) -> dict[str, float]:
    return {
        c["openalex_id"]: float(c.get("weighted_score") or 0.0)
        for c in doc["candidates"]
        if c.get("class") == "foundation"
    }


def measure_a_and_b(labels: list[dict], canon: set[str], weights: dict[str, float]) -> dict:
    """A: does `cites_canon` still separate keep from not_our_kind. B: zero band."""
    with_refs = [r for r in labels if r["refs"]]

    def rate(label: str) -> tuple[int, int]:
        rows = [r for r in with_refs if r["label"] == label]
        hits = sum(1 for r in rows if any(x in canon for x in r["refs"]))
        return hits, len(rows)

    keep_hits, keep_n = rate("keep")
    not_hits, not_n = rate("drop_not_our_kind")

    keep_rate = keep_hits / keep_n if keep_n else None
    not_rate = not_hits / not_n if not_n else None
    separation = (keep_rate - not_rate) if (keep_rate is not None and not_rate is not None) else None

    def affinity(refs: list[str]) -> float:
        return sum(weights.get(r, 0.0) for r in refs)

    return {
        "labelled_total": len(labels),
        "with_references": len(with_refs),
        "keep": {"cites_canon": keep_hits, "n": keep_n, "rate": round(keep_rate, 4) if keep_rate is not None else None},
        "not_our_kind": {"cites_canon": not_hits, "n": not_n, "rate": round(not_rate, 4) if not_rate is not None else None},
        "separation": round(separation, 4) if separation is not None else None,
        "fisher_p": fisher_exact(keep_hits, keep_n - keep_hits, not_hits, not_n - not_hits),
        "auc_affinity": auc(
            [affinity(r["refs"]) for r in with_refs if r["label"] == "keep"],
            [affinity(r["refs"]) for r in with_refs if r["label"] == "drop_not_our_kind"],
        ),
        # B
        "zero_band": sum(1 for r in with_refs if not any(x in canon for x in r["refs"])),
        "zero_band_keeps": sum(
            1
            for r in with_refs
            if r["label"] == "keep" and not any(x in canon for x in r["refs"])
        ),
    }


def _is_transport(entry: dict) -> bool:
    text = " ".join(
        str(entry.get(k) or "").lower() for k in ("topic", "subfield", "title", "venue")
    )
    return any(m in text for m in TRANSPORT_MARKERS)


def measure_c(ours_doc: dict, merged_doc: dict) -> dict:
    """C: the anchor lines across every issue that has one, before and after.

    Anchors are recomputed from the same code path the issues use, with the
    canon swapped underneath. Nothing is rewritten — the issues on disk stay as
    they are; this only asks what today's code would choose.
    """
    from pipeline import synthesis

    issues = sorted((paths.CONTENT / "issues").glob("*.json"))
    entries = {c["openalex_id"]: c for c in merged_doc["candidates"]}

    def anchors_with(doc: dict) -> dict[str, list[dict]]:
        canon = {
            c["openalex_id"]: c for c in doc["candidates"] if c.get("class") == "foundation"
        }
        original = synthesis._foundation_canon
        synthesis._foundation_canon = lambda: canon  # type: ignore[assignment]
        try:
            out = {}
            for path in issues:
                issue = json.loads(path.read_text(encoding="utf-8"))
                d = date.fromisoformat(issue["date"])
                items = [it for it in (store.load_item(k) for k in issue["items"]) if it]
                out[issue["date"]] = synthesis.canon_anchors(d, items)
            return out
        finally:
            synthesis._foundation_canon = original  # type: ignore[assignment]

    before = anchors_with(ours_doc)
    after = anchors_with(merged_doc)

    def tally(anchors: dict[str, list[dict]]) -> dict:
        flat = [a for rows in anchors.values() for a in rows]
        transport = sum(1 for a in flat if _is_transport(entries.get(a["openalex_id"], a)))
        return {
            "days_with_an_anchor": sum(1 for rows in anchors.values() if rows),
            "anchor_lines": len(flat),
            "transport": transport,
            "transport_share": round(transport / len(flat), 4) if flat else None,
            "titles": [a["title"][:70] for a in flat],
        }

    b, a = tally(before), tally(after)

    # The anchor tally alone is not enough to answer C. Eight issues produce two
    # anchor lines between them, so "the anchors did not change" can mean the
    # merge is harmless or that the archive is too small to show harm. What the
    # merge *would* do is visible in what it adds, so the composition of the
    # added entries is reported next to the anchors and read as the stronger of
    # the two signals.
    additions = [c for c in merged_doc["candidates"] if c.get("source") == "external"]
    ours_foundation = [
        c for c in ours_doc["candidates"] if c.get("class") == "foundation"
    ]
    composition = {
        "added": len(additions),
        "added_transport": sum(1 for c in additions if _is_transport(c)),
        "our_foundation": len(ours_foundation),
        "our_foundation_transport": sum(1 for c in ours_foundation if _is_transport(c)),
        "added_topics": Counter(
            str(c.get("topic") or "unknown") for c in additions
        ).most_common(8),
    }
    composition["added_transport_share"] = (
        round(composition["added_transport"] / composition["added"], 4)
        if composition["added"]
        else None
    )
    composition["our_transport_share"] = (
        round(composition["our_foundation_transport"] / composition["our_foundation"], 4)
        if composition["our_foundation"]
        else None
    )
    changed = {
        d: {"before": [x["title"][:60] for x in before[d]], "after": [x["title"][:60] for x in after[d]]}
        for d in before
        if [x["openalex_id"] for x in before[d]] != [x["openalex_id"] for x in after[d]]
    }
    return {
        "before": b,
        "after": a,
        "share_delta": (
            round(a["transport_share"] - b["transport_share"], 4)
            if a["transport_share"] is not None and b["transport_share"] is not None
            else None
        ),
        "days_changed": changed,
        "composition": composition,
        "anchor_lines_available": b["anchor_lines"],
    }


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="Resolve external ids via OpenAlex")
    ap.add_argument("--measure", action="store_true", help="Run A, B and C")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.fetch:
        fetch_external()
        if not args.measure:
            return

    if not RESOLVED.exists():
        print("run --fetch first")
        return

    ours_doc = json.loads(OURS.read_text(encoding="utf-8"))
    merged_doc = build_merged()
    labels = _labelled_with_refs()

    ours_ids, merged_ids = _canon_ids(ours_doc), _canon_ids(merged_doc)
    before = measure_a_and_b(labels, ours_ids, _weights(ours_doc))
    after = measure_a_and_b(labels, merged_ids, _weights(merged_doc))
    c = measure_c(ours_doc, merged_doc)

    # What the merge actually touched, item by item. "The zero band did not move"
    # is a stronger statement when it sits next to "and here is how many items
    # gained a citation at all".
    added_ids = merged_ids - ours_ids
    touched = [
        r for r in labels if r["refs"] and any(x in added_ids for x in r["refs"])
    ]
    reach = {
        "canon_entries_added": len(added_ids),
        "labelled_items_citing_something_new": len(touched),
        "of_those_previously_in_the_zero_band": sum(
            1 for r in touched if not any(x in ours_ids for x in r["refs"])
        ),
        "labels_of_touched": dict(Counter(r["label"] for r in touched)),
    }

    sep_b, sep_a = before["separation"], after["separation"]
    a_pass = sep_b is None or sep_a is None or sep_a >= sep_b - 1e-9
    b_pass = after["zero_band"] < before["zero_band"]

    # C on the anchors alone has no power here — five issues yield two anchor
    # lines, and "unchanged" over n=2 is not evidence of anything. The
    # composition of what would be added does have power, so C passes only if
    # the anchors do not tilt AND the additions are not more transport-heavy
    # than the list they join. Reading "no observed change" as a pass on n=2
    # would be the measured-zero-versus-could-not-measure confusion again.
    comp = c["composition"]
    anchors_ok = (c["share_delta"] or 0.0) <= 1e-9
    composition_ok = (
        comp["added_transport_share"] is not None
        and comp["our_transport_share"] is not None
        and comp["added_transport_share"] <= comp["our_transport_share"] + 1e-9
    )
    c_pass = bool(anchors_ok and composition_ok)
    adopt = bool(a_pass and b_pass and c_pass)

    result = {
        "merged": {k: v for k, v in merged_doc.items() if k != "candidates"},
        "A_separation": {"before": before, "after": after, "pass": a_pass},
        "B_zero_band": {
            "before": before["zero_band"],
            "after": after["zero_band"],
            "keeps_in_zero_band": {
                "before": before["zero_band_keeps"],
                "after": after["zero_band_keeps"],
            },
            "reach": reach,
            "pass": b_pass,
        },
        "C_anchors": {
            **c,
            "anchors_ok": anchors_ok,
            "composition_ok": composition_ok,
            "pass": c_pass,
        },
        "adopt": adopt,
        "rule": "adopt only if A and B and C all pass; A is the veto",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    print(f"\nmerged: {merged_doc['ours']} ours + {merged_doc['external_added']} external "
          f"({merged_doc['overlap']} already shared, "
          f"{merged_doc['external_out_of_scope']} out of scope) = {merged_doc['total']}")
    print(f"\nA  separation keep - not_our_kind: {sep_b} → {sep_a}   {'PASS' if a_pass else 'FAIL'}")
    print(f"     keep {before['keep']['rate']} → {after['keep']['rate']}   "
          f"not_our_kind {before['not_our_kind']['rate']} → {after['not_our_kind']['rate']}")
    print(f"     AUC(affinity) {before['auc_affinity']} → {after['auc_affinity']}")
    print(f"B  zero band: {before['zero_band']} → {after['zero_band']}   {'PASS' if b_pass else 'FAIL'}")
    print(f"     {reach['canon_entries_added']} entries added reached "
          f"{reach['labelled_items_citing_something_new']} labelled items, "
          f"{reach['of_those_previously_in_the_zero_band']} of them in the zero band")
    print(f"C  transport share of anchors: {c['before']['transport_share']} → "
          f"{c['after']['transport_share']}  (n={c['anchor_lines_available']} lines — "
          f"{'ok' if anchors_ok else 'tilted'})")
    print(f"     composition: additions {comp['added_transport_share']} transport vs "
          f"ours {comp['our_transport_share']}   {'PASS' if c_pass else 'FAIL'}")
    print(f"     added topics: {comp['added_topics'][:4]}")
    print(f"\nADOPT: {adopt}")
    print(f"→ {REPORT}")

    if adopt:
        MERGED.write_text(
            json.dumps(merged_doc, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"→ {MERGED}  (point citation.canon_file at it to switch)")


if __name__ == "__main__":
    main()
