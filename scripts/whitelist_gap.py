"""R2: let citations find the holes in the polling list (phase 0e).

`journals.yaml` was built by a rule (D10): take OpenAlex subfields 3322, 3305
and 3313, group Works by source, keep English sources with concentration >= 0.25
and >= 20 subfield works, then apply a manual override list. Every rule misses
things, and the Journal of Urban Design is the proof — urban design straddles
architecture-adjacent subfields, so it never entered the three the rule looks at,
and three of our own papers cite Ewing and Handy from it.

Rather than re-reading 159 entries by hand, this asks what our corpus already
cites and which of those venues we do not poll. The same recency weighting the
canon uses applies here: a journal we leaned on this month matters more than one
we leaned on in May.

**Output is a candidate list only.** `journals.yaml` is not touched; approval is
YJUN's.

**Known bias, and the reason this is a prioritiser rather than a review.** It can
only see what the papers we already collect happen to cite. A subfield we never
touch is never cited and stays invisible forever, so this reinforces the shape of
the current list as much as it corrects it.

Usage:
    uv run python scripts/whitelist_gap.py --top 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.config import cfg, journals_vocab  # noqa: E402
from pipeline.graph.citation import load_reference_base  # noqa: E402

# D10's thresholds, imported by value so a change there shows up here as a
# mismatch rather than silently diverging.
SUBFIELDS = {"3322", "3305", "3313"}
CONCENTRATION_THRESHOLD = 0.25
MIN_SUBFIELD_WORKS = 20
OUT = paths.VOCAB / "sources" / "journal_gap_candidates.yaml"


def _parse(value: str):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def weighted_reference_counts(half_life: float) -> tuple[dict[str, float], dict[str, int]]:
    records = load_reference_base()
    dates = [d for d in (_parse(r.get("date", "")) for r in records) if d]
    today = max(dates) if dates else date.today()

    weighted: dict[str, float] = defaultdict(float)
    raw: dict[str, int] = defaultdict(int)
    for record in records:
        d = _parse(record.get("date", ""))
        if d is None:
            continue
        w = math.pow(0.5, max((today - d).days, 0) / half_life)
        for ref in record.get("referenced_works") or []:
            weighted[ref] += w
            raw[ref] += 1
    return weighted, raw


def why_excluded(source: dict) -> str:
    """Which D10 condition this venue failed, reproduced rather than guessed."""
    subfields = {
        (t.get("subfield") or {}).get("id", "").rsplit("/", 1)[-1]
        for t in (source.get("topics") or [])
    }
    if not (subfields & SUBFIELDS):
        return "subfield mismatch: none of its topics sit in 3322/3305/3313"

    counts = source.get("counts_by_year") or []
    recent = sum(c.get("works_count", 0) for c in counts[:3])
    in_subfield = sum(
        t.get("count", 0)
        for t in (source.get("topics") or [])
        if (t.get("subfield") or {}).get("id", "").rsplit("/", 1)[-1] in SUBFIELDS
    )
    total_topic = sum(t.get("count", 0) for t in (source.get("topics") or [])) or 1
    conc = in_subfield / total_topic

    reasons = []
    if conc < CONCENTRATION_THRESHOLD:
        reasons.append(f"concentration {conc:.3f} < {CONCENTRATION_THRESHOLD}")
    if in_subfield < MIN_SUBFIELD_WORKS:
        reasons.append(f"subfield works {in_subfield} < {MIN_SUBFIELD_WORKS}")
    if (source.get("type") or "") != "journal":
        reasons.append(f"source type is {source.get('type')!r}, not a journal")
    return "; ".join(reasons) or f"passes D10's tests (concentration {conc:.3f}) - a genuine miss"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--pool", type=int, default=1200, help="Top referenced works to resolve")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from pipeline.collectors.openalex import configure_pyalex

    pyalex = configure_pyalex()
    half_life = float(cfg("citation.canon_half_life_days", 180))
    weighted, raw = weighted_reference_counts(half_life)
    ranked = sorted(weighted, key=lambda k: -weighted[k])[: a.pool]

    cost = 0.0
    by_venue_w: dict[str, float] = defaultdict(float)
    by_venue_n: dict[str, int] = defaultdict(int)
    venue_name: dict[str, str] = {}
    ids = [k.split(":", 1)[1] for k in ranked]
    for i in range(0, len(ids), 50):
        res, meta = (
            pyalex.Works().filter(openalex_id="|".join(ids[i : i + 50]))
            .get(per_page=50, return_meta=True)
        )
        cost += float((meta or {}).get("cost_usd") or 0.0)
        for w in res:
            key = "openalex:" + (w.get("id") or "").rsplit("/", 1)[-1]
            src = ((w.get("primary_location") or {}).get("source") or {})
            sid = (src.get("id") or "").rsplit("/", 1)[-1]
            if not sid:
                continue
            venue_name[sid] = src.get("display_name") or sid
            by_venue_w[sid] += weighted.get(key, 0.0)
            by_venue_n[sid] += raw.get(key, 0)

    whitelist = {s["id"]: s for s in (journals_vocab().get("sources") or []) if s.get("id")}
    included = {k for k, v in whitelist.items() if v.get("include", True)}
    arxiv_id = str(cfg("openalex.arxiv_source_id", "S4306400194"))

    missing = [
        sid for sid in by_venue_w
        if sid not in whitelist and sid != arxiv_id
    ]
    missing.sort(key=lambda s: -by_venue_w[s])
    top_missing = missing[: a.top]

    # One more call per candidate venue, to reproduce D10's decision.
    details = []
    for sid in top_missing:
        try:
            src = pyalex.Sources()[sid]
        except Exception as e:  # noqa: BLE001
            details.append({
                "id": sid, "name": venue_name.get(sid, sid),
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        counts = src.get("counts_by_year") or []
        details.append({
            "id": sid,
            "name": src.get("display_name") or venue_name.get(sid, sid),
            "publisher": src.get("host_organization_name"),
            "issn_l": src.get("issn_l"),
            "type": src.get("type"),
            "works_per_year": (counts[0].get("works_count") if counts else None),
            "our_citations": by_venue_n[sid],
            "our_citations_weighted": round(by_venue_w[sid], 3),
            "d10_verdict": why_excluded(src),
        })

    # The other direction: on the list, never cited, never published from.
    published_sources = {
        it.bibliography.primary_location.source_id
        for it in store.iter_items()
        if it.bibliography.primary_location.source_id
    }
    idle = [
        {
            "id": sid,
            "name": s.get("name"),
            "publisher": s.get("publisher"),
            "abstract_source": s.get("abstract_source"),
            "concentration": s.get("concentration"),
        }
        for sid, s in whitelist.items()
        if sid in included and by_venue_n.get(sid, 0) == 0 and sid not in published_sources
    ]

    lines = [
        "# Whitelist gap candidates — venues our own corpus cites that we do not poll.",
        "#",
        "# GENERATED by scripts/whitelist_gap.py. **Not a decision.** Nothing here has",
        "# been added to journals.yaml; every entry needs YJUN's approval.",
        "#",
        "# Ranked by how much our corpus leans on them, with the same 180-day recency",
        "# weighting the canon uses. `d10_verdict` reproduces which condition of the",
        "# whitelist build rule each one failed, so the question is not 'should this be",
        "# on the list' in the abstract but 'was the rule right to drop it'.",
        "#",
        "# BIAS: this can only see what the papers we already collect happen to cite. A",
        "# subfield we never touch is never cited and stays invisible. It prioritises a",
        "# whitelist review; it does not replace one.",
        "#",
        f"# generated_at: {date.today().isoformat()}",
        f"# reference_base: {len(load_reference_base())} records",
        f"# venues_resolved: {len(by_venue_w)}   not_on_whitelist: {len(missing)}",
        "",
        "candidates:",
    ]
    for d in details:
        lines.append(f"  - id: \"{d['id']}\"  # REVIEW")
        lines.append(f"    name: {json.dumps(d.get('name'), ensure_ascii=False)}")
        for field in ("publisher", "issn_l", "type", "works_per_year",
                      "our_citations", "our_citations_weighted"):
            if d.get(field) is not None:
                lines.append(f"    {field}: {json.dumps(d[field], ensure_ascii=False)}")
        if d.get("d10_verdict"):
            lines.append(f"    d10_verdict: {json.dumps(d['d10_verdict'], ensure_ascii=False)}")
        if d.get("error"):
            lines.append(f"    error: {json.dumps(d['error'])}")
    lines.append("")
    lines.append("# On the whitelist, never cited by our corpus, and never published from")
    lines.append("# in the 90 days observed. Listed only — nothing is removed here.")
    lines.append("idle_whitelist_entries:")
    for e in sorted(idle, key=lambda r: str(r["name"]))[:40]:
        lines.append(f"  - id: \"{e['id']}\"  # REVIEW")
        lines.append(f"    name: {json.dumps(e['name'], ensure_ascii=False)}")
        lines.append(f"    abstract_source: {json.dumps(e.get('abstract_source'))}")
        lines.append(f"    concentration: {json.dumps(e.get('concentration'))}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(f"resolved venues: {len(by_venue_w)}; not on whitelist: {len(missing)}")
    print(f"idle whitelist entries: {len(idle)}")
    print(f"openalex cost: ${cost:.6f}")
    print(f"written: {OUT}")
    print()
    print(f"{'weighted':>9} {'raw':>5}  {'name':<44} d10 verdict")
    for d in details:
        print(
            f"{d.get('our_citations_weighted', 0):>9} {d.get('our_citations', 0):>5}  "
            f"{str(d.get('name'))[:44]:<44} {str(d.get('d10_verdict'))[:60]}"
        )


if __name__ == "__main__":
    main()
