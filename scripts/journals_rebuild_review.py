"""U3: the data behind the journal whitelist rebuild review (phase 0h).

Its output is an analysis document and is **not committed here** (0W, G4b):
it belongs with the working record, and `.gitignore` keeps a re-run from
quietly putting it back into a public repository.


164 YAML entries are not a thing anyone can judge. This joins the rebuild
against the current whitelist, the 0g source metrics, and the citations our own
corpus makes, so the review sheet can be read rather than parsed.

`vocab/sources/journals.yaml` is not touched. Nothing here adopts anything.

Usage:
    uv run python scripts/journals_rebuild_review.py
    uv run python scripts/journals_rebuild_review_render.py   # writes the sheet
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import yaml  # noqa: E402

CONC = 0.25
MIN_WORKS = 20
ES_MIN = 0.5
WINDOW_DAYS = (2026 - 2023) * 365 + 224  # 2023-01-01 .. 2026-08-13

v2 = yaml.safe_load((ROOT / "vocab/sources/journals.rebuilt.v2.yaml").read_text(encoding="utf-8"))
cur = yaml.safe_load((ROOT / "vocab/sources/journals.yaml").read_text(encoding="utf-8"))
cur_ids = {s["id"] for s in cur["sources"] if s.get("include", True)}

jm = json.loads((ROOT / "runs/journal_metrics.json").read_text(encoding="utf-8"))
prestige = {s["id"]: s for s in jm["sources"]}
subfield_names = {sf["id"]: sf["name"] for sf in v2["subfields"]}

# Our corpus's citations, by venue. Only 16% of distinct references are
# resolved, so these are a floor, not a count.
resolved = {}
for line in (ROOT / "runs/state/canon_resolved.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        resolved[r["openalex_id"]] = r

mentions = Counter()
for line in (ROOT / "content/graph/references.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    for ref in json.loads(line).get("referenced_works") or []:
        mentions[ref] += 1

venue_mentions = Counter()
venue_works = Counter()
for wid, rec in resolved.items():
    vid = rec.get("venue_id")
    if not vid:
        continue
    n = mentions.get(wid, 0)
    if n:
        venue_mentions[vid] += n
        venue_works[vid] += 1

resolved_share = round(len(resolved) / len(mentions), 4) if mentions else 0.0


def why_excluded(s: dict) -> list[str]:
    reasons = []
    if s.get("concentration", 0) < CONC:
        reasons.append(f"집중도 {s.get('concentration')} < {CONC}")
    if s.get("subfield_works", 0) < MIN_WORKS:
        reasons.append(f"서브필드 논문 {s.get('subfield_works')} < {MIN_WORKS}")
    es = s.get("english_share")
    if es is not None and es < ES_MIN:
        reasons.append(f"english_share {es} < {ES_MIN}")
    if s.get("title_script") == "non_latin":
        reasons.append("제목이 비라틴 문자")
    return reasons


def norm_vid(vid: str) -> str:
    return (vid or "").rsplit("/", 1)[-1]


venue_mentions = Counter({norm_vid(k): v for k, v in venue_mentions.items()})
venue_works = Counter({norm_vid(k): v for k, v in venue_works.items()})

rows = []
for s in v2["sources"]:
    sid = s["id"]
    p = prestige.get(sid, {})
    rows.append({
        **s,
        "new": s.get("include", True) and sid not in cur_ids,
        # 0.5 is what `percentile()` returns for an empty population, not a
        # median-ranked journal. Carrying the population size is what lets the
        # sheet print "—" instead of a figure that reads as a measurement.
        "prestige_pct_in_subfield": p.get("prestige_pct_in_subfield"),
        "prestige_population": p.get("subfield_population"),
        "two_year_mean_citedness": p.get("two_year_mean_citedness"),
        "h_index": p.get("h_index"),
        "subfield_name": subfield_names.get((s.get("subfields") or [""])[0], "—"),
        "annual": round(s.get("subfield_works", 0) / (WINDOW_DAYS / 365), 1),
        "daily": round(s.get("subfield_works", 0) / WINDOW_DAYS, 3),
        "our_citations": venue_mentions.get(sid, 0),
        "our_cited_works": venue_works.get(sid, 0),
        "exclude_reasons": [] if s.get("include", True) else why_excluded(s),
    })

out = {
    "window_days": WINDOW_DAYS,
    "resolved_reference_share": resolved_share,
    "resolved_works": len(resolved),
    "distinct_references": len(mentions),
    "current_included": len(cur_ids),
    "v2_included": sum(1 for r in rows if r.get("include", True)),
    "v2_total": len(rows),
    "new_included": sum(1 for r in rows if r["new"]),
    "daily_now": round(sum(r["daily"] for r in rows if r["id"] in cur_ids), 2),
    "daily_v2": round(sum(r["daily"] for r in rows if r.get("include", True)), 2),
    "rows": rows,
}
(ROOT / "runs/journals_rebuild_review.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, ensure_ascii=False))
