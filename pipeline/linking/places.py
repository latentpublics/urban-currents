"""Place resolution — best-effort, de-prioritised (PRD §2, v1.1).

Places is no longer a signature axis. Candidates fall out of the summarize call
for free, so we keep collecting them, but a failed lookup logs and moves on.
``places_status`` stays in the schema so that reviving this axis later does not
require reprocessing the whole archive.

Wikidata lookup is off by default in Phase 0 (``resolve_online=False``): the
local alias file plus a small built-in table covers the common cases without
adding a third external dependency to the critical path.
"""

from __future__ import annotations

from typing import Iterable, Optional

import httpx

from ..config import vocab_file
from ..models import PlaceRef, PlacesStatus

# A deliberately small table of cities that appear constantly in this literature.
# Anything outside it either resolves through vocab/places_aliases.yaml or stays
# unresolved — inventing a Wikidata QID would be worse than an empty field.
_BUILTIN = {
    "new york": ("wikidata:Q60", "New York City"),
    "new york city": ("wikidata:Q60", "New York City"),
    "nyc": ("wikidata:Q60", "New York City"),
    "london": ("wikidata:Q84", "London"),
    "paris": ("wikidata:Q90", "Paris"),
    "berlin": ("wikidata:Q64", "Berlin"),
    "tokyo": ("wikidata:Q1490", "Tokyo"),
    "seoul": ("wikidata:Q8684", "Seoul"),
    "beijing": ("wikidata:Q956", "Beijing"),
    "shanghai": ("wikidata:Q8686", "Shanghai"),
    "shenzhen": ("wikidata:Q15174", "Shenzhen"),
    "singapore": ("wikidata:Q334", "Singapore"),
    "hong kong": ("wikidata:Q8646", "Hong Kong"),
    "chicago": ("wikidata:Q1297", "Chicago"),
    "los angeles": ("wikidata:Q65", "Los Angeles"),
    "san francisco": ("wikidata:Q62", "San Francisco"),
    "boston": ("wikidata:Q100", "Boston"),
    "washington dc": ("wikidata:Q61", "Washington, D.C."),
    "toronto": ("wikidata:Q172", "Toronto"),
    "vancouver": ("wikidata:Q24639", "Vancouver"),
    "mexico city": ("wikidata:Q1489", "Mexico City"),
    "sao paulo": ("wikidata:Q174", "São Paulo"),
    "são paulo": ("wikidata:Q174", "São Paulo"),
    "bogota": ("wikidata:Q2841", "Bogotá"),
    "santiago": ("wikidata:Q2887", "Santiago"),
    "madrid": ("wikidata:Q2807", "Madrid"),
    "barcelona": ("wikidata:Q1492", "Barcelona"),
    "amsterdam": ("wikidata:Q727", "Amsterdam"),
    "rotterdam": ("wikidata:Q34370", "Rotterdam"),
    "copenhagen": ("wikidata:Q1748", "Copenhagen"),
    "stockholm": ("wikidata:Q1754", "Stockholm"),
    "zurich": ("wikidata:Q72", "Zurich"),
    "milan": ("wikidata:Q490", "Milan"),
    "rome": ("wikidata:Q220", "Rome"),
    "istanbul": ("wikidata:Q406", "Istanbul"),
    "moscow": ("wikidata:Q649", "Moscow"),
    "delhi": ("wikidata:Q1353", "Delhi"),
    "mumbai": ("wikidata:Q1156", "Mumbai"),
    "bangalore": ("wikidata:Q1355", "Bangalore"),
    "jakarta": ("wikidata:Q3630", "Jakarta"),
    "manila": ("wikidata:Q1461", "Manila"),
    "bangkok": ("wikidata:Q1861", "Bangkok"),
    "sydney": ("wikidata:Q3130", "Sydney"),
    "melbourne": ("wikidata:Q3141", "Melbourne"),
    "nairobi": ("wikidata:Q3870", "Nairobi"),
    "lagos": ("wikidata:Q8673", "Lagos"),
    "cairo": ("wikidata:Q85", "Cairo"),
    "johannesburg": ("wikidata:Q34647", "Johannesburg"),
    "united states": ("wikidata:Q30", "United States"),
    "china": ("wikidata:Q148", "China"),
    "netherlands": ("wikidata:Q55", "Netherlands"),
    "united kingdom": ("wikidata:Q145", "United Kingdom"),
}

WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"


def _aliases() -> dict[str, tuple[str, str]]:
    doc = vocab_file("places_aliases.yaml") or {}
    out: dict[str, tuple[str, str]] = {}
    for key, val in (doc.get("aliases") or {}).items():
        if isinstance(val, dict) and val.get("id"):
            out[key.strip().lower()] = (val["id"], val.get("label") or key)
    return out


def resolve_place(name: str, resolve_online: bool = False) -> Optional[tuple[str, str]]:
    key = " ".join(name.strip().lower().split())
    if not key:
        return None
    hit = _aliases().get(key) or _BUILTIN.get(key)
    if hit:
        return hit
    if not resolve_online:
        return None
    try:  # pragma: no cover - network path, best-effort by design
        r = httpx.get(
            WIKIDATA_SEARCH,
            params={
                "action": "wbsearchentities",
                "search": name,
                "language": "en",
                "format": "json",
                "type": "item",
                "limit": 1,
            },
            timeout=10.0,
        )
        r.raise_for_status()
        results = r.json().get("search") or []
        if results:
            return f"wikidata:{results[0]['id']}", results[0].get("label") or name
    except Exception:
        return None
    return None


def link_places(
    candidates: Iterable[str], resolve_online: bool = False
) -> tuple[list[PlaceRef], PlacesStatus, list[str]]:
    cands = [c for c in candidates if c and c.strip()]
    if not cands:
        return [], "unspecified", []
    refs: list[PlaceRef] = []
    unmatched: list[str] = []
    seen: set[str] = set()
    for c in cands:
        hit = resolve_place(c, resolve_online=resolve_online)
        if hit is None:
            unmatched.append(c)
            continue
        pid, label = hit
        if pid in seen:
            continue
        seen.add(pid)
        refs.append(PlaceRef(id=pid, label=label, role="study_area", confidence=0.9))
    status: PlacesStatus = "resolved" if refs else "unspecified"
    return refs, status, unmatched
