"""Generate vocab/sources/journals.yaml from OpenAlex (PRD §5.4).

The idea (YJUN): do not hand-define what counts as urban research. Take the
field's own settled answer — what the Urban Studies / Geography-Planning /
Transportation journals publish — and use that as the training set. It is more
defensible than a hand-picked seed list and, more importantly, reproducible.

**Deviation from PRD §5.4's literal query.** The PRD specifies

    GET /sources?filter=topics.subfield.id:3322,type:journal&sort=works_count:desc

but ``topics.subfield.id`` is not a valid Sources filter in the current API, and
the two filters that do exist (``topics.id``, ``topic_share.id``) match a source
if *any* of its works touch the topic. Sorted by ``works_count`` that returns
17,000+ sources led by a Polish economics journal and a radiology journal — a
useless whitelist. So we rank instead by **how many works a source actually
published in the subfield recently**: group Works by
``primary_location.source.id``, then keep the sources whose type is ``journal``.
Same intent, working query.

Output carries `# REVIEW:` markers. YJUN passes over it once (about 20 minutes)
and flips `include:` to false on anything that does not belong.

Usage:
    uv run python scripts/build_journal_whitelist.py [--since 2023-01-01] [--top 200]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.collectors.abstracts import abstract_source_for_publisher  # noqa: E402
from pipeline.collectors.openalex import configure_pyalex  # noqa: E402
from pipeline.config import cfg  # noqa: E402
from pipeline.paths import VOCAB  # noqa: E402

# Auto-inclusion needs BOTH signals, because each fails alone:
#   - concentration alone lets in small national generalist journals whose short
#     topic list happens to be planning-heavy (Magyar Tudomány scores 0.60)
#   - subfield_works alone lets in megajournals (PLoS ONE publishes more urban
#     papers in absolute terms than most urban journals, at ~0% concentration)
# Where the two disagree the row is marked REVIEW rather than silently decided.
CONCENTRATION_THRESHOLD = 0.25
MIN_SUBFIELD_WORKS = 20
# Minimum share of a source's subfield output that is in English. Set at 0.5 in
# phase 0g: `language: en` filters works and not sources, so a journal that
# publishes mostly in another language qualified on its English minority. Half
# is the point at which "this is an English-language journal" stops being true.
ENGLISH_SHARE_MIN = 0.5


def title_script_signal(name: str) -> str:
    """`non_latin`, `diacritic`, or `ascii` — a second, independent language read.

    `english_share` alone does not work, and measuring showed why: OpenAlex tags
    the works of `Środowisko Mieszkaniowe` (Polish) and `Культура и искусство`
    (Russian) as English, so both score 1.00 and are indistinguishable from
    Nature. No threshold can separate them.

    A title in a non-Latin script is decisive and excludes. Latin-with-diacritics
    is not: `Archaeology in Oceania/Archæology & physical anthropology` is an
    English journal with a ligature, so those are flagged for review rather than
    dropped. Neither signal is sufficient alone, and neither is
    `english_share` — which still catches `Cadernos Metrópole` at 0.35 and
    `Aetas` at 0.46, both plain-ASCII titles the script test would miss.
    """
    import unicodedata

    scripts = set()
    for ch in name or "":
        if ch.isalpha():
            try:
                scripts.add(unicodedata.name(ch).split()[0])
            except ValueError:
                pass
    if scripts - {"LATIN"}:
        return "non_latin"
    return "diacritic" if any(ord(c) > 127 for c in name or "") else "ascii"

SUBFIELD_NAMES = {
    "3322": "Urban Studies",
    "3305": "Geography, Planning and Development",
    "3313": "Transportation",
    "2215": "Building and Construction",
    "3312": "Sociology and Political Science",
    "1904": "Earth-Surface Processes",
}

# The hand list. PRD §5.4 says to use it as a cross-check against the automatic
# ranking; here it also acts as an override, because the two failure modes of the
# automatic signals both hit well-known urban journals (young ones score low on
# counts, interdisciplinary ones score low on concentration).
#
# REVIEW: this list is YJUN's to own. It encodes "these are urban research
# journals regardless of what the metrics say".
MANUAL_CANDIDATES = [
    # young / emerging
    "npj Urban Sustainability",
    "Urban Informatics",
    "Computational Urban Science",
    "Journal of Urban Mobility",
    "Environment and Planning B Urban Analytics and City Science",
    # core urban studies and planning
    "Cities",
    "Urban Studies",
    "Journal of the American Planning Association",
    "Landscape and Urban Planning",
    "Habitat International",
    "Urban Geography",
    "International Journal of Urban and Regional Research",
    "Journal of Urban Affairs",
    "Housing Studies",
    "Urban Forestry & Urban Greening",
    "Sustainable Cities and Society",
    "Computers Environment and Urban Systems",
    "Landscape and Urban Planning",
    "Cities & Health",
    # transport
    "Journal of Transport Geography",
    "Transport Reviews",
    "Transportation Research Part A Policy and Practice",
    "Transportation Research Part C Emerging Technologies",
    "Transportation Research Part D Transport and Environment",
    "Travel Behaviour and Society",
    "Transportation",
    "Journal of Transport & Health",
    "Transport Policy",
    # geography / GIScience with an urban centre of gravity
    "International Journal of Geographical Information Science",
    "Annals of the American Association of Geographers",
    "Applied Geography",
    "Geographical Analysis",
]


def group_sources_for_subfield_any_language(
    pyalex, subfield: str, since: str
) -> tuple[dict[str, int], float]:
    """The same grouping without the language filter — the denominator.

    `language: en` filters *works*, not sources, so a journal publishing mostly
    in Hungarian or Polish qualifies on its English minority. Measured in phase
    0f: at least 11 of 166 included entries. The 200-result cap had been hiding
    this by accident, and removing the cap exposed it.

    Dividing the English count by this gives `english_share`, which is a property
    of the source rather than of the works that happened to match.
    """
    q = (
        pyalex.Works()
        .filter(
            **{
                "primary_topic.subfield.id": subfield,
                "from_publication_date": since,
                "type": "article",
            }
        )
        .group_by("primary_location.source.id")
    )
    out: dict[str, int] = {}
    cost = 0.0
    for page in q.paginate(per_page=200, n_max=None):
        cost += float((getattr(page, "meta", {}) or {}).get("cost_usd") or 0.0)
        for g in page:
            key = (g.get("key") or "").rsplit("/", 1)[-1]
            if key and key.startswith("S"):
                out[key] = int(g.get("count") or 0)
    return out, cost


def group_sources_for_subfield(pyalex, subfield: str, since: str) -> dict[str, int]:
    """source_id → works published in this subfield since ``since``."""
    # `language: en` is not a convenience filter. Without it the ranking fills
    # with Hungarian and Russian generalist journals that OpenAlex classifies
    # into subfield 3305 — Magyar Nyelvőr, a linguistics journal, outranks most
    # real urban journals. They would enter the positive training set as
    # non-English humanities prose. The classifier embeds English text
    # (bge-base-en-v1.5) and the product is English-only, so restricting the
    # training set to English works is the honest scope, not a shortcut.
    q = (
        pyalex.Works()
        .filter(
            **{
                "primary_topic.subfield.id": subfield,
                "from_publication_date": since,
                "type": "article",
                "language": "en",
            }
        )
        .group_by("primary_location.source.id")
    )
    # Paginated. A single `get(per_page=200)` returns the first 200 groups and
    # silently stops, so any journal ranked below 200th *within a subfield* was
    # invisible to this builder — which is how Environment and Planning B:
    # Planning and Design never entered the candidate pool at all, and how
    # Environment and Planning A and the Journal of Urban Design entered it too
    # far down to survive the row cap below. The subfields return roughly 1,600
    # groups each, so the first page was about an eighth of the field.
    out: dict[str, int] = {}
    cost = 0.0
    for page in q.paginate(per_page=200, n_max=None):
        cost += float((getattr(page, "meta", {}) or {}).get("cost_usd") or 0.0)
        for g in page:
            key = (g.get("key") or "").rsplit("/", 1)[-1]
            if key and key.startswith("S"):
                out[key] = int(g.get("count") or 0)
    return out, cost


def fetch_sources(pyalex, ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    cost = 0.0
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        res = pyalex.Sources().filter(openalex="|".join(chunk)).get(per_page=50)
        cost += float((getattr(res, "meta", {}) or {}).get("cost_usd") or 0.0)
        for s in res:
            out[(s.get("id") or "").rsplit("/", 1)[-1]] = dict(s)
    return out, cost


def search_source_by_name(pyalex, name: str) -> dict | None:
    try:
        res = pyalex.Sources().search(name).get(per_page=3)
    except Exception:
        return None
    for s in res:
        if (s.get("display_name") or "").lower().strip() == name.lower().strip():
            return dict(s)
    return None


def subfield_concentration(source: dict, subfields: set[str]) -> float:
    """Share of a journal's listed topic output that falls in the target subfields.

    Ranking by subfield output alone lets megajournals in — PLoS ONE and
    Scientific Reports publish more urban papers in absolute terms than most
    urban journals, while being ~0% urban. Concentration separates "a journal
    about cities" from "a journal that occasionally mentions cities", which is
    the distinction the training set actually depends on.
    """
    topics = source.get("topics") or []
    total = sum(int(t.get("count") or 0) for t in topics)
    if not total:
        return 0.0
    hit = sum(
        int(t.get("count") or 0)
        for t in topics
        if ((t.get("subfield") or {}).get("id") or "").rsplit("/", 1)[-1] in subfields
    )
    return round(hit / total, 4)


_CONTROL = {c: None for c in range(0x20)} | {c: None for c in range(0x7F, 0xA0)}


def _yaml_str(value) -> str:
    """Quote a scalar, stripping control characters.

    Some OpenAlex display names carry stray C1 control bytes, which YAML refuses
    to read back. Dropping them is safer than escaping: nothing downstream keys
    off a journal name.
    """
    if value is None:
        return "null"
    s = str(value).translate(_CONTROL).replace('"', "'").replace("\\", "/")
    return f'"{s.strip()}"'


def build(since: str, top: int, out_path: Path) -> int:
    pyalex = configure_pyalex()
    subfields = [str(s) for s in (cfg("openalex.whitelist_subfields", ["3322"]) or [])]

    per_subfield: dict[str, dict[str, int]] = {}
    any_language: dict[str, int] = defaultdict(int)
    total_cost = 0.0
    for sf in subfields:
        counts, cost = group_sources_for_subfield(pyalex, sf, since)
        total_cost += cost
        per_subfield[sf] = counts
        all_counts, cost = group_sources_for_subfield_any_language(pyalex, sf, since)
        total_cost += cost
        for sid, n in all_counts.items():
            any_language[sid] += n
        print(f"subfield {sf} ({SUBFIELD_NAMES.get(sf, '?')}): {len(counts)} candidate sources "
              f"({len(all_counts)} in any language)")

    combined: dict[str, int] = defaultdict(int)
    membership: dict[str, list[str]] = defaultdict(list)
    for sf, counts in per_subfield.items():
        for sid, n in counts.items():
            combined[sid] += n
            membership[sid].append(sf)

    ranked = sorted(combined.items(), key=lambda kv: -kv[1])[: top * 2]
    sources, cost = fetch_sources(pyalex, [sid for sid, _ in ranked])
    total_cost += cost

    sf_set = set(subfields)
    manual_lookup = {n.lower().strip() for n in MANUAL_CANDIDATES}
    rows = []
    for sid, n in ranked:
        s = sources.get(sid)
        if not s or s.get("type") != "journal":
            continue
        conc = subfield_concentration(s, sf_set)
        name = s.get("display_name") or sid
        on_hand_list = name.lower().strip() in manual_lookup
        # English share of this source's subfield output. A source we cannot
        # read is not a source we can summarise, and the classifier embeds
        # English text — so this is scope, not convenience.
        denominator = any_language.get(sid, 0)
        english_share = round(n / denominator, 4) if denominator else None
        script = title_script_signal(name)
        rows.append(
            {
                "id": sid,
                "name": name,
                "issn_l": s.get("issn_l"),
                "publisher": (s.get("host_organization_name") or None),
                "works_count": int(s.get("works_count") or 0),
                "subfield_works": n,
                "subfields": sorted(membership[sid]),
                "concentration": conc,
                "english_share": english_share,
                "title_script": script,
                "include": on_hand_list
                or (
                    conc >= CONCENTRATION_THRESHOLD
                    and n >= MIN_SUBFIELD_WORKS
                    and (english_share is None or english_share >= ENGLISH_SHARE_MIN)
                    and script != "non_latin"
                ),
                "manual": on_hand_list,
            }
        )
        # `top` caps the list, not the *examination*. A source that fails the
        # concentration test still belongs in the file marked `include: false`,
        # because the review is over what was considered as much as over what
        # was kept — and breaking here meant sources past the cap were never
        # even written down as rejected.
        if sum(1 for r in rows if r["include"]) >= top:
            break

    have = {r["name"].lower().strip() for r in rows}
    for name in MANUAL_CANDIDATES:
        if name.lower().strip() in have:
            continue
        s = search_source_by_name(pyalex, name)
        if not s:
            print(f"  manual candidate not found: {name}")
            continue
        rows.append(
            {
                "id": (s.get("id") or "").rsplit("/", 1)[-1],
                "name": s.get("display_name"),
                "issn_l": s.get("issn_l"),
                "publisher": s.get("host_organization_name"),
                "works_count": int(s.get("works_count") or 0),
                "subfield_works": 0,
                "subfields": [],
                "concentration": subfield_concentration(s, sf_set),
                "include": True,
                "manual": True,
            }
        )

    included = sum(1 for r in rows if r["include"])
    lines = [
        "# Journal whitelist for the relevance classifier training set (PRD §5.4).",
        "#",
        "# GENERATED by scripts/build_journal_whitelist.py — do not hand-sort, but DO",
        "# hand-review. Ranking is by works published in the target subfields since",
        f"# {since}, not by total works_count (see the script docstring for why).",
        "#",
        "# `concentration` is the share of the journal's topic output that sits in the",
        "# target subfields. `include` was set automatically from it: >= "
        f"{CONCENTRATION_THRESHOLD} is in.",
        "# That is a heuristic, not a decision — it exists so the human pass is about",
        "# judgement rather than deleting megajournals.",
        "#",
        "# REVIEW: YJUN passes over this list once. Expect roughly 20 minutes.",
        "# REVIEW:   1. entries with `manual: true` — young journals a count ranking misses",
        "# REVIEW:   2. `include: false` entries with concentration near the threshold",
        "# REVIEW:   3. anything included that is not urban research",
        f"# REVIEW: currently {included} of {len(rows)} journals are included.",
        "#",
        f"generated_at: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        f"generated_since: {since}",
        "subfields:",
    ]
    for sf in subfields:
        lines.append(f"  - id: \"{sf}\"")
        lines.append(f"    name: {_yaml_str(SUBFIELD_NAMES.get(sf))}")
    lines.append("")
    lines.append("sources:")
    for r in rows:
        lines.append(f"  - id: \"{r['id']}\"")
        lines.append(f"    name: {_yaml_str(r['name'])}")
        lines.append(f"    issn_l: {_yaml_str(r['issn_l'])}")
        lines.append(f"    publisher: {_yaml_str(r['publisher'])}")
        # Which route reaches this publisher's abstracts. A routing field, not
        # an exclusion rule — see scripts/annotate_journal_abstract_source.py.
        lines.append(f"    abstract_source: \"{abstract_source_for_publisher(r['publisher'])}\"")
        lines.append(f"    works_count: {r['works_count']}")
        lines.append(f"    subfield_works: {r['subfield_works']}")
        lines.append(f"    subfields: [{', '.join(chr(34) + s + chr(34) for s in r['subfields'])}]")
        lines.append(f"    concentration: {r['concentration']}")
        if r.get("english_share") is not None:
            lines.append(f"    english_share: {r['english_share']}")
        if r.get("title_script") and r["title_script"] != "ascii":
            lines.append(f"    title_script: \"{r['title_script']}\"  # REVIEW: language")
        if r["manual"]:
            lines.append("    manual: true")
        signals_disagree = (r["concentration"] >= CONCENTRATION_THRESHOLD) != (
            r["subfield_works"] >= MIN_SUBFIELD_WORKS
        )
        marker = "REVIEW" if (r["manual"] or signals_disagree) else "auto"
        lines.append(f"    include: {'true' if r['include'] else 'false'}  # {marker}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(f"wrote {len(rows)} journals to {out_path}")
    print(f"openalex cost for this build: ${total_cost:.4f}")
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="2023-01-01")
    p.add_argument("--top", type=int, default=int(cfg("openalex.whitelist_max_sources", 200)))
    p.add_argument("--out", default=str(VOCAB / "sources" / "journals.yaml"))
    args = p.parse_args()
    build(args.since, args.top, Path(args.out))


if __name__ == "__main__":
    main()
