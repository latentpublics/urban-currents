"""OpenAlex-native entities, taken as given (PRD §2, §4.1).

Topics, authors and institutions come straight from OpenAlex with **no LLM
involvement**. Rebuilding any of these ourselves would cost months and be worse:
topic hierarchy, author disambiguation, institution ROR normalisation.

Field names are not renamed on the way in — that is a Phase 1 interface promise
(PRD §12).
"""

from __future__ import annotations

from typing import Any, Optional

from ..collectors.base import normalize_openalex_id, normalize_ror
from ..models import Author, EntityRef, Graph, Institution, TopicRef


def topics_from_work(work: dict[str, Any]) -> list[TopicRef]:
    out: list[TopicRef] = []
    primary = work.get("primary_topic") or {}
    primary_id = normalize_openalex_id((primary or {}).get("id"))
    for t in work.get("topics") or []:
        tid = normalize_openalex_id(t.get("id"))
        if not tid:
            continue
        subfield = ((t.get("subfield") or {}).get("id") or "")
        out.append(
            TopicRef(
                id=f"openalex:{tid}",
                label=t.get("display_name") or tid,
                subfield=normalize_openalex_id(subfield) if subfield else None,
                score=round(float(t.get("score") or 0.0), 4),
                is_primary=(tid == primary_id),
            )
        )
    return out


def people_from_work(work: dict[str, Any]) -> list[EntityRef]:
    """ORCID-keyed where possible; OpenAlex author ID otherwise. Authors without
    either are omitted rather than given a made-up ID."""
    out: list[EntityRef] = []
    seen: set[str] = set()
    for a in work.get("authorships") or []:
        author = a.get("author") or {}
        orcid = author.get("orcid")
        oa = normalize_openalex_id(author.get("id"))
        if orcid:
            eid = f"orcid:{orcid.rsplit('/', 1)[-1]}"
        elif oa:
            eid = f"openalex:{oa}"
        else:
            continue
        if eid in seen:
            continue
        seen.add(eid)
        out.append(EntityRef(id=eid, label=author.get("display_name") or eid))
    return out


def orgs_from_work(work: dict[str, Any]) -> list[EntityRef]:
    out: list[EntityRef] = []
    seen: set[str] = set()
    for a in work.get("authorships") or []:
        for inst in a.get("institutions") or []:
            ror = normalize_ror(inst.get("ror"))
            oa = normalize_openalex_id(inst.get("id"))
            if ror:
                eid = f"ror:{ror}"
            elif oa:
                eid = f"openalex:{oa}"
            else:
                continue
            if eid in seen:
                continue
            seen.add(eid)
            out.append(EntityRef(id=eid, label=inst.get("display_name") or eid))
    return out


def graph_from_work(work: dict[str, Any]) -> Graph:
    return Graph(
        referenced_works=[
            f"openalex:{normalize_openalex_id(w)}"
            for w in (work.get("referenced_works") or [])
            if normalize_openalex_id(w)
        ],
        related_works=[
            f"openalex:{normalize_openalex_id(w)}"
            for w in (work.get("related_works") or [])
            if normalize_openalex_id(w)
        ],
        cited_by_count=int(work.get("cited_by_count") or 0),
    )


def authors_from_work(work: dict[str, Any]) -> list[Author]:
    out: list[Author] = []
    for a in work.get("authorships") or []:
        author = a.get("author") or {}
        out.append(
            Author(
                name=author.get("display_name") or "",
                orcid=author.get("orcid"),
                openalex=normalize_openalex_id(author.get("id")),
                institutions=[
                    Institution(ror=normalize_ror(i.get("ror")), name=i.get("display_name"))
                    for i in (a.get("institutions") or [])
                ],
            )
        )
    return [a for a in out if a.name]


def apply_passthrough(item, work: Optional[dict[str, Any]]) -> int:
    """Fill an Item's OpenAlex-native entities. Returns the topic count added."""
    if not work:
        return 0
    oa_id = normalize_openalex_id(work.get("id"))
    if oa_id:
        item.ids.openalex = oa_id
    topics = topics_from_work(work)
    if topics:
        item.entities.topics = topics
    people = people_from_work(work)
    if people:
        item.entities.people = people
    orgs = orgs_from_work(work)
    if orgs:
        item.entities.orgs = orgs
    g = graph_from_work(work)
    if g.referenced_works or g.related_works or g.cited_by_count:
        item.graph = g
    if "openalex" not in item.provenance.collectors:
        item.provenance.collectors = sorted(set(item.provenance.collectors) | {"openalex"})
    return len(topics)
