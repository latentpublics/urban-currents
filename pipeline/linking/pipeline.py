"""Entity linking orchestration (PRD §5, M4).

Three sources, in order of trust:

1. **OpenAlex passthrough** — topics / people / orgs, verbatim, no LLM.
2. **LLM overlay** — methods / data / tools candidates, produced as a by-product
   of the summarize call, then matched against controlled vocabulary. Nothing
   unmatched reaches ``entities``; it goes to ``unmatched.jsonl`` instead.
3. **Places** — best-effort, and an empty result is a normal outcome.

Also maintains ``content/entities/`` nodes and their counts.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional, Sequence

from .. import store
from ..metrics import Run
from ..models import Entity, Item
from .places import link_places
from .vocab_match import Vocabulary, match_facet, scan_text


def _overlay_stash(run: Run) -> dict[str, dict[str, list[str]]]:
    from ..summarize.run import load_overlay_stash

    return load_overlay_stash(run)


def link_items(
    items: Sequence[Item],
    run: Run,
    use_llm: bool = True,
    resolve_places_online: bool = False,
) -> dict[str, Any]:
    """Link in place. ``use_llm`` only controls whether overlay candidates are
    consumed — the OpenAlex passthrough runs regardless."""
    stash = _overlay_stash(run)
    vocabs = {f: Vocabulary.load(f) for f in ("methods", "data", "tools")}

    topics_from_openalex = 0
    rule_matched = 0
    unmatched_counts = {"methods": 0, "data": 0, "tools": 0, "places": 0}

    for item in items:
        topics_from_openalex += len(item.entities.topics)

        cands = stash.get(item.work_key) if use_llm else None
        if not cands:
            # No LLM candidates (backfill, missing key, or a summarize failure):
            # fall back to scanning the abstract for known vocabulary. Lower
            # recall, zero cost, and it keeps the novelty score meaningful.
            text = f"{item.bibliography.title} {item.bibliography.abstract or ''}"
            for facet in ("methods", "data", "tools"):
                refs = scan_text(text, facet, vocabs[facet])
                if refs:
                    setattr(item.entities, facet, refs)
            rule_matched += 1
            continue

        for facet in ("methods", "data", "tools"):
            result = match_facet(cands.get(facet, []), facet, vocabs[facet])
            setattr(item.entities, facet, result.refs)
            unmatched_counts[facet] += len(result.unmatched)
            for raw in result.unmatched:
                run.append_jsonl(
                    "unmatched.jsonl",
                    {"work_key": item.work_key, "facet": facet, "candidate": raw},
                )

        place_refs, status, place_unmatched = link_places(
            cands.get("places", []), resolve_online=resolve_places_online
        )
        item.entities.places = place_refs
        item.entities.places_status = status
        unmatched_counts["places"] += len(place_unmatched)
        for raw in place_unmatched:
            run.append_jsonl(
                "unmatched.jsonl",
                {"work_key": item.work_key, "facet": "places", "candidate": raw},
            )

    return {
        "status": "OK",
        "topics_from_openalex": topics_from_openalex,
        "rule_matched_items": rule_matched,
        "unmatched_methods": unmatched_counts["methods"],
        "unmatched_data": unmatched_counts["data"],
        "unmatched_tools": unmatched_counts["tools"],
        "unmatched_places": unmatched_counts["places"],
    }


FACETS = ("topics", "methods", "data", "tools", "people", "orgs", "places")


def rebuild_entity_nodes(today: Optional[date] = None) -> int:
    """Regenerate ``content/entities/`` from every Item. Counts are derived, so
    they cannot drift out of sync with the archive."""
    nodes: dict[tuple[str, str], Entity] = {}
    for item in store.iter_items():
        seen_date = item.first_published or item.updated
        for facet in FACETS:
            for ref in getattr(item.entities, facet):
                key = (facet, ref.id)
                node = nodes.get(key)
                if node is None:
                    node = Entity(
                        id=ref.id,
                        facet=facet,  # type: ignore[arg-type]
                        label=ref.label,
                        item_count=0,
                        first_seen=seen_date,
                        last_seen=seen_date,
                    )
                    if ref.id.startswith("openalex:"):
                        node.canonical.openalex = ref.id.split(":", 1)[1]
                    elif ref.id.startswith("wikidata:"):
                        node.canonical.wikidata = ref.id.split(":", 1)[1]
                    nodes[key] = node
                node.item_count += 1
                if seen_date:
                    if node.first_seen is None or seen_date < node.first_seen:
                        node.first_seen = seen_date
                    if node.last_seen is None or seen_date > node.last_seen:
                        node.last_seen = seen_date

    _attach_method_parents(nodes)
    for node in nodes.values():
        store.save_entity(node)
    return len(nodes)


def _attach_method_parents(nodes: dict[tuple[str, str], Entity]) -> None:
    v = Vocabulary.load("methods")
    parents = {e.id: e.parent for e in v._by_surface.values() if e.parent}
    for (facet, eid), node in nodes.items():
        if facet == "methods" and eid in parents:
            node.parent = parents[eid]
