"""Derive ``content/graph/edges.jsonl`` from Items (PRD §3.5).

Edges are not a separate source of truth — keeping them in the Item schema as
well would create a two-source sync debt. This is a build output: regenerate it,
never hand-edit it.

``cites`` edges come free from OpenAlex ``referenced_works``. Phase 2's "related
papers" and the Monthly Map are queries over this file, not new pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .. import paths, store
from ..models import Edge, Item

FACET_EDGE_TYPE = {
    "topics": "has_topic",
    "methods": "uses_method",
    "data": "uses_data",
    "tools": "uses_tool",
    "people": "authored_by",
    "orgs": "affiliated_with",
    "places": "studies_place",
}


def edges_for_item(item: Item) -> Iterator[Edge]:
    d = item.first_published or item.updated
    for facet, etype in FACET_EDGE_TYPE.items():
        for ref in getattr(item.entities, facet):
            yield Edge(src=item.work_key, dst=ref.id, type=etype, date=d)  # type: ignore[arg-type]
    for w in item.graph.referenced_works:
        yield Edge(src=item.work_key, dst=w, type="cites", date=d)
    for w in item.graph.related_works:
        yield Edge(src=item.work_key, dst=w, type="related_to", date=d)


def build_edges(items: Iterable[Item] | None = None, out: Path | None = None) -> int:
    items = list(items) if items is not None else list(store.iter_items())
    rows: list[str] = []
    for item in sorted(items, key=lambda i: i.work_key):
        for edge in edges_for_item(item):
            rows.append(
                json.dumps(
                    edge.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
                )
            )
    # Sorted and de-duplicated so the file is byte-stable across runs.
    rows = sorted(set(rows))
    target = out or (paths.GRAPH / "edges.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    store.write_text_atomic(target, "\n".join(rows) + ("\n" if rows else ""))
    return len(rows)
