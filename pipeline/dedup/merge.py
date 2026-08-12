"""Duplicate merge (PRD §5.2).

The same paper arrives twice: once as an arXiv preprint, once as an OpenAlex
journal Work. OpenAlex is itself inconsistent — sometimes the preprint is a
separate Work, sometimes it is folded into ``locations[]`` of the published
Work — so both shapes have to be handled.

Merge keys, in priority order:

1. DOI match, after reducing ``10.48550/arxiv.*`` back to an arXiv ID
2. arXiv location match — an arXiv ``landing_page_url`` inside the Work's locations
3. Normalised title + first-author surname, ``token_sort_ratio >= 95``

The branch that matters most: **if a merge result matches an Item that was
already published, this is not a new publication.** The existing Item's
``publication_status`` is updated and the Issue records a ``status_changes``
line. Missing that branch means the same paper headlines twice, four months
apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from rapidfuzz.fuzz import token_sort_ratio

from ..collectors.base import last_name, normalize_arxiv_id, normalize_doi, normalize_title
from ..models import Cluster, Item, MergeBasis, StatusChange

TITLE_MATCH_THRESHOLD = 95


def merge_keys(item: Item) -> set[str]:
    """Identity keys for rules 1 and 2. An arXiv DOI reduces to the arXiv key."""
    keys: set[str] = set()
    arxiv = item.ids.arxiv or normalize_arxiv_id(item.ids.doi)
    if not arxiv:
        arxiv = normalize_arxiv_id(item.bibliography.primary_location.landing_page_url)
    if arxiv:
        keys.add(f"arxiv:{arxiv}")
    doi = normalize_doi(item.ids.doi)
    if doi and not normalize_arxiv_id(doi):
        keys.add(f"doi:{doi}")
    if item.ids.openalex:
        keys.add(f"openalex:{item.ids.openalex}")
    return keys


def _first_author_surname(item: Item) -> str:
    if not item.bibliography.authors:
        return ""
    return last_name(item.bibliography.authors[0].name)


def fuzzy_same(a: Item, b: Item) -> bool:
    """Rule 3. Requires both a high title ratio and a matching first surname."""
    ta, tb = normalize_title(a.bibliography.title), normalize_title(b.bibliography.title)
    if not ta or not tb:
        return False
    if token_sort_ratio(ta, tb) < TITLE_MATCH_THRESHOLD:
        return False
    sa, sb = _first_author_surname(a), _first_author_surname(b)
    return bool(sa) and sa == sb


def _work_key_rank(item: Item) -> tuple[int, str]:
    """arXiv ID → DOI → OpenAlex ID (PRD §5.2). Lower rank wins."""
    scheme = item.work_key.split(":", 1)[0]
    order = {"arxiv": 0, "doi": 1, "openalex": 2}
    return (order.get(scheme, 3), item.work_key)


def _merge_pair(base: Item, other: Item) -> Item:
    """Fold ``other`` into ``base``. ``base.work_key`` is authoritative."""
    for f in ("openalex", "doi", "arxiv", "pmid"):
        if getattr(base.ids, f) is None:
            setattr(base.ids, f, getattr(other.ids, f))

    bib, obib = base.bibliography, other.bibliography
    if not bib.abstract and obib.abstract:
        bib.abstract = obib.abstract
    if not bib.authors and obib.authors:
        bib.authors = obib.authors
    elif obib.authors and len(obib.authors) == len(bib.authors):
        # OpenAlex authorships carry ORCID/ROR that arXiv does not.
        for a, o in zip(bib.authors, obib.authors):
            a.orcid = a.orcid or o.orcid
            a.openalex = a.openalex or o.openalex
            if not a.institutions:
                a.institutions = o.institutions
    if not bib.publication_date and obib.publication_date:
        bib.publication_date = obib.publication_date
    bib.categories = sorted(set(bib.categories) | set(obib.categories))
    # Keep the journal location if one side has it; arXiv stays reachable via ids.
    if other.publication_status.state == "published" and (
        bib.primary_location.type in (None, "repository")
    ):
        if obib.primary_location.source_id:
            bib.primary_location = obib.primary_location

    if other.publication_status.state == "published":
        base.publication_status = other.publication_status
    if other.graph.cited_by_count > base.graph.cited_by_count:
        base.graph.cited_by_count = other.graph.cited_by_count
    if not base.graph.referenced_works:
        base.graph.referenced_works = other.graph.referenced_works
    if not base.graph.related_works:
        base.graph.related_works = other.graph.related_works
    if not base.entities.topics:
        base.entities.topics = other.entities.topics
    if not base.entities.people:
        base.entities.people = other.entities.people
    if not base.entities.orgs:
        base.entities.orgs = other.entities.orgs

    base.provenance.collectors = sorted(
        set(base.provenance.collectors) | set(other.provenance.collectors)
    )
    return base


@dataclass
class MergeResult:
    items: list[Item]
    clusters: int
    merged_away: int


def merge_candidates(candidates: Iterable[Item], run_date: Optional[date] = None) -> MergeResult:
    """Cluster candidates and collapse each cluster into one Item."""
    cands = list(candidates)
    n = len(cands)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    # Rules 1 & 2: shared identity keys.
    seen: dict[str, int] = {}
    basis: dict[int, MergeBasis] = {}
    for i, it in enumerate(cands):
        for k in merge_keys(it):
            if k in seen:
                union(seen[k], i)
                basis[find(i)] = "arxiv_location" if k.startswith("arxiv:") else "doi_match"
            else:
                seen[k] = i

    # Rule 3: fuzzy title + surname, only between items not already clustered.
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if fuzzy_same(cands[i], cands[j]):
                union(i, j)
                basis.setdefault(find(i), "title_author_fuzzy")

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: list[Item] = []
    merged_away = 0
    for root, members in sorted(groups.items()):
        group = sorted((cands[i] for i in members), key=_work_key_rank)
        base = group[0]
        for other in group[1:]:
            base = _merge_pair(base, other)
            merged_away += 1
        member_keys = sorted({it.work_key for it in group})
        base.cluster = Cluster(
            cluster_id=(
                f"clu_{run_date or base.first_published or ''}_{base.work_key.split(':',1)[1][:16]}"
            ),
            members=member_keys,
            merge_basis=basis.get(root, "singleton") if len(group) > 1 else "singleton",
        )
        out.append(base)

    out.sort(key=lambda it: it.work_key)
    return MergeResult(items=out, clusters=len(out), merged_away=merged_away)


def reconcile_with_archive(
    items: Iterable[Item], load_existing
) -> tuple[list[Item], list[Item], list[StatusChange]]:
    """Split merged items into (new, updates) against already-published Items.

    ``load_existing(work_key) -> Item | None``. An item whose ``work_key`` is
    already on disk is **not** a new publication; if its state moved from
    preprint to published we emit a ``status_changes`` entry instead.
    """
    fresh: list[Item] = []
    updates: list[Item] = []
    changes: list[StatusChange] = []

    for it in items:
        existing = load_existing(it.work_key)
        if existing is None:
            fresh.append(it)
            continue
        before = existing.publication_status.state
        merged = _merge_pair(existing, it)
        after = merged.publication_status.state
        if before != after:
            changes.append(
                StatusChange(
                    work_key=merged.work_key,
                    **{"from": before},
                    to=after,
                    journal=merged.publication_status.journal,
                )
            )
        updates.append(merged)

    return fresh, updates, changes
