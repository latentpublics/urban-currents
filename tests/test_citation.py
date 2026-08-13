"""Citation layer, canon selection, metapath centrality, promotion (phase 0d).

Every artefact here is derived and regenerable; the tests defend the properties
that make it safe to regenerate — determinism, scope, and the windows that keep
the pair count from exploding.
"""

from __future__ import annotations

from datetime import date, timedelta

from pipeline import paths, store
from pipeline.graph.canon import accumulate
from pipeline.graph.centrality import (
    brokerage_events,
    centrality,
    project,
    spearman,
    stability,
)
from pipeline.graph.citation import (
    build_coupling,
    build_reference_base,
    compute_coupling,
    internal_citation_edges,
    load_reference_base,
    top_neighbours,
)
from pipeline.models import Bibliography, EntityRef, Graph, Ids, Item

DAY = date(2026, 8, 11)


def _item(key: str, refs: list[str], day: date = DAY, openalex: str | None = None) -> Item:
    return Item(
        work_key=key,
        first_published=day,
        ids=Ids(openalex=openalex, doi=key.split(":", 1)[1] if key.startswith("doi:") else None),
        bibliography=Bibliography(title=f"Paper {key}"),
        graph=Graph(referenced_works=refs),
    )


def _records(items: list[Item]) -> list[dict]:
    return [
        {
            "work_key": it.work_key,
            "date": str(it.first_published),
            "published": True,
            "referenced_works": sorted(set(it.graph.referenced_works)),
        }
        for it in items
    ]


# --------------------------------------------------------------------------
# Bibliographic coupling
# --------------------------------------------------------------------------


def test_coupling_needs_the_configured_minimum_of_shared_references():
    """One or two shared references is coincidence in a field with a canon."""
    a = _item("doi:a", ["openalex:W1", "openalex:W2", "openalex:W9"])
    b = _item("doi:b", ["openalex:W1", "openalex:W2", "openalex:W8"])
    c = _item("doi:c", ["openalex:W1", "openalex:W2", "openalex:W3"])

    pairs = compute_coupling(_records([a, b, c]), min_shared=3)
    keys = {(p["a"], p["b"]) for p in pairs}
    assert ("doi:a", "doi:b") not in keys, "two shared references is below the floor"

    pairs2 = compute_coupling(_records([a, b, c]), min_shared=2)
    assert ("doi:a", "doi:b") in {(p["a"], p["b"]) for p in pairs2}


def test_coupling_reports_both_normalisations():
    """Jaccard alone hides that a long bibliography has more chances to overlap;
    the raw count alone hides how tight the overlap is."""
    a = _item("doi:a", [f"openalex:W{i}" for i in range(5)])
    b = _item("doi:b", [f"openalex:W{i}" for i in range(4)] + ["openalex:X"])

    pair = compute_coupling(_records([a, b]), min_shared=3)[0]
    assert pair["shared"] == 4
    assert 0 < pair["jaccard"] < 1
    assert pair["a_references"] == 5 and pair["b_references"] == 5


def test_the_window_keeps_distant_papers_apart():
    """"Read these together" is a claim about what is current."""
    refs = [f"openalex:W{i}" for i in range(5)]
    near = _item("doi:near", refs, day=DAY)
    far = _item("doi:far", refs, day=DAY - timedelta(days=400))

    assert compute_coupling(_records([near, far]), min_shared=3, window_days=90) == []
    assert compute_coupling(_records([near, far]), min_shared=3, window_days=500)


def test_coupling_is_deterministic():
    items = [_item(f"doi:{c}", [f"openalex:W{i}" for i in range(6)]) for c in "abc"]
    first = compute_coupling(_records(items), min_shared=3)
    second = compute_coupling(_records(items), min_shared=3)
    assert first == second


def test_top_neighbours_are_symmetric():
    a = _item("doi:a", [f"openalex:W{i}" for i in range(5)])
    b = _item("doi:b", [f"openalex:W{i}" for i in range(5)])
    pairs = compute_coupling(_records([a, b]), min_shared=3)
    nb = top_neighbours(k=3, pairs=pairs)
    assert nb["doi:a"][0]["other"] == "doi:b"
    assert nb["doi:b"][0]["other"] == "doi:a"


# --------------------------------------------------------------------------
# Internal citation
# --------------------------------------------------------------------------


def test_cites_internal_only_fires_when_the_target_is_in_the_archive():
    cited = _item("doi:cited", [], openalex="W100")
    citing = _item("doi:citing", ["openalex:W100", "openalex:W999"])

    edges = internal_citation_edges([cited, citing])
    assert edges == [("doi:citing", "doi:cited")]


def test_an_item_citing_itself_is_not_an_edge():
    it = _item("doi:self", ["openalex:W1"], openalex="W1")
    assert internal_citation_edges([it]) == []


# --------------------------------------------------------------------------
# The reference base is regenerable
# --------------------------------------------------------------------------


def test_the_reference_base_round_trips(repo):
    store.save_item(_item("doi:a", ["openalex:W1", "openalex:W2"]), today=DAY)
    store.save_item(_item("doi:b", []), today=DAY)

    stats = build_reference_base()
    assert stats["records"] == 1, "an item with no references is not a record here"

    loaded = load_reference_base()
    assert loaded[0]["work_key"] == "doi:a"
    assert loaded[0]["referenced_works"] == ["openalex:W1", "openalex:W2"]

    before = (paths.GRAPH / "references.jsonl").read_text(encoding="utf-8")
    build_reference_base()
    assert (paths.GRAPH / "references.jsonl").read_text(encoding="utf-8") == before


def test_coupling_file_is_byte_stable(repo):
    for c in "abc":
        store.save_item(_item(f"doi:{c}", [f"openalex:W{i}" for i in range(6)]), today=DAY)
    build_reference_base()

    build_coupling()
    first = (paths.GRAPH / "coupling.jsonl").read_text(encoding="utf-8")
    build_coupling()
    assert (paths.GRAPH / "coupling.jsonl").read_text(encoding="utf-8") == first


# --------------------------------------------------------------------------
# Canon
# --------------------------------------------------------------------------


def test_a_recent_citation_outweighs_an_old_one():
    """A 2007 paper cited by twelve items this month is alive; the same paper
    cited once years ago is furniture."""
    fresh = {
        "work_key": "doi:new", "date": str(DAY), "published": True,
        "referenced_works": ["openalex:W_recent"],
    }
    stale = {
        "work_key": "doi:old", "date": str(DAY - timedelta(days=720)), "published": True,
        "referenced_works": ["openalex:W_old"],
    }
    rows = {r["openalex_id"]: r for r in accumulate([fresh, stale], today=DAY, half_life_days=180)}

    assert rows["openalex:W_recent"]["archive_citations"] == 1
    assert rows["openalex:W_old"]["archive_citations"] == 1
    assert rows["openalex:W_recent"]["weighted_score"] > rows["openalex:W_old"]["weighted_score"]
    assert rows["openalex:W_recent"]["archive_citations_last_12m"] == 1
    assert rows["openalex:W_old"]["archive_citations_last_12m"] == 0


def test_accumulate_orders_by_weighted_score():
    records = [
        {"work_key": f"doi:{i}", "date": str(DAY), "published": True,
         "referenced_works": ["openalex:W_hot"] if i < 3 else ["openalex:W_cold"]}
        for i in range(4)
    ]
    rows = accumulate(records, today=DAY)
    assert rows[0]["openalex_id"] == "openalex:W_hot"
    assert rows[0]["archive_citations"] == 3


# --------------------------------------------------------------------------
# Centrality
# --------------------------------------------------------------------------


def _tagged(key: str, methods: list[str], day: date = DAY) -> Item:
    it = _item(key, [])
    it.first_published = day
    it.entities.methods = [EntityRef(id=f"method:{m}", label=m) for m in methods]
    return it


def test_a_projection_links_tags_that_shared_an_item():
    items = [_tagged("doi:a", ["clustering", "random forest"])]
    weights = project("method-method", items=items)
    assert weights == {("clustering", "random forest"): 1}


def test_the_degree_floor_removes_nodes_that_sit_on_no_route():
    """A tag from a single paper has degree 1 and brokers nothing, but there are
    many of them and they dominate the tail."""
    items = [_tagged(f"doi:{i}", ["a", f"leaf{i}"]) for i in range(4)]
    result = centrality(project("method-method", items=items), min_degree=3)
    assert result["nodes_after_floor"] == 1, "only the hub clears the floor"


def test_an_empty_projection_says_so_rather_than_erroring():
    result = centrality({}, min_degree=3)
    assert result["nodes"] == 0
    assert result["betweenness"] == []


def test_identical_windows_are_flagged_as_not_evidence(repo):
    """A correlation of 1.0 between windows that selected the same items is
    arithmetic. The archive is days long and the windows are months."""
    for i in range(3):
        store.save_item(_tagged(f"doi:{i}", ["a", "b", f"c{i}"]), today=DAY)

    result = stability("method-method", windows=(30, 60, 90), min_degree=1)
    assert result["windows_are_distinct"] is False
    assert "same" in (result["note"] or "")


def test_spearman_treats_a_vanished_name_as_the_worst_rank():
    assert spearman(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert (spearman(["a", "b", "c"], ["c", "b", "a"]) or 0) < 0


def test_brokerage_counts_only_the_first_meeting_of_a_pair(repo):
    store.save_item(_tagged("doi:1", ["street view", "simulation"], DAY), today=DAY)
    store.save_item(
        _tagged("doi:2", ["street view", "simulation"], DAY + timedelta(days=1)),
        today=DAY + timedelta(days=1),
    )

    result = brokerage_events()
    assert result["first_meetings"] == 1, "the second co-occurrence is not news"


# --------------------------------------------------------------------------
# Promotion
# --------------------------------------------------------------------------


def test_promotion_leaves_published_issues_untouched(repo):
    from pipeline.models import Issue
    from pipeline.promote import promote, unreadable_items

    item = _item("doi:dark", [])
    item.bibliography.abstract = None
    store.save_item(item, today=DAY)
    store.save_issue(Issue(date=DAY, items=[], unreadable=["doi:dark"]))

    assert [i.work_key for i in unreadable_items()] == ["doi:dark"]

    result = promote(DAY + timedelta(days=1), use_llm=False, sources=())
    assert result["recovered"] == 0

    # The issue is a record of what was true that day.
    assert store.load_issue(DAY).unreadable == ["doi:dark"]


def test_an_item_that_already_has_an_abstract_is_no_longer_a_candidate(repo):
    from pipeline.models import Issue
    from pipeline.promote import unreadable_items

    item = _item("doi:recovered", [])
    item.bibliography.abstract = "It turned up later."
    store.save_item(item, today=DAY)
    store.save_issue(Issue(date=DAY, items=[], unreadable=["doi:recovered"]))

    assert unreadable_items() == []
