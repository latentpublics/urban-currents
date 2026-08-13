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


# --------------------------------------------------------------------------
# R0 — canon scope: the corpus decides, not the polling list
# --------------------------------------------------------------------------


def _work(topic_id: str, subfield_id: str, source_id: str = "S999") -> dict:
    return {
        "id": "https://openalex.org/W1",
        "display_name": "A Work",
        "primary_topic": {
            "id": f"https://openalex.org/{topic_id}",
            "display_name": topic_id,
            "subfield": {"id": f"https://openalex.org/{subfield_id}", "display_name": subfield_id},
        },
        "primary_location": {"source": {"id": f"https://openalex.org/{source_id}"}},
    }


def test_the_venue_rule_dropped_work_our_own_corpus_had_cited(repo):
    """Ewing & Handy in the Journal of Urban Design: cited three times by our
    papers and excluded for appearing in a journal we do not poll daily."""
    from pipeline.graph.canon import _in_scope

    off_whitelist = _work("T10000", "3322", source_id="S_NOT_POLLED")
    assert _in_scope(off_whitelist, {"S_POLLED"}, "venue") is False
    assert _in_scope(off_whitelist, {"S_POLLED"}, "subfield") is True


def test_a_generic_instrument_is_excluded_by_topic(repo):
    """Hu & Bentler sits in the same subfield as Moran's I, so the exclusion has
    to be finer than subfield or it takes the wrong one."""
    from pipeline.graph.canon import _in_scope

    psychometrics = _work("T10467", "1803")
    morans_i = _work("T11798", "1803")

    assert _in_scope(psychometrics, set(), "subfield") is False
    assert _in_scope(morans_i, set(), "subfield") is True, "Moran's I must survive"


def test_scope_modes_are_independent(repo):
    from pipeline.graph.canon import _in_scope

    polled_but_generic = _work("T10467", "1803", source_id="S_POLLED")
    assert _in_scope(polled_but_generic, {"S_POLLED"}, "venue") is True
    assert _in_scope(polled_but_generic, {"S_POLLED"}, "subfield") is False
    assert _in_scope(polled_but_generic, {"S_POLLED"}, "both") is False


def test_every_exclusion_names_the_work_it_removes():
    """An exclusion without evidence is a guess with a config entry."""
    from pipeline.config import vocab_file

    doc = vocab_file("canon_exclude_subfields.yaml")
    entries = (doc.get("topics") or []) + (doc.get("subfields") or [])
    assert entries, "the exclusion list must not be empty"
    for entry in entries:
        assert entry.get("id"), entry
        assert entry.get("removes"), f"{entry['id']} names no work it removes"
        assert entry.get("why"), f"{entry['id']} gives no reason"


# --------------------------------------------------------------------------
# R3 — daily accumulation stays inside its budget and its queue drains
# --------------------------------------------------------------------------


def test_the_pending_queue_is_ordered_by_how_much_we_cite(repo, monkeypatch):
    """Resolving in arbitrary order spends days on the tail before reaching the
    head. 139,540 distinct references, of which 23,362 are cited more than once."""
    from pipeline.graph import daily_canon

    for i in range(3):
        store.save_item(_item(f"doi:{i}", ["openalex:W_hot", f"openalex:W_cold{i}"]), today=DAY)

    asked: list[list[str]] = []

    def fake_resolve(ids, batch=50):
        asked.append(list(ids))
        return [], 0.0

    monkeypatch.setattr(daily_canon, "_resolve", fake_resolve)
    daily_canon.accumulate_day(DAY, max_ids=2)

    assert asked and asked[0][0] == "openalex:W_hot", "the most-cited goes first"


def test_running_the_same_day_twice_leaves_no_duplicate_records(repo, monkeypatch):
    from pipeline.graph import daily_canon
    from pipeline.graph.citation import load_reference_base

    store.save_item(_item("doi:a", ["openalex:W1", "openalex:W2"]), today=DAY)
    monkeypatch.setattr(daily_canon, "_resolve", lambda ids, batch=50: ([], 0.0))

    daily_canon.accumulate_day(DAY)
    daily_canon.accumulate_day(DAY)

    keys = [r["work_key"] for r in load_reference_base()]
    assert len(keys) == len(set(keys))


def test_the_queue_shrinks_as_ids_resolve(repo, monkeypatch):
    from pipeline.graph import daily_canon

    store.save_item(_item("doi:a", [f"openalex:W{i}" for i in range(6)]), today=DAY)

    def resolve_two(ids, batch=50):
        return [{"openalex_id": i, "title": i} for i in ids[:2]], 0.0

    monkeypatch.setattr(daily_canon, "_resolve", resolve_two)
    first = daily_canon.accumulate_day(DAY, max_ids=2)
    second = daily_canon.accumulate_day(DAY, max_ids=2)

    assert first["pending_after"] == 4
    assert second["pending_after"] == 2, "the queue must actually drain"


# --------------------------------------------------------------------------
# S0 — foundation and instrument are two lists, not a filter
# --------------------------------------------------------------------------


def test_a_listed_instrument_topic_beats_a_low_ratio(repo):
    """Difference-in-differences sits at ratio 327, below Tobler's 346. Only the
    topic can tell them apart."""
    from pipeline.graph.canon import classify_candidate, instrument_topics

    topics, _ = instrument_topics()
    an_instrument_topic = sorted(topics)[0]

    cls, basis = classify_candidate(an_instrument_topic, 5.0)
    assert (cls, basis) == ("instrument", "topic")


def test_the_ratio_still_catches_topics_nobody_listed(repo):
    from pipeline.graph.canon import classify_candidate, instrument_topics

    _, threshold = instrument_topics()
    assert classify_candidate("T_UNLISTED", threshold + 1) == ("instrument", "ratio")
    assert classify_candidate("T_UNLISTED", threshold - 1)[0] == "foundation"


def test_holling_and_arnstein_stay_foundations(repo):
    """Both have high ratios and both are foundations. A rule that demotes them
    is wrong, whatever else it gets right."""
    from pipeline.graph.canon import classify_candidate

    assert classify_candidate("T10202", 672.0)[0] == "foundation"  # Holling
    assert classify_candidate("T10704", 569.0)[0] == "foundation"  # Arnstein


def test_spatial_statistics_is_not_an_instrument(repo):
    """LISA, Getis-Ord and Moran's I are method papers and still foundations of
    this field — spatial statistics is its own apparatus, not borrowed kit."""
    from pipeline.config import vocab_file

    doc = vocab_file("canon_instrument_topics.yaml")
    names = {t["name"] for t in doc["topics"]}
    assert "Spatial and Panel Data Analysis" not in names


def test_every_instrument_topic_names_a_work_it_classifies():
    from pipeline.config import vocab_file

    doc = vocab_file("canon_instrument_topics.yaml")
    assert doc["topics"]
    for entry in doc["topics"]:
        assert entry.get("id") and entry.get("classifies") and entry.get("why"), entry


# --------------------------------------------------------------------------
# The JSONL stores survive a title that contains a line separator
# --------------------------------------------------------------------------


def test_a_line_separator_in_a_title_does_not_split_the_record():
    """U+2028 is legal inside a JSON string and `splitlines()` breaks on it, so
    one paper title silently corrupted the resolved store."""
    import json as _json

    from pipeline.store import jsonl_line

    row = {"openalex_id": "openalex:W1", "title": "Before" + chr(0x2028) + "After"}
    line = jsonl_line(row)

    assert len(line.splitlines()) == 1
    assert _json.loads(line) == row


def test_one_bad_line_does_not_make_the_whole_store_unreadable(repo, tmp_path):
    from pipeline.store import read_jsonl

    p = tmp_path / "store.jsonl"
    p.write_text('{"a": 1}\n{"a": broken\n{"a": 3}\n', encoding="utf-8")

    errors: list = []
    rows = read_jsonl(p, on_error=errors)

    assert [r["a"] for r in rows] == [1, 3]
    assert len(errors) == 1
