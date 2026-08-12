"""Entity linking, vocabulary matching, and the zero-free-strings rule."""

from __future__ import annotations

from datetime import date

from pipeline import store
from pipeline.linking.pipeline import link_items, rebuild_entity_nodes
from pipeline.linking.places import link_places, resolve_place
from pipeline.linking.vocab_match import Vocabulary, match_facet, scan_text
from pipeline.metrics import Run
from pipeline.models import Bibliography, EntityRef, Item, TopicRef


def _fake_extractor(payload: dict):
    """An LLMClient whose extraction call always returns `payload`."""
    import json

    from pipeline.llm import LLMClient, LLMResponse

    def caller(system, user):
        return LLMResponse(text=json.dumps(payload), input_tokens=10, output_tokens=5,
                           model="gemini-3.5-flash")

    return LLMClient(task="extract", caller=caller)


def _item(work_key="arxiv:2608.01234", abstract="") -> Item:
    return Item(
        work_key=work_key,
        first_published=date(2026, 8, 11),
        bibliography=Bibliography(title="A Paper", abstract=abstract),
    )


# --------------------------------------------------------------------------
# Vocabulary matching
# --------------------------------------------------------------------------


def test_alias_matches_resolve_to_the_canonical_id(repo):
    v = Vocabulary.load("methods")
    entry, score = v.match("GNN")
    assert entry.id == "method:gnn"
    assert score == 1.0
    assert entry.parent == "method:deep-learning"


def test_fuzzy_match_absorbs_minor_variation(repo):
    v = Vocabulary.load("data")
    entry, score = v.match("street-view imagery")
    assert entry.id == "data:street-view"
    assert score >= 0.88


def test_unmatched_candidates_are_reported_not_invented(repo):
    result = match_facet(
        ["graph neural network", "quantum tea leaf reading"], "methods"
    )
    assert [r.id for r in result.refs] == ["method:gnn"]
    assert result.unmatched == ["quantum tea leaf reading"]


def test_scan_text_finds_vocabulary_without_an_llm(repo):
    text = (
        "We use a graph neural network over street view imagery and OSMnx road "
        "network extraction to estimate accessibility."
    )
    methods = scan_text(text, "methods")
    data = scan_text(text, "data")
    tools = scan_text(text, "tools")
    assert "method:gnn" in [r.id for r in methods]
    assert "data:street-view" in [r.id for r in data]
    assert "github:gboeing/osmnx" in [r.id for r in tools]


def test_scan_text_returns_nothing_for_unrelated_prose(repo):
    assert scan_text("A study of medieval poetry metre.", "methods") == []


# --------------------------------------------------------------------------
# Linking stage
# --------------------------------------------------------------------------


def test_llm_candidates_only_enter_entities_through_the_vocabulary(repo):
    """PRD §9: zero free strings in entities."""
    run = Run.for_date(date(2026, 8, 11))
    item = _item(abstract="An abstract long enough to be summarised.")
    client = _fake_extractor(
        {
            "methods": ["graph neural network", "an entirely made up technique"],
            "data": ["street view imagery"],
            "tools": ["OSMnx"],
            "places": ["Seoul", "Atlantis"],
        }
    )

    stats = link_items([item], run, use_llm=True, client=client)

    assert [r.id for r in item.entities.methods] == ["method:gnn"]
    assert [r.id for r in item.entities.data] == ["data:street-view"]
    assert [r.id for r in item.entities.tools] == ["github:gboeing/osmnx"]
    assert [r.id for r in item.entities.places] == ["wikidata:Q8684"]
    assert stats["unmatched_methods"] == 1
    assert stats["unmatched_places"] == 1

    # Candidates are normalised to lower case before matching, so the log holds
    # the normalised form.
    unmatched = (run.dir / "unmatched.jsonl").read_text(encoding="utf-8").lower()
    assert "an entirely made up technique" in unmatched
    assert "atlantis" in unmatched


def test_link_falls_back_to_text_scan_without_llm_candidates(repo):
    run = Run.for_date(date(2026, 8, 11))
    item = _item(abstract="We train a convolutional neural network on satellite imagery.")
    link_items([item], run, use_llm=False)
    assert "method:cnn" in [r.id for r in item.entities.methods]
    assert "data:satellite-imagery" in [r.id for r in item.entities.data]


def test_places_status_is_unspecified_when_nothing_resolves(repo):
    refs, status, unmatched = link_places(["Atlantis"])
    assert refs == []
    assert status == "unspecified"
    assert unmatched == ["Atlantis"]


def test_places_status_is_unspecified_when_no_candidates(repo):
    refs, status, unmatched = link_places([])
    assert (refs, status, unmatched) == ([], "unspecified", [])


def test_place_alias_table_is_used_before_any_network_call(repo):
    assert resolve_place("NYC") == ("wikidata:Q60", "New York City")
    assert resolve_place("nowhere at all", resolve_online=False) is None


# --------------------------------------------------------------------------
# Entity nodes
# --------------------------------------------------------------------------


def test_entity_nodes_are_derived_from_items(repo):
    a = _item("arxiv:2608.00001")
    a.entities.methods = [EntityRef(id="method:gnn", label="graph neural network")]
    a.entities.topics = [TopicRef(id="openalex:T10746", label="Urban Transport")]
    b = _item("arxiv:2608.00002")
    b.first_published = date(2026, 8, 12)
    b.entities.methods = [EntityRef(id="method:gnn", label="graph neural network")]
    store.save_item(a)
    store.save_item(b)

    n = rebuild_entity_nodes()
    assert n == 2

    node = store.load_entity("methods", "method:gnn")
    assert node.item_count == 2
    assert node.first_seen == date(2026, 8, 11)
    assert node.last_seen == date(2026, 8, 12)
    assert node.parent == "method:deep-learning"

    topic = store.load_entity("topics", "openalex:T10746")
    assert topic.canonical.openalex == "T10746"


def test_edges_are_derived_and_byte_stable(repo):
    from pipeline.graph.build import build_edges

    item = _item()
    item.entities.methods = [EntityRef(id="method:gnn", label="graph neural network")]
    item.graph.referenced_works = ["openalex:W2145"]
    store.save_item(item)

    n1 = build_edges()
    first = (repo / "content" / "graph" / "edges.jsonl").read_bytes()
    n2 = build_edges()
    second = (repo / "content" / "graph" / "edges.jsonl").read_bytes()

    assert n1 == n2 == 2
    assert first == second
    assert b'"type": "uses_method"' in first
    assert b'"type": "cites"' in first


# --------------------------------------------------------------------------
# Overlay extraction (its own prompt and version — D24 reverted D8)
# --------------------------------------------------------------------------


def test_extraction_has_its_own_prompt_version(repo):
    """The whole point of the split: editing the summary prompt must not
    invalidate extraction's cache, and vice versa."""
    from pipeline.llm import LLMClient

    assert LLMClient(task="extract").prompt_version != LLMClient(task="summarize").prompt_version


def test_extraction_normalises_and_caps_candidate_lists(repo):
    from pipeline.linking.extract import normalize_payload

    out = normalize_payload(
        {
            "methods": ["GNN", "gnn", "  Graph Neural Network  ", "a", "b", "c", "d", "e"],
            "data": ["Street View Imagery"],
            "tools": [],
            "places": None,
        }
    )
    assert out["methods"][:2] == ["gnn", "graph neural network"]  # deduped, lowered
    assert len(out["methods"]) <= 6
    assert out["data"] == ["street view imagery"]
    assert out["tools"] == [] and out["places"] == []


def test_extraction_rejects_a_non_object_response(repo):
    from pipeline.linking.extract import normalize_payload

    assert normalize_payload(None) is None
    assert normalize_payload(["not", "an", "object"]) is None


def test_extraction_is_skipped_without_credentials(repo):
    from pipeline.linking.extract import extract_overlay
    from pipeline.llm import LLMClient

    run = Run.for_date(date(2026, 8, 11))
    item = _item(abstract="Some abstract.")
    results, stats = extract_overlay([item], run, client=LLMClient(task="extract"))
    assert results == {}
    assert stats["status"] == "SKIPPED"


def test_extraction_failure_leaves_the_rule_scan_result(repo):
    """An extraction failure must cost tags, not the run."""
    from pipeline.llm import LLMClient, LLMResponse

    def broken(system, user):
        return LLMResponse(text="not json", input_tokens=1, output_tokens=1,
                           model="gemini-3.5-flash")

    run = Run.for_date(date(2026, 8, 11))
    item = _item(abstract="We train a convolutional neural network on satellite imagery.")
    stats = link_items(
        [item], run, use_llm=True, client=LLMClient(task="extract", caller=broken)
    )
    assert stats["extracted"] == 0
    # Fell through to the vocabulary scan rather than ending up with nothing.
    assert "method:cnn" in [r.id for r in item.entities.methods]
