"""Data hygiene (phase 0c, P4).

Three separate problems that share a shape: a field is being filled now and read
much later, so a wrong value written today is an archive nobody can trust when
the axis it belongs to is revived.
"""

from __future__ import annotations

from datetime import date

from pipeline.collectors.base import clean_text
from pipeline.graph.build import edges_for_item
from pipeline.models import (
    Author,
    Bibliography,
    EntityRef,
    Institution,
    Item,
    PlaceRef,
    Signal,
    TopicRef,
)
from pipeline.run_stages import reconcile_places_status

DAY = date(2026, 8, 11)


def _item(**kw) -> Item:
    kw.setdefault("bibliography", Bibliography(title="A Paper"))
    return Item(
        work_key=kw.pop("work_key", "arxiv:2608.00001"), first_published=DAY, **kw
    )


# --------------------------------------------------------------------------
# P4-1 — markup in titles
# --------------------------------------------------------------------------


def test_a_title_arrives_without_publisher_markup():
    """The exact string that reached the Q1b labeller."""
    raw = "<scp>DIFFERENTIATED INFRASTRUCTURAL CITIZENSHIP</scp> : Claims‐Making"
    assert clean_text(raw) == "DIFFERENTIATED INFRASTRUCTURAL CITIZENSHIP: Claims‐Making"


def test_clean_text_still_collapses_ordinary_whitespace():
    assert clean_text("  two\n  lines  ") == "two lines"
    assert clean_text(None) is None
    assert clean_text("   ") is None


# --------------------------------------------------------------------------
# P4-2 — places_status
# --------------------------------------------------------------------------


def test_a_paper_with_no_study_area_is_not_recorded_as_unresolved():
    """`not_applicable` has been in the schema since it was written and nothing
    ever set it: "there is no place" and "we could not find the place" were
    stored identically."""
    item = _item()
    item.signals.geographic_scope = Signal(value="not_applicable", basis="llm")
    assert item.entities.places_status == "not_attempted"

    assert reconcile_places_status([item]) == 1
    assert item.entities.places_status == "not_applicable"


def test_an_unresolved_place_stays_unspecified():
    """The distinction only holds if the other side keeps its own value."""
    item = _item()
    item.signals.geographic_scope = Signal(value="city", basis="llm")
    item.entities.places_status = "unspecified"

    assert reconcile_places_status([item]) == 0
    assert item.entities.places_status == "unspecified"


def test_a_resolved_place_is_never_overwritten():
    """A scope of not_applicable next to a resolved place is a contradiction;
    the resolved place is the harder evidence, so it wins."""
    item = _item()
    item.signals.geographic_scope = Signal(value="not_applicable", basis="llm")
    item.entities.places = [PlaceRef(id="wikidata:Q60", label="New York City")]
    item.entities.places_status = "resolved"

    assert reconcile_places_status([item]) == 0
    assert item.entities.places_status == "resolved"


def test_reconciling_twice_changes_nothing_the_second_time():
    item = _item()
    item.signals.geographic_scope = Signal(value="not_applicable", basis="llm")
    reconcile_places_status([item])
    assert reconcile_places_status([item]) == 0


# --------------------------------------------------------------------------
# P4-3 — author and affiliation edges
# --------------------------------------------------------------------------


def test_author_and_affiliation_edges_are_emitted():
    """People & Orgs is a Phase 2 signature candidate and centrality needs these
    edges to already exist. If they stop being written, the archive has to be
    reprocessed to get them back — so this is asserted, not assumed."""
    item = _item(
        bibliography=Bibliography(
            title="A Paper",
            authors=[Author(name="Ada Lovelace", institutions=[Institution(ror="x")])],
        ),
    )
    item.entities.people = [EntityRef(id="orcid:0000-0001-0000-0000", label="Ada Lovelace")]
    item.entities.orgs = [EntityRef(id="ror:012345678", label="Test University")]
    item.entities.topics = [TopicRef(id="openalex:T1", label="Urban Studies", score=0.9)]

    edges = {(e.type, e.dst) for e in edges_for_item(item)}

    assert ("authored_by", "orcid:0000-0001-0000-0000") in edges
    assert ("affiliated_with", "ror:012345678") in edges
    assert ("has_topic", "openalex:T1") in edges


def test_every_edge_carries_the_items_date():
    item = _item()
    item.entities.people = [EntityRef(id="orcid:0000-0001-0000-0000", label="Ada")]
    assert all(e.date == DAY for e in edges_for_item(item))


# --------------------------------------------------------------------------
# A re-run's own outputs must reach content/
# --------------------------------------------------------------------------


def test_a_resummarised_item_updates_in_the_archive(repo):
    """`_merge_pair` accumulates — the base keeps what it has. Folding the
    stored copy in as base meant a re-run's summary was thrown away, so a
    prompt version bump could never reach a published item. Found when 121
    summaries were regenerated at 0.4.0 and content/ still said 0.3.0."""
    from pipeline.models import LlmProvenance, SummaryEn
    from pipeline.run_stages import _restore_run_outputs

    stored = _item()
    stored.summary.en = SummaryEn(what="Old what.", why="Old why.")
    stored.provenance.llm = LlmProvenance(model="m", prompt_version="papers@0.3.0")

    fresh = _item()
    fresh.summary.en = SummaryEn(what="New what.", why="New why.")
    fresh.provenance.llm = LlmProvenance(model="m", prompt_version="papers@0.4.0")
    fresh.scores.headline = 0.77

    _restore_run_outputs(stored, fresh)

    assert stored.summary.en.what == "New what."
    assert stored.provenance.llm.prompt_version == "papers@0.4.0"
    assert stored.scores.headline == 0.77


def test_a_run_that_produced_no_summary_does_not_erase_the_stored_one(repo):
    """`--no-llm`, a missing key, or a budget stop must not blank the archive."""
    from pipeline.models import LlmProvenance, SummaryEn
    from pipeline.run_stages import _restore_run_outputs

    stored = _item()
    stored.summary.en = SummaryEn(what="Old what.", why="Old why.")
    stored.provenance.llm = LlmProvenance(model="m", prompt_version="papers@0.3.0")

    _restore_run_outputs(stored, _item())

    assert stored.summary.en.what == "Old what."
    assert stored.provenance.llm.prompt_version == "papers@0.3.0"


# --------------------------------------------------------------------------
# Q0-1 — ROR identifiers are bare, like every other canonical prefix
# --------------------------------------------------------------------------


def test_a_ror_url_becomes_a_bare_identifier():
    from pipeline.collectors.base import normalize_ror

    assert normalize_ror("https://ror.org/02mhbdp94") == "02mhbdp94"
    assert normalize_ror("https://ror.org/02mhbdp94/") == "02mhbdp94"
    assert normalize_ror("02mhbdp94") == "02mhbdp94"
    assert normalize_ror(None) is None
    assert normalize_ror("") is None


def test_the_ror_migration_round_trips():
    """It has to be reversible: an identifier migration that cannot be undone is
    a decision nobody can walk back."""
    from scripts.migrate_ror_ids import to_bare, to_url

    url = "ror:https://ror.org/02mhbdp94"
    bare = "ror:02mhbdp94"
    assert to_bare(url) == bare
    assert to_url(bare) == url
    assert to_bare(bare) == bare, "already migrated: leave it alone"
    assert to_url(url) == url


# --------------------------------------------------------------------------
# T0 — identifiers are a lock-in point, so the slug rule has to be right
# --------------------------------------------------------------------------


def _singularise(word: str) -> str:
    import sys
    from pathlib import Path

    scripts = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from vocab_candidates import singularise

    return singularise(word)


def test_words_that_only_look_plural_keep_their_s():
    """`analysis` and `statistics` produced `method:comparative-analysi` and
    `method:descriptive-statistic`. Caught before approval this time; after
    approval it would have been a migration."""
    for word in ("analysis", "statistics", "economics", "hypothesis", "basis",
                 "mass", "census", "series", "species", "informatics", "robotics"):
        assert _singularise(word) == word, word


def test_real_plurals_are_still_singularised():
    cases = {
        "interviews": "interview", "methods": "method", "images": "image",
        "analyses": "analysis", "studies": "study", "cities": "city",
        "boxes": "box", "matrices": "matrix", "indices": "index",
    }
    for plural, singular in cases.items():
        assert _singularise(plural) == singular, plural


def test_only_the_last_word_is_singularised():
    from vocab_candidates import normalise

    assert normalise("comparative analysis") == "comparative analysis"
    assert normalise("descriptive statistics") == "descriptive statistics"
    assert normalise("case studies") == "case study"
    assert normalise("Kernel Density Estimation") == "kernel density estimation"


def test_no_candidate_identifier_ends_in_a_mangled_stem():
    """The whole point of the rule: nothing in the files carries the bug."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for name in ("methods", "data", "tools"):
        doc = yaml.safe_load((root / "vocab" / f"{name}.yaml").read_text(encoding="utf-8")) or {}
        for entry in doc.get("candidates") or []:
            slug = entry["suggested_id"]
            assert not slug.endswith("-analysi"), slug
            assert not slug.endswith("-statistic"), slug
            assert not slug.endswith("-serie"), slug


def test_the_stoplist_gives_a_reason_for_every_rejection():
    """A rejection with no recorded why is indistinguishable from an oversight,
    and the next harvest proposes the same term again."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    doc = yaml.safe_load((root / "vocab" / "extraction_stoplist.yaml").read_text(encoding="utf-8"))
    assert doc
    for facet, entries in doc.items():
        for entry in entries:
            assert entry.get("term"), entry
            assert entry.get("why"), f"{facet}/{entry.get('term')} gives no reason"


def test_generic_activities_do_not_reach_candidates():
    """`case study` appeared more often than most kept terms and is still
    worthless: it partitions nothing and connects everything to everything."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    doc = yaml.safe_load((root / "vocab" / "methods.yaml").read_text(encoding="utf-8")) or {}
    labels = {c["label"] for c in doc.get("candidates") or []}
    for term in ("case study", "literature review", "thematic analysis",
                 "comparative analysis", "descriptive statistics"):
        assert term not in labels, term
    rejected = {c["label"] for c in doc.get("rejected") or []}
    assert "case study" in rejected
