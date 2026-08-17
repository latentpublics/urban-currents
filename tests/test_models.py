"""Schema invariants (PRD §3, §9)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.models import (
    Bibliography,
    Entity,
    EntityRef,
    Institution,
    Issue,
    Item,
    TopicRef,
    work_key_to_filename,
)


def _item(**kw) -> Item:
    return Item(work_key="arxiv:2608.01234", bibliography=Bibliography(title="T"), **kw)


def test_entity_ref_rejects_free_strings():
    """A tag without a canonical prefix must not be constructible (PRD §9)."""
    with pytest.raises(ValidationError):
        EntityRef(id="graph neural network", label="graph neural network")


@pytest.mark.parametrize(
    "eid",
    ["method:gnn", "data:street-view", "github:gboeing/osmnx", "openalex:T10746",
     "orcid:0000-0002-1825-0097", "ror:02jx3x895", "wikidata:Q8684"],
)
def test_entity_ref_accepts_canonical_prefixes(eid):
    assert EntityRef(id=eid, label="x").id == eid


def test_a_ror_url_is_normalised_rather_than_accepted():
    """`ror:https://ror.org/X` passes the prefix test and is still wrong.

    The prefix announces the scheme and the value repeats it, and every other
    canonical prefix in this schema carries a bare identifier. Phase 0d migrated
    the archive; the form came back the moment a stage file written before that
    migration was merged into a published item, so the rule now lives in the
    schema instead of in a script that ran once.
    """
    assert EntityRef(id="ror:https://ror.org/02jx3x895", label="x").id == "ror:02jx3x895"
    assert Institution(ror="https://ror.org/02jx3x895").ror == "02jx3x895"
    assert Institution(ror="02jx3x895").ror == "02jx3x895"


def test_work_key_shape_is_enforced():
    with pytest.raises(ValidationError):
        Item(work_key="2608.01234", bibliography=Bibliography(title="T"))


def test_work_key_to_filename_replaces_colon():
    assert work_key_to_filename("arxiv:2608.01234") == "arxiv_2608.01234.json"
    assert work_key_to_filename("arxiv:cs.CY/0701001") == "arxiv_cs.CY_0701001.json"


def test_unknown_fields_are_rejected():
    """Schema drift should surface as a failure, not be silently absorbed."""
    with pytest.raises(ValidationError):
        Item(
            work_key="arxiv:2608.01234",
            bibliography=Bibliography(title="T"),
            surprise_field=1,
        )


def test_item_roundtrips_through_json():
    item = _item()
    item.entities.topics = [TopicRef(id="openalex:T10746", label="Urban Transport", score=0.8)]
    again = Item.model_validate_json(item.model_dump_json(by_alias=True))
    assert again.entities.topics[0].id == "openalex:T10746"


def test_status_change_from_alias_roundtrips():
    issue = Issue(date="2026-08-14")
    issue.status_changes.append(
        __import__("pipeline.models", fromlist=["StatusChange"]).StatusChange(
            work_key="arxiv:2604.09876", **{"from": "preprint"}, to="published", journal="Cities"
        )
    )
    dumped = issue.model_dump(mode="json", by_alias=True)
    assert dumped["status_changes"][0]["from"] == "preprint"
    assert Issue.model_validate(dumped).status_changes[0].from_ == "preprint"


def test_entity_requires_canonical_id():
    with pytest.raises(ValidationError):
        Entity(id="gnn", facet="methods", label="graph neural network")


def test_places_status_defaults_to_not_attempted():
    """The field stays in the schema even though Places is de-prioritised, so
    reviving the axis later does not mean reprocessing the archive (PRD §2)."""
    assert _item().entities.places_status == "not_attempted"
