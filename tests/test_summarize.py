"""Summarize stage — fully mocked. No test may reach a real API (CLAUDE.md)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from pipeline.llm import LLMClient, LLMResponse, cache_get, parse_json
from pipeline.metrics import Run
from pipeline.models import Bibliography, Item
from pipeline.summarize.run import summarize_items, validate_payload

GOOD = {
    "what": "The authors train a model on 3.4M street-view images across 12 cities "
            "to predict pedestrian volume at 15 m resolution, reaching R2 = 0.71.",
    "why": "Pedestrian counts have been available only where counters exist; this "
           "puts a number on the rest of the street network.",
    "caveats": None,
    "geographic_scope": "multi_city",
    "data_available": False,
    "methods": ["convolutional neural network"],
    "data": ["street view imagery", "traffic sensor data"],
    "tools": ["OSMnx"],
    "places": ["Seoul", "Toronto"],
}


def _item(work_key="arxiv:2608.01234") -> Item:
    return Item(
        work_key=work_key,
        first_published=date(2026, 8, 11),
        bibliography=Bibliography(
            title="Street-View Imagery and Pedestrian Volume",
            abstract="We train a convolutional model on 3.4M street-view images "
                     "across 12 cities to predict pedestrian volume at 15 m "
                     "resolution using 2019-2023 counter data.",
        ),
    )


def fake_caller(payload, tokens=(1000, 200)):
    def _call(system: str, user: str) -> LLMResponse:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResponse(text=body, input_tokens=tokens[0], output_tokens=tokens[1],
                           model="claude-sonnet-5")
    return _call


def test_valid_response_fills_both_layers(repo):
    run = Run.for_date(date(2026, 8, 11))
    item = _item()
    client = LLMClient(caller=fake_caller(GOOD))

    stats = summarize_items([item], run, client=client)

    assert stats["summarized"] == 1
    assert item.summary.en.what.startswith("The authors train")
    assert item.summary.en.why
    assert item.summary.en.caveats is None
    assert item.signals.geographic_scope.value == "multi_city"
    assert item.signals.geographic_scope.basis == "llm"
    assert item.provenance.llm.model == "claude-sonnet-5"
    assert item.provenance.tokens.input == 1000


def test_bibliography_is_never_taken_from_the_model(repo):
    """The model invents author lists and years; those come from collectors."""
    run = Run.for_date(date(2026, 8, 11))
    item = _item()
    hostile = dict(GOOD)
    hostile["title"] = "A Totally Different Title"
    hostile["authors"] = ["Fake Person"]
    hostile["publication_date"] = "1999-01-01"

    summarize_items([item], run, client=LLMClient(caller=fake_caller(hostile)))

    assert item.bibliography.title == "Street-View Imagery and Pedestrian Volume"
    assert item.bibliography.authors == []
    assert item.bibliography.publication_date is None


def test_schema_violation_retries_then_leaves_the_item_pending(repo):
    """A bad response must not stop the day's issue (PRD §5.5)."""
    run = Run.for_date(date(2026, 8, 11))
    items = [_item("arxiv:2608.00001"), _item("arxiv:2608.00002")]
    calls = {"n": 0}

    def flaky(system, user):
        calls["n"] += 1
        # First item always fails (both attempts); second item succeeds.
        if "00001" in user or calls["n"] <= 2:
            return LLMResponse(text="not json at all", input_tokens=10, output_tokens=5,
                               model="claude-sonnet-5")
        return LLMResponse(text=json.dumps(GOOD), input_tokens=10, output_tokens=5,
                           model="claude-sonnet-5")

    stats = summarize_items(items, run, client=LLMClient(caller=flaky))

    assert stats["failures"] >= 1
    assert items[0].review.status == "pending"
    assert stats["summarized"] >= 1  # the run continued past the failure


def test_response_is_cached_and_not_requested_twice(repo):
    """Same item + same prompt version must not hit the API again."""
    run = Run.for_date(date(2026, 8, 11))
    calls = {"n": 0}

    def counting(system, user):
        calls["n"] += 1
        return LLMResponse(text=json.dumps(GOOD), input_tokens=10, output_tokens=5,
                           model="claude-sonnet-5")

    summarize_items([_item()], run, client=LLMClient(caller=counting))
    assert calls["n"] == 1
    assert cache_get("summarize/papers@0.2.0", "arxiv:2608.01234") is not None

    summarize_items([_item()], run, client=LLMClient(caller=counting))
    assert calls["n"] == 1, "second run should have been served from the cache"


def test_no_api_key_is_skipped_not_fatal(repo):
    run = Run.for_date(date(2026, 8, 11))
    item = _item()
    stats = summarize_items([item], run, client=LLMClient())
    assert stats["status"] == "SKIPPED"
    assert item.summary.en is None
    # Rule-based signals still run, so the card is not empty.
    assert item.signals.sample_size_reported is not None
    assert item.badges


def test_per_run_call_cap_stops_the_stage(repo):
    run = Run.for_date(date(2026, 8, 11))
    items = [_item(f"arxiv:2608.0000{i}") for i in range(1, 6)]
    client = LLMClient(caller=fake_caller(GOOD))
    client.max_calls_this_run = 2

    stats = summarize_items(items, run, client=client)

    assert stats["status"] == "PARTIAL"
    assert stats["summarized"] == 2
    assert "cap reached" in (stats["budget_stop"] or "")


def test_overlay_candidates_are_stashed_for_the_link_stage(repo):
    run = Run.for_date(date(2026, 8, 11))
    summarize_items([_item()], run, client=LLMClient(caller=fake_caller(GOOD)))

    from pipeline.summarize.run import load_overlay_stash

    stash = load_overlay_stash(run)
    assert stash["arxiv:2608.01234"]["methods"] == ["convolutional neural network"]
    assert "seoul" in stash["arxiv:2608.01234"]["places"]


# --------------------------------------------------------------------------
# Contract helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"what": ""}, {"what": "x"}, {"why": "y"}, {"what": "x", "why": "  "}],
)
def test_validate_payload_rejects_incomplete_output(payload):
    assert validate_payload(payload) is None


def test_validate_payload_accepts_both_layers():
    assert validate_payload({"what": "x", "why": "y"}) is not None


def test_parse_json_tolerates_code_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here you go:\n{"a": 1}') == {"a": 1}
    assert parse_json("no json here") is None
