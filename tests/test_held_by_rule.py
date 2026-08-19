"""One rule producing the whole queue must be visible (phase 0Q, R2).

0P emptied the `off_subfield` deny-list, so every *future* withholding comes
from `at_the_floor` alone — a rule whose 0.03 margin rests on a calibration
figure measured over a window containing no relevance labels (D196).

A single `withheld` total hides that. Three rules sharing a queue and one rule
owning it are the same number and completely different situations.
"""

from __future__ import annotations

import json

import pytest

from pipeline import paths, store
from pipeline.models import Bibliography, Item


def _day(date_str, rows):
    d = paths.CONTENT / "held"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{date_str}.json").write_text(
        json.dumps({"date": date_str, "published": 10, "items": rows}, sort_keys=True),
        encoding="utf-8", newline="\n",
    )


def _row(key, rule, kind):
    store.save_item(Item(work_key=key, bibliography=Bibliography(title=key)))
    return {"work_key": key, "rule": rule, "kind": kind, "detail": "d",
            "score": 0.81, "source": "arxiv", "title": key}


@pytest.fixture
def queue(repo):
    _day("2026-06-17", [
        _row("arxiv:1", "at_the_floor", "withheld"),
        _row("arxiv:2", "at_the_floor", "withheld"),
        _row("arxiv:3", "uncertain_score", "near_miss"),
        _row("doi:10.1/a", "off_subfield", "withheld"),
    ])


def test_counts_break_down_by_rule(queue):
    from pipeline.held import counts

    c = counts()

    assert c["withheld"] == 3
    assert c["by_rule"]["at_the_floor"] == {"withheld": 2, "near_miss": 0}
    assert c["by_rule"]["uncertain_score"] == {"withheld": 0, "near_miss": 1}
    assert c["by_rule"]["off_subfield"] == {"withheld": 1, "near_miss": 0}


def test_a_rule_that_cannot_fire_again_is_marked_inert(queue):
    """`off_subfield` holds items and can produce no more of them: the deny-list
    is empty. Unmarked, it reads as the rule withholding the most."""
    from pipeline.held import counts

    assert counts()["inert_rules"] == ["off_subfield"]


def test_a_rule_with_something_to_deny_is_not_inert(queue, monkeypatch):
    import pipeline.held as held_mod

    monkeypatch.setattr(held_mod, "rejected_subfield_ids", lambda: {"9999"})

    assert held_mod.counts()["inert_rules"] == []


def test_one_rule_owning_the_whole_withheld_queue_is_named(repo):
    _day("2026-06-18", [
        _row("arxiv:10", "at_the_floor", "withheld"),
        _row("arxiv:11", "at_the_floor", "withheld"),
        _row("arxiv:12", "uncertain_score", "near_miss"),
    ])
    from pipeline.held import counts

    assert counts()["withheld_by_one_rule"] == "at_the_floor"


def test_two_rules_withholding_is_not_flagged_as_one(queue):
    from pipeline.held import counts

    assert counts()["withheld_by_one_rule"] is None


def test_status_prints_the_breakdown(queue):
    from typer.testing import CliRunner

    from pipeline.cli import app

    out = CliRunner().invoke(app, ["status"]).stdout

    assert "at_the_floor" in out
    assert "off_subfield" in out
    assert "inert" in out
