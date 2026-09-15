"""One rule producing the whole queue must be visible (phase 0Q, R2).

0P emptied the `off_subfield` deny-list, so every *future* withholding came
from `at_the_floor` alone — a rule whose 0.03 margin rested on a calibration
figure measured over a window containing no relevance labels (D196).

A single `withheld` total hides that. Three rules sharing a queue and one rule
owning it are the same number and completely different situations.

**1H turned that premise over rather than out.** `at_the_floor` was demoted to
`near_miss`, so the answer to "which rule owns the withheld queue" is now "no
rule, there is no withheld queue" — and *that* state has to be legible too. The
breakdown is what makes it legible, which is why these tests stay: the cases
that had one rule owning the queue still hold, with the switch on, and a new
one pins the resting state under the shipped config.

The distinction the whole file turns on: `at_the_floor` is **inert and busy**.
It files more rows than anything else and withholds none of them. `inert_rules`
answers "can it withhold again", not "does it still run".
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
def enforcing(monkeypatch):
    """The world before 1H: R3 removes items rather than filing them.

    The historical rows in `queue` were written by that rule while it was
    withholding, and reading them back says nothing about whether it still can.
    `_rule_can_fire` asks the config, so the config has to say so.
    """
    import pipeline.held as held_mod

    monkeypatch.setattr(held_mod, "_at_the_floor_withholds", lambda: True)


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


def test_a_rule_that_cannot_fire_again_is_marked_inert(queue, enforcing):
    """`off_subfield` holds items and can produce no more of them: the deny-list
    is empty. Unmarked, it reads as the rule withholding the most."""
    from pipeline.held import counts

    assert counts()["inert_rules"] == ["off_subfield"]


def test_a_rule_with_something_to_deny_is_not_inert(queue, enforcing, monkeypatch):
    import pipeline.held as held_mod

    monkeypatch.setattr(held_mod, "rejected_subfield_ids", lambda: {"9999"})

    assert held_mod.counts()["inert_rules"] == []


def test_the_floor_rule_is_inert_under_the_shipped_config(queue):
    """1H. Both switchable rules are off, by different routes.

    `off_subfield` has nothing to deny; `at_the_floor` has plenty to catch and
    is switched off. The mark has to reach the second case too, or the largest
    block of history in this file reads as standing policy.
    """
    from pipeline.held import counts

    assert counts()["inert_rules"] == ["at_the_floor", "off_subfield"]


def test_an_inert_rule_still_fills_the_queue(repo):
    """Inert is about withholding, not about running.

    `at_the_floor` is the busiest rule in `content/held/` and takes nothing out
    of an issue. If demotion had stopped it filing, [0.80,0.83) would stay
    unmeasured forever, which is the one thing 1H was careful not to do.
    """
    from pipeline.held import RULE_AT_THE_FLOOR, counts, inspect
    from pipeline.models import Bibliography, Item

    item = Item(work_key="arxiv:floor", bibliography=Bibliography(title="t"))
    item.scores.relevance = 0.8018
    suspicion = inspect(item, "arxiv", selected=True, floor=0.80)

    assert suspicion is not None and suspicion.rule == RULE_AT_THE_FLOOR

    _day("2026-06-19", [_row("arxiv:20", "at_the_floor", "near_miss")])
    c = counts()
    assert c["by_rule"]["at_the_floor"] == {"withheld": 0, "near_miss": 1}
    assert c["inert_rules"] == ["at_the_floor"]


def test_an_empty_withheld_queue_names_no_rule_and_is_not_a_fault(repo):
    """The resting state since 1H: everything waiting, nothing withheld.

    `withheld_by_one_rule` is None here and None when several rules withhold,
    and neither of those is what the field claims — it only ever says "one rule
    is the queue". Zero is not a diagnosis and must not read as one.
    """
    _day("2026-06-20", [
        _row("arxiv:30", "at_the_floor", "near_miss"),
        _row("arxiv:31", "at_the_floor", "near_miss"),
        _row("arxiv:32", "uncertain_score", "near_miss"),
    ])
    from pipeline.held import counts

    c = counts()
    assert c["withheld"] == 0
    assert c["near_miss"] == 3
    assert c["waiting"] == 3
    assert c["withheld_by_one_rule"] is None


def test_a_queue_with_nothing_withheld_does_not_warn(repo):
    """Nothing withheld cannot cross a withholding-rate line."""
    from pipeline.held import over_warn_threshold

    assert over_warn_threshold(published=3, withheld=0) is None


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


def test_status_prints_the_breakdown(queue, enforcing):
    from typer.testing import CliRunner

    from pipeline.cli import app

    out = CliRunner().invoke(app, ["status"]).stdout

    assert "at_the_floor" in out
    assert "off_subfield" in out
    assert "inert" in out
