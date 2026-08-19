"""**Anything shown to a person must be recordable** (0Q hotfix, G3).

The same outcome — a human judgement silently discarded — has now happened
three times by three different mechanisms:

| # | what | mechanism |
|---|---|---|
| 1 | `code_probe`, 30 judgements | session not wired; the guard refused the rows at the end, and the session batched its writes so the refusal took all thirty |
| 2 | `--pending` (latent) | batched writes; caught before it cost anything |
| 3 | **`--pending`, 25 judgements** | **the write path required something the display path did not** |

The third is the general form of all of them. `run_pending_session` read the
title from the held row — which always has one — and wrote from
`store.load_item()`, which for a **withheld** item is always `None`, because a
withheld item was by definition never published and has no file. 116 of the 118
withheld rows are in that state, the queue shows withheld first, so all 25 rows
on page one were unwritable and every answer hit `continue`.

The invariant is one sentence and these tests are it:
**if a session can display a row, it can record a judgement on that row.**
"""

from __future__ import annotations

import json

import pytest

from pipeline import paths
from pipeline.labeling import (
    LabelWriteFailed,
    can_record,
    held_review_row,
    load_labels,
)

# A held row exactly as `content/held/*.json` stores it. No item, no summary,
# no score beyond what the rule recorded — the real withheld rows look like
# this, and this is the shape that used to be unwritable.
BARE_HELD_ROW = {
    "work_key": "arxiv:2606.20742",
    "date": "2026-06-17",
    "rule": "at_the_floor",
    "kind": "withheld",
    "detail": "0.826 is within 0.03 of the 0.8 floor",
    "score": 0.8261,
    "source": "arxiv",
    "title": "A Digital Twin Framework for Traffic-Aware UAV Pavement Inspection",
}


def _held_day(date_str, rows):
    d = paths.CONTENT / "held"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{date_str}.json").write_text(
        json.dumps({"date": date_str, "published": 10, "items": rows}, sort_keys=True),
        encoding="utf-8", newline="\n",
    )


# --------------------------------------------------------------------------
# G2 — the held row is enough on its own
# --------------------------------------------------------------------------


def test_a_judgement_is_written_with_no_item_file_at_all(repo):
    """The exact case that lost 25 answers."""
    from pipeline import store

    assert store.load_item(BARE_HELD_ROW["work_key"]) is None
    row = held_review_row(BARE_HELD_ROW, "keep")

    assert row["work_key"] == BARE_HELD_ROW["work_key"]
    assert row["rule"] == "at_the_floor"
    assert row["why_held"].startswith("0.826")
    assert row["score"] == 0.8261
    assert row["sampling"] == "held_review"


def test_the_row_records_how_much_the_labeller_could_see(repo):
    """A verdict reached on a held row alone must be distinguishable from one
    reached with the item in front of you — the `subfield_check` lesson."""
    from pipeline.models import Bibliography, Item

    bare = held_review_row(BARE_HELD_ROW, "keep")
    item = Item(work_key=BARE_HELD_ROW["work_key"],
                bibliography=Bibliography(title="The real title"))
    item.scores.relevance = 0.8261
    rich = held_review_row(BARE_HELD_ROW, "keep", item)

    assert bare["shown"] == "held_row"
    assert rich["shown"] == "held_row+item"
    assert rich["title"] == "The real title"


def test_an_item_enriches_and_is_never_required(repo):
    assert held_review_row(BARE_HELD_ROW, "keep", None)["title"]
    assert "has_summary" not in held_review_row(BARE_HELD_ROW, "keep")


# --------------------------------------------------------------------------
# G1 — no `continue` between an answer and a write
# --------------------------------------------------------------------------


@pytest.fixture
def queue_without_items(repo):
    """Twenty-five withheld rows, none of which has an item file. This is the
    real first page, not a synthetic edge case."""
    rows = []
    for i in range(25):
        rows.append({**BARE_HELD_ROW, "work_key": f"arxiv:2606.{i:05d}",
                     "title": f"A withheld paper {i}"})
    _held_day("2026-06-17", rows)
    return rows


def test_a_full_page_of_itemless_rows_is_fully_recorded(repo, queue_without_items):
    from pipeline import store
    from pipeline.review import run_pending_session

    assert all(store.load_item(r["work_key"]) is None for r in queue_without_items)

    answers = iter(["k", "n", "q", "m", "r"] * 5)
    result = run_pending_session(prompt=lambda _p: next(answers), printer=lambda *a: None)

    assert result["judged"] == 25
    assert len(load_labels("held_review")) == 25


def test_answered_and_written_are_reconciled_out_loud(repo, queue_without_items):
    said: list = []
    from pipeline.review import run_pending_session

    answers = iter(["k"] * 3 + ["quit"])
    run_pending_session(prompt=lambda _p: next(answers),
                        printer=lambda *a: said.append(" ".join(map(str, a))))

    assert any("answered 3, wrote 3" in line for line in said)


def test_a_failed_write_raises_rather_than_continues(repo, queue_without_items, monkeypatch):
    """The worst failure is the silent one: the person answered, the system did
    not take it, and nobody found out."""
    import pipeline.review as review_mod

    monkeypatch.setattr(review_mod, "run_pending_session",
                        review_mod.run_pending_session)
    import pipeline.labeling as lab
    monkeypatch.setattr(lab, "append_one", lambda facet, row: 0)

    from pipeline.review import run_pending_session

    with pytest.raises(LabelWriteFailed) as excinfo:
        run_pending_session(prompt=lambda _p: "k", printer=lambda *a: None)

    assert "answered and not written" in str(excinfo.value)


def test_a_skip_is_the_only_answer_that_writes_nothing(repo, queue_without_items):
    from pipeline.review import run_pending_session

    answers = iter(["s", "s", "k", "quit"])
    result = run_pending_session(prompt=lambda _p: next(answers), printer=lambda *a: None)

    assert result["counts"]["skip"] == 2
    assert len(load_labels("held_review")) == 1, "skips cost no row and no error"


# --------------------------------------------------------------------------
# G3 — the invariant itself, over every session
# --------------------------------------------------------------------------


def test_every_row_the_pending_session_would_show_is_recordable(repo, queue_without_items):
    """The invariant, checked the way it failed: take what the session would
    display and confirm each one can be written."""
    from pipeline.held import pending

    shown = pending()

    assert shown, "nothing to check would make this test vacuous"
    for row in shown:
        assert can_record(row), f"{row['work_key']} would be shown and not recordable"
        assert held_review_row(row, "keep")["work_key"] == row["work_key"]


def test_a_row_that_cannot_be_recorded_is_never_shown(repo):
    """Dropped before the first question, and counted on screen."""
    from pipeline.review import run_pending_session

    _held_day("2026-06-17", [
        {**BARE_HELD_ROW, "work_key": "arxiv:1"},
        {**BARE_HELD_ROW, "work_key": "", "title": "a row with no key"},
    ])
    said: list = []
    asked: list = []

    def prompt(_text=""):
        asked.append(1)
        return "quit"

    run_pending_session(prompt=prompt, printer=lambda *a: said.append(" ".join(map(str, a))))

    assert any("[SKIPPED] 1 held row" in line for line in said)
    assert len(asked) == 1, "only the recordable row was offered"


def test_the_subfield_session_already_held_the_invariant(repo):
    """Not every session had the bug. `subfield_check` loads the item purely to
    enrich the display and says so when it is absent, which is the pattern the
    pending session has now been moved to."""
    from pipeline.labeling import subfield_check_body

    body, basis = subfield_check_body(None)

    assert body == ""
    assert "not on disk" in basis
