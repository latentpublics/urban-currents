"""`uc review --pending` had both of the faults the F1 hotfix was written for.

The hotfix reached the sessions that had already gone wrong and stopped there.
`--pending` batched its judgements to the end of the sitting, and — worse —
wrote them into `relevance.jsonl` with `rank=0`.

That second one is not merely pooling. `precision_at_k` sorts each day by rank
and takes the head, so a rank-0 row sorts **above rank 1**: every held
judgement would have occupied the top of its day's top-ten window and moved the
one number this project reports. It had not fired yet — there is no rank-0 row
in the real file — because nobody had run `--pending` to the end.
"""

from __future__ import annotations

import json

import pytest

from pipeline import paths
from pipeline.labeling import (
    LabelSetMisuse,
    append_labels,
    load_labels,
    precision_at_k,
    sampling_of,
)


def _held_day(date_str, rows):
    paths.CONTENT.mkdir(parents=True, exist_ok=True)
    d = paths.CONTENT / "held"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{date_str}.json").write_text(
        json.dumps({"date": date_str, "published": 10, "items": rows}, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


@pytest.fixture
def queue(repo):
    from pipeline import store
    from pipeline.models import Bibliography, Item

    rows = []
    for i in range(4):
        key = f"arxiv:2606.{i:05d}"
        item = Item(work_key=key, bibliography=Bibliography(title=f"A held paper {i}"))
        item.scores.relevance = 0.81
        store.save_item(item)
        rows.append({
            "work_key": key,
            "rule": "at_the_floor",
            "kind": "withheld",
            "detail": "0.810 is within 0.03 of the 0.8 floor",
            "score": 0.81,
            "source": "arxiv",
            "title": f"A held paper {i}",
        })
    _held_day("2026-06-17", rows)
    return rows


def _answers(*values):
    it = iter(values)

    def prompt(_text=""):
        return next(it)

    return prompt


# --------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------


def test_the_held_queue_is_its_own_frame(repo):
    """Drawn by which rule stopped an item, not by where a ranking put it."""
    from pipeline.labeling import PROBE_FACETS, RANKED_FACETS

    assert sampling_of("held_review") == "held_review"
    assert "held_review" in PROBE_FACETS
    assert "held_review" not in RANKED_FACETS

    with pytest.raises(LabelSetMisuse):
        precision_at_k("held_review")


def test_held_rows_cannot_be_written_into_the_ranked_file(repo):
    """The exact write the old code performed."""
    with pytest.raises(LabelSetMisuse):
        append_labels("relevance", [
            {"work_key": "x", "label": "keep", "sampling": "held_review"}
        ])
    with pytest.raises(LabelSetMisuse):
        append_labels("held_review", [
            {"work_key": "x", "label": "keep", "sampling": "ranked_top_n"}
        ])


def test_a_judgement_records_which_rule_held_it(repo, queue):
    """Without that the row is an opinion about a paper and answers nothing
    about the queue it was drawn from."""
    from pipeline.review import run_pending_session

    run_pending_session(prompt=_answers("k", "quit"), printer=lambda *a: None)
    row = load_labels("held_review")[0]

    assert row["rule"] == "at_the_floor"
    assert row["kind"] == "withheld"
    assert row["why_held"].startswith("0.810 is within")
    assert row["sampling"] == "held_review"
    assert row["not_for_precision_at_k"] is True
    assert "rank" not in row, "the queue has no ranking"


# --------------------------------------------------------------------------
# Durability, the F1 fault again
# --------------------------------------------------------------------------


def test_judgements_survive_a_crash_mid_queue(repo, queue):
    from pipeline.review import run_pending_session

    answered = 0

    def prompt(_text=""):
        nonlocal answered
        if answered >= 2:
            raise KeyboardInterrupt("terminal died")
        answered += 1
        return "k"

    with pytest.raises(KeyboardInterrupt):
        run_pending_session(prompt=prompt, printer=lambda *a: None)

    assert len(load_labels("held_review")) == 2


def test_a_judged_item_is_not_offered_again(repo, queue):
    from pipeline.held import pending
    from pipeline.review import run_pending_session

    assert len(pending()) == 4
    run_pending_session(prompt=_answers("k", "n", "quit"), printer=lambda *a: None)

    assert len(load_labels("held_review")) == 2
    assert len(pending()) == 2


def test_a_relevance_judgement_also_settles_a_held_item(repo, queue):
    """Either file can answer for an item; asking again would be asking a
    question we have the answer to."""
    from pipeline.held import pending
    from pipeline.labeling import append_one

    append_one("relevance", {
        "work_key": "arxiv:2606.00000",
        "date": "2026-06-17",
        "label": "keep",
        "rank": 1,
        "sampling": "ranked_top_n",
    })

    assert len(pending()) == 3


def test_the_queue_pages_at_twenty_five_withheld_first(repo):
    """116 withheld and 640 near misses in the real queue; a page that mixed
    them would bury the ones that actually cost an issue a slot."""
    from pipeline import store
    from pipeline.held import WITHHELD, pending
    from pipeline.models import Bibliography, Item
    from pipeline.review import PENDING_BATCH

    rows = []
    for i in range(40):
        key = f"arxiv:2607.{i:05d}"
        store.save_item(Item(work_key=key, bibliography=Bibliography(title=f"p{i}")))
        rows.append({
            "work_key": key,
            "rule": "uncertain_score" if i % 2 else "at_the_floor",
            "kind": "near_miss" if i % 2 else WITHHELD,
            "detail": "d",
            "score": 0.7,
            "source": "arxiv",
            "title": f"p{i}",
        })
    _held_day("2026-06-18", rows)

    q = pending()
    assert PENDING_BATCH == 25
    page = q[:PENDING_BATCH]
    withheld_total = sum(1 for r in q if r["kind"] == WITHHELD)
    assert sum(1 for r in page if r["kind"] == WITHHELD) == withheld_total
    assert [r["kind"] for r in page][:withheld_total] == [WITHHELD] * withheld_total
