"""What population is precision@10 computed over? (phase 0P, Q2)

Two ways the number could be right-looking and wrong, both of which the label
file can actually produce today:

- **Double counting.** The relabel session appends rather than edits, so a
  re-judged paper is a second row for the same (date, work_key). The file holds
  163 rows and 148 judgements. Counting rows would count 15 papers twice, once
  under the verdict that was withdrawn.
- **Thin days.** A day whose top ten is only three-tenths labelled produced a
  precision figure with the same shape as a full day's and the same weight in
  the mean. `scripts/journal_gate.py` has refused those since N4;
  `precision_at_k` did not.
"""

from __future__ import annotations

import json


from pipeline import paths
from pipeline.labeling import (
    MIN_TOP_K_COVERAGE,
    load_labels,
    precision_at_k,
    superseded,
)


def _row(work_key, rank, label, date="2026-08-05", source="arxiv", **extra):
    row = {
        "work_key": work_key,
        "title": f"paper {work_key}",
        "date": date,
        "rank": rank,
        "label": label,
        "score": 0.9,
        "source": source,
        "sampling": "ranked_top_n",
        "labelled_at": f"2026-08-12T00:00:{rank:02d}+00:00",
    }
    row.update(extra)
    return row


def _write(repo, rows):
    paths.LABELS.mkdir(parents=True, exist_ok=True)
    (paths.LABELS / "relevance.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


# --------------------------------------------------------------------------
# superseded(): one judgement per paper, and it is the newest one
# --------------------------------------------------------------------------


def test_a_rejudged_paper_is_counted_once_and_by_its_new_verdict(repo):
    """The whole point of `superseded`. Ten items; the first was `drop_weak`
    and has been re-judged `keep`. Precision must be 0.1, not 0.0909."""
    rows = [_row(f"arxiv:{i}", i, "drop_not_urban") for i in range(1, 11)]
    rows[0] = _row("arxiv:1", 1, "drop_weak")
    rows.append(
        _row(
            "arxiv:1",
            1,
            "keep",
            corrected_from="drop_weak",
            corrected_at="2026-08-17T16:35:07+00:00",
        )
    )
    _write(repo, rows)

    assert len(load_labels("relevance")) == 11
    assert len(superseded(load_labels("relevance"))) == 10

    result = precision_at_k("relevance", k=10)
    assert result["n_labels"] == 10, "the file has 11 rows and 10 judgements"
    assert result["by_source"]["arxiv"]["precision_at_10"] == 0.1


def test_the_newest_verdict_wins_regardless_of_file_order(repo):
    """A correction sorts by `corrected_at`, not by where it landed in the file."""
    late = _row("arxiv:1", 1, "keep", corrected_from="drop_weak",
                corrected_at="2026-08-17T16:35:07+00:00")
    original = _row("arxiv:1", 1, "drop_weak")
    _write(repo, [late, original])

    kept = superseded(load_labels("relevance"))
    assert len(kept) == 1
    assert kept[0]["label"] == "keep"


def test_a_correction_whose_original_is_missing_still_counts_once(repo):
    """The real file has exactly one of these: a row carrying `corrected_from`
    whose original write never made it to disk. It is a judgement, and it must
    neither be dropped nor duplicated."""
    _write(repo, [_row("arxiv:1", 1, "keep", corrected_from="drop_weak",
                       corrected_at="2026-08-17T16:35:07+00:00")])

    kept = superseded(load_labels("relevance"))
    assert len(kept) == 1
    assert kept[0]["label"] == "keep"


def test_the_same_paper_on_two_days_is_two_judgements(repo):
    """Collapsing is by (date, work_key): the same paper offered on two days is
    two chances the ranking had, not one."""
    _write(repo, [
        _row("arxiv:1", 1, "keep", date="2026-08-05"),
        _row("arxiv:1", 1, "drop_not_urban", date="2026-08-06"),
    ])

    assert len(superseded(load_labels("relevance"))) == 2


# --------------------------------------------------------------------------
# Thin days are named, not averaged in
# --------------------------------------------------------------------------


def test_a_thinly_labelled_day_does_not_produce_a_precision_figure(repo):
    """Three of ten labelled, all keeps. That is not a 1.0 day."""
    full = [_row(f"arxiv:a{i}", i, "keep" if i <= 6 else "drop_not_urban",
                 date="2026-08-05") for i in range(1, 11)]
    thin = [_row(f"arxiv:b{i}", i, "keep", date="2026-08-06") for i in range(1, 4)]
    _write(repo, full + thin)

    arx = precision_at_k("relevance", k=10)["by_source"]["arxiv"]

    assert arx["precision_at_10"] == 0.6, "only the full day counts"
    assert arx["days"] == 1
    assert arx["days_unmeasured"] == 1
    assert arx["unmeasured_days"] == [{"date": "2026-08-06", "labelled_in_top_k": 3}]


def test_an_unmeasured_day_is_named_rather_than_dropped(repo):
    """`days_unmeasured` exists so a day nobody labelled cannot be mistaken for
    a day that scored zero."""
    thin = [_row(f"arxiv:b{i}", i, "drop_not_urban", date="2026-08-06")
            for i in range(1, 3)]
    _write(repo, thin)

    arx = precision_at_k("relevance", k=10)["by_source"]["arxiv"]

    assert arx["precision_at_10"] is None
    assert arx["days"] == 0
    assert [d["date"] for d in arx["unmeasured_days"]] == ["2026-08-06"]


def test_labels_below_the_window_do_not_count_as_coverage(repo):
    """Ten labels, none of them in the top ten by rank."""
    rows = [_row(f"arxiv:{i}", i, "keep", date="2026-08-05") for i in range(11, 21)]
    _write(repo, rows)

    arx = precision_at_k("relevance", k=10)["by_source"]["arxiv"]

    assert arx["precision_at_10"] is None
    assert arx["unmeasured_days"] == [{"date": "2026-08-05", "labelled_in_top_k": 0}]


def test_the_bar_is_the_one_the_journal_gate_has_used_since_n4(repo):
    assert MIN_TOP_K_COVERAGE == 8
    rows = [_row(f"arxiv:{i}", i, "keep", date="2026-08-05") for i in range(1, 9)]
    _write(repo, rows)

    arx = precision_at_k("relevance", k=10)["by_source"]["arxiv"]

    assert arx["days"] == 1, "8 of 10 is the bar, and it is met"
    assert arx["precision_at_10"] == 1.0


# --------------------------------------------------------------------------
# The split changed the diagnosis, not the metric
# --------------------------------------------------------------------------


def test_splitting_drop_weak_leaves_precision_alone(repo):
    """M1 split one label into two. Both are still drops, so a precision that
    moved would mean the metric had been moved rather than measured."""
    before = [_row(f"arxiv:{i}", i, "keep" if i <= 6 else "drop_weak",
                   date="2026-08-05") for i in range(1, 11)]
    _write(repo, before)
    was = precision_at_k("relevance", k=10)["by_source"]["arxiv"]["precision_at_10"]

    after = before + [
        _row(f"arxiv:{i}", i, "drop_weak_method" if i % 2 else "drop_weak_arguments",
             date="2026-08-05", corrected_from="drop_weak",
             corrected_at="2026-08-17T16:35:07+00:00")
        for i in range(7, 11)
    ]
    _write(repo, after)
    now = precision_at_k("relevance", k=10)["by_source"]["arxiv"]

    assert now["precision_at_10"] == was
    assert now["n_labels"] == 10
    assert now["weak_detail"] == {"method": 2, "arguments": 2, "unsplit": 0}


def test_a_row_with_no_source_does_not_break_the_metric(repo):
    """The group set defaulted a missing `source` to "unknown" and the filter
    compared against the raw value, so the group matched nothing and divided by
    zero. Found by writing exactly such a row into the real file by accident."""
    rows = [_row(f"arxiv:{i}", i, "keep", date="2026-08-05") for i in range(1, 11)]
    rows.append({"work_key": "x", "date": "2026-08-05", "rank": 1,
                 "label": "keep", "sampling": "ranked_top_n"})
    _write(repo, rows)

    result = precision_at_k("relevance", k=10)

    assert result["by_source"]["arxiv"]["precision_at_10"] == 1.0
    assert "unknown" in result["by_source"], "the odd row is visible, not fatal"
    assert result["by_source"]["unknown"]["n_labels"] == 1
