"""The two label files may never become one.

`relevance.jsonl` is a ranked top-N sample per source; `affinity_probe.jsonl` is
equal draws from three `canon_affinity` bands, which deliberately over-samples
the bottom of the distribution. Pooling them produces a number shaped exactly
like precision@k and carrying no meaning, and nothing in the data would show it:
the rows have the same fields, the same label vocabulary, the same work keys.

So the separation is enforced in four places, and each one is pinned here:

1. `precision_at_k` refuses a probe facet by name.
2. `append_labels` refuses to write a row into a file drawn a different way.
3. `load_labels` refuses to read a file that already holds both.
4. `uc labels` routes by facet and offers no way to ask for both at once.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from pipeline.labeling import (
    PROBE_FACETS,
    RANKED_FACETS,
    LabelSetMisuse,
    append_labels,
    labels_path,
    load_labels,
    precision_at_k,
    probe_summary,
    sampling_of,
)

DAY = date(2026, 8, 11)


def _ranked_row(work_key: str = "W1", label: str = "keep") -> dict:
    return {
        "date": str(DAY),
        "work_key": work_key,
        "source": "journal",
        "rank": 1,
        "label": label,
        "score": 1.0,
        "sampling": "ranked_top_n",
    }


def _probe_row(work_key: str = "W2", band: str = "high", label: str = "keep") -> dict:
    return {
        "date": str(DAY),
        "work_key": work_key,
        "band": band,
        "rank_in_band": 1,
        "label": label,
        "canon_affinity": 12.2,
        "canon_hits": 6,
        "refs_total": 38,
        "source": "journal",
        "sampling": "band_stratified",
        "not_for_precision_at_k": True,
    }


# --------------------------------------------------------------------------
# 1. precision@k refuses the probe by name
# --------------------------------------------------------------------------


def test_precision_at_k_refuses_the_probe_facet(repo):
    append_labels("affinity_probe", [_probe_row()])
    with pytest.raises(LabelSetMisuse) as err:
        precision_at_k(facet="affinity_probe")
    assert "band-stratified" in str(err.value)
    assert "probe_summary" in str(err.value)


def test_precision_at_k_refuses_any_unknown_facet(repo):
    # Not a deny list. Anything that has not been declared a ranked sample is
    # refused, so a third label file added later cannot slip into the metric by
    # simply not being named here.
    with pytest.raises(LabelSetMisuse):
        precision_at_k(facet="whatever_comes_next")


def test_the_two_facet_sets_do_not_overlap():
    assert not (RANKED_FACETS & PROBE_FACETS)
    assert sampling_of("relevance") != sampling_of("affinity_probe")


# --------------------------------------------------------------------------
# 2. the write side
# --------------------------------------------------------------------------


def test_a_probe_row_cannot_be_appended_to_the_ranked_file(repo):
    with pytest.raises(LabelSetMisuse):
        append_labels("relevance", [_probe_row()])
    assert not labels_path("relevance").exists() or not load_labels("relevance")


def test_a_ranked_row_cannot_be_appended_to_the_probe_file(repo):
    with pytest.raises(LabelSetMisuse):
        append_labels("affinity_probe", [_ranked_row()])


def test_a_rejected_batch_writes_nothing_at_all(repo):
    # The check runs over the whole batch before the file is opened. A partial
    # write would leave the good rows in place and the bad ones lost, which is
    # the one outcome that cannot be recovered from the file afterwards.
    with pytest.raises(LabelSetMisuse):
        append_labels("affinity_probe", [_probe_row("W2"), _ranked_row("W3")])
    assert load_labels("affinity_probe") == []


# --------------------------------------------------------------------------
# 3. the read side, for files mixed outside the pipeline
# --------------------------------------------------------------------------


def test_a_concatenated_file_is_refused_on_read(repo):
    # `cat relevance.jsonl >> affinity_probe.jsonl` — the exact accident the
    # constraint exists to prevent, done outside every guard the code has.
    path = labels_path("affinity_probe")
    path.write_text(
        json.dumps(_probe_row()) + "\n" + json.dumps(_ranked_row()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LabelSetMisuse):
        load_labels("affinity_probe")
    with pytest.raises(LabelSetMisuse):
        probe_summary()


def test_legacy_ranked_rows_without_the_field_still_read(repo):
    # Labels already collected predate the field. Absent means ranked, because
    # the ranked file is the only one that existed when they were written.
    row = _ranked_row()
    del row["sampling"]
    labels_path("relevance").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert len(load_labels("relevance")) == 1


# --------------------------------------------------------------------------
# 4. what each file can answer
# --------------------------------------------------------------------------


def test_probe_summary_refuses_the_ranked_facet(repo):
    with pytest.raises(LabelSetMisuse):
        probe_summary("relevance")


def test_probe_summary_reports_keep_rate_by_band_and_says_it_is_not_precision(repo):
    append_labels(
        "affinity_probe",
        [
            _probe_row("W1", "high", "keep"),
            _probe_row("W2", "high", "keep"),
            _probe_row("W3", "zero", "drop_not_our_kind"),
            _probe_row("W4", "zero", "keep"),
        ],
    )
    out = probe_summary()
    assert out["by_band"]["high"]["keep_rate"] == 1.0
    assert out["by_band"]["zero"]["keep_rate"] == 0.5
    assert out["by_band"]["zero"]["drop_reasons"]["not_our_kind"] == 1
    # No precision figure is offered anywhere in the probe's summary.
    assert not any("precision" in k for k in out)
    assert out["not_comparable_with"].startswith("relevance.jsonl")


def test_the_two_files_are_separate_on_disk(repo):
    append_labels("relevance", [_ranked_row("W1")])
    append_labels("affinity_probe", [_probe_row("W2")])
    assert labels_path("relevance") != labels_path("affinity_probe")
    assert len(load_labels("relevance")) == 1
    assert len(load_labels("affinity_probe")) == 1


def test_the_probe_records_the_signal_it_was_drawn_on(repo):
    # A row must be re-scorable from this file alone: the pool it came from is
    # rebuilt from a canon and a reference base that both keep moving.
    append_labels("affinity_probe", [_probe_row()])
    row = load_labels("affinity_probe")[0]
    for field in ("canon_affinity", "canon_hits", "refs_total", "band"):
        assert field in row
