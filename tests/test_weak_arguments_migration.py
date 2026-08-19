"""Renaming a category is not re-judging papers (phase 0Q, R1).

The labeller who made all fifteen weak judgements corrected what the label
meant: not that the *results* were thin but that the **argument** — the claim
the paper wants to make — is too narrow or too thin for what was measured. And,
crucially, that this **is** visible in an abstract.

So the verdicts stand and only the string changes, which is why this is a
migration with `--check` and `--revert` rather than a `corrected_from` append.
An append would record that someone changed their mind about a paper. Nobody
did.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline import paths

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "migrate_weak_arguments.py"


def _run(repo, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "UC_ROOT": str(paths.ROOT),
             "PYTHONIOENCODING": "utf-8"},
    )


def _write(rows, name="relevance.jsonl"):
    paths.LABELS.mkdir(parents=True, exist_ok=True)
    (paths.LABELS / name).write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read(name="relevance.jsonl"):
    return [json.loads(l) for l in
            (paths.LABELS / name).read_text(encoding="utf-8").splitlines() if l.strip()]


ROWS = [
    {"work_key": "a", "label": "keep", "date": "2026-08-05"},
    {"work_key": "b", "label": "drop_weak_results", "date": "2026-08-05"},
    {"work_key": "c", "label": "drop_weak_method", "date": "2026-08-05"},
]


def test_the_rename_moves_the_rows(repo):
    _write(ROWS)
    r = _run(repo)

    assert r.returncode == 0, r.stdout + r.stderr
    labels = [row["label"] for row in _read()]
    assert labels == ["keep", "drop_weak_arguments", "drop_weak_method"]


def test_check_reports_without_writing(repo):
    _write(ROWS)
    r = _run(repo, "--check")

    assert r.returncode == 0
    assert '"would rename": 1' in r.stdout
    assert [row["label"] for row in _read()][1] == "drop_weak_results", "untouched"


def test_revert_round_trips_byte_for_byte(repo):
    _write(ROWS)
    before = (paths.LABELS / "relevance.jsonl").read_bytes()

    _run(repo)
    assert (paths.LABELS / "relevance.jsonl").read_bytes() != before
    _run(repo, "--revert")

    assert (paths.LABELS / "relevance.jsonl").read_bytes() == before


def test_an_unknown_label_stops_the_whole_run(repo):
    """The same shape as `migrate_drop_lens.py` refusing to discard a non-null
    lens: data this does not understand is data it must not rewrite."""
    _write(ROWS + [{"work_key": "d", "label": "drop_something_new", "date": "2026-08-05"}])
    r = _run(repo)

    assert r.returncode == 1
    assert "does not know" in r.stdout
    assert [row["label"] for row in _read()][1] == "drop_weak_results", "nothing written"


def test_corrected_from_moves_with_the_label(repo):
    """`corrected_from` is an audit trail. Left behind, it would name a category
    that no longer exists."""
    _write([{"work_key": "b", "label": "keep", "date": "2026-08-05",
             "corrected_from": "drop_weak_results"}])
    _run(repo)

    assert _read()[0]["corrected_from"] == "drop_weak_arguments"


def test_every_label_file_is_migrated_not_just_the_ranked_one(repo):
    """Two of the seven real rows are in `subfield_check.jsonl`. A rename that
    reached one file would leave the same category under two names."""
    _write(ROWS)
    _write([{"work_key": "x", "label": "drop_weak_results", "date": "2026-06-13",
             "subfield": "2208", "sampling": "subfield_check"}], "subfield_check.jsonl")

    _run(repo)

    assert _read("subfield_check.jsonl")[0]["label"] == "drop_weak_arguments"


def test_the_judgements_themselves_are_untouched(repo):
    """Nothing but the name. No verdict flips, no row is added or removed."""
    _write(ROWS)
    before = _read()
    _run(repo)
    after = _read()

    assert len(after) == len(before)
    for b, a in zip(before, after):
        assert b["work_key"] == a["work_key"]
        assert b["date"] == a["date"]
        assert (b["label"] == a["label"]) or (
            b["label"] == "drop_weak_results" and a["label"] == "drop_weak_arguments"
        )
