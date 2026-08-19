"""A judgement survives the session it was made in (hotfix, F1-F3).

YJUN ran `uc review --label code_probe`, answered all thirty items, and lost
every one of them. Two faults met:

- `--label code_probe` fell through to the relevance session, which builds
  `ranked_top_n` rows, and the write guard correctly refused them;
- the session collected its rows in a list and wrote them **once at the end**,
  so the refusal took the whole sitting with it.

The guard was right. The wiring was missing, and the batching turned a wiring
mistake into thirty lost judgements.

This is the same shape as 0i's review timer, which recorded only sessions that
ran to completion and so dropped precisely the sessions of people who were busy.
That was moved into a `finally`. **Labels need more than that: a timing can be
measured again and a person's judgement cannot.**
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from pipeline import paths
from pipeline.labeling import (
    LabelSetMisuse,
    append_one,
    assert_writable,
    code_probe_pool,
    code_probe_row,
    labels_path,
    load_labels,
    run_code_probe_session,
)

POOL_ROWS = [
    {
        "work_key": f"arxiv:2606.{i:05d}",
        "title": f"A paper that releases code, number {i}",
        "date": "2026-06-26",
        "band": ["high", "mid", "low"][i % 3],
        "rank_in_band": i,
        "score": 0.9 - 0.1 * i,
        "code_basis": "rule",
        "source": "arxiv",
        "sampling": "code_stratified",
        "not_for_precision_at_k": True,
    }
    for i in range(6)
]


@pytest.fixture
def pool(repo):
    paths.LABELS.mkdir(parents=True, exist_ok=True)
    path = paths.LABELS / "code_probe_pool.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in POOL_ROWS) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _answers(*values):
    it = iter(values)

    def prompt(_text=""):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("the session asked more questions than expected")

    return prompt


# --------------------------------------------------------------------------
# F1 — every judgement is written when it is made
# --------------------------------------------------------------------------


def test_judgements_survive_an_exception_mid_session(repo, pool):
    """The whole point. Three answered, then the world ends: three are on disk."""
    answered = 0

    def prompt(_text=""):
        nonlocal answered
        if answered >= 3:
            raise KeyboardInterrupt("the labeller's terminal died")
        answered += 1
        return "k"

    with pytest.raises(KeyboardInterrupt):
        run_code_probe_session(prompt=prompt, printer=lambda *a: None)

    rows = load_labels("code_probe")
    assert len(rows) == 3, "answers given before the failure must be on disk"
    assert all(r["label"] == "keep" for r in rows)


def test_a_session_stopped_by_quit_keeps_what_was_answered(repo, pool):
    result = run_code_probe_session(
        prompt=_answers("k", "n", "quit"), printer=lambda *a: None
    )

    assert result["stopped_early"] is True
    assert len(load_labels("code_probe")) == 2


def test_re_running_does_not_ask_again(repo, pool):
    """Resuming must not re-ask what was already judged."""
    run_code_probe_session(prompt=_answers("k", "n", "quit"), printer=lambda *a: None)
    judged = {r["work_key"] for r in load_labels("code_probe")}

    remaining = code_probe_pool()
    assert judged.isdisjoint(r["work_key"] for r in remaining)
    assert len(remaining) == len(POOL_ROWS) - 2


def test_the_session_says_how_many_are_left(repo, pool):
    said: list = []
    run_code_probe_session(
        prompt=_answers("k", "quit"), printer=lambda *a: said.append(" ".join(map(str, a)))
    )

    assert any("left" in line for line in said)


def test_a_skip_is_not_written_but_is_not_asked_again_either(repo, pool):
    """A skip is an answer of sorts: it is counted, and it costs no row."""
    result = run_code_probe_session(
        prompt=_answers("s", "k", "quit"), printer=lambda *a: None
    )

    assert result["counts"].get("skip") == 1
    assert len(load_labels("code_probe")) == 1


# --------------------------------------------------------------------------
# F2 — the frame is checked before the first question
# --------------------------------------------------------------------------


def test_a_frame_mismatch_is_refused_before_anything_is_asked(repo):
    """The guard fired after thirty questions. It could have fired before one."""
    with pytest.raises(LabelSetMisuse) as excinfo:
        assert_writable("code_probe", "ranked_top_n")

    message = str(excinfo.value)
    assert "nothing was asked" in message
    # And it says what to run instead, not only what went wrong.
    assert "uc review --label code_probe" in message


def test_the_matching_frame_passes_silently(repo):
    assert_writable("code_probe", "code_stratified")
    assert_writable("relevance", "ranked_top_n")
    assert_writable("affinity_probe", "band_stratified")


def test_a_session_refuses_to_start_on_the_wrong_file(repo, pool, monkeypatch):
    """No question is asked when the session cannot write what it produces."""
    import pipeline.labeling as lab

    monkeypatch.setattr(lab, "sampling_of", lambda facet: "ranked_top_n")
    asked: list = []

    def prompt(_text=""):
        asked.append(1)
        return "k"

    with pytest.raises(LabelSetMisuse):
        run_code_probe_session(prompt=prompt, printer=lambda *a: None)

    assert asked == [], "not a single item may be shown"


# --------------------------------------------------------------------------
# F3 — the wiring, and no fall-through
# --------------------------------------------------------------------------


def test_the_code_probe_row_carries_why_the_item_was_in_the_pool(repo):
    row = code_probe_row(POOL_ROWS[0], "k")

    assert row["sampling"] == "code_stratified"
    assert row["label"] == "keep"
    assert row["score"] == POOL_ROWS[0]["score"]
    assert row["band"] == POOL_ROWS[0]["band"]
    assert row["code_basis"] == "rule"
    assert row["not_for_precision_at_k"] is True


def test_an_unknown_label_is_refused_rather_than_guessed():
    """The direct cause: `--label code_probe` fell through to the relevance
    session, whose rows the guard then refused."""
    from typer.testing import CliRunner

    from pipeline.cli import app

    result = CliRunner().invoke(app, ["review", "--label", "nonsense"])

    assert result.exit_code == 2
    assert "unknown --label" in result.stdout
    assert "code_probe" in result.stdout


def test_the_three_files_still_cannot_be_pooled(repo, pool):
    """Three frames, three questions. Any two concatenated mean nothing."""
    from pipeline.labeling import append_labels, precision_at_k

    with pytest.raises(LabelSetMisuse):
        precision_at_k("code_probe")

    for facet, wrong in (
        ("code_probe", "ranked_top_n"),
        ("relevance", "code_stratified"),
        ("affinity_probe", "code_stratified"),
    ):
        with pytest.raises(LabelSetMisuse):
            append_labels(facet, [{"work_key": "x", "label": "keep", "sampling": wrong}])


def test_append_one_writes_immediately(repo):
    """The primitive the sessions are built on."""
    append_one("code_probe", code_probe_row(POOL_ROWS[0], "k"))

    assert labels_path("code_probe").exists()
    assert len(load_labels("code_probe")) == 1


# --------------------------------------------------------------------------
# The subfield check — same discipline, fourth frame
# --------------------------------------------------------------------------

SUBFIELD_POOL = [
    {
        "work_key": f"doi:10.1/sub{i}",
        "title": f"A paper in a contested subfield, number {i}",
        "date": "2026-06-18",
        "subfield": ["1408", "2208", "2306", "3312"][i % 4],
        "source": "journal",
        "sampling": "subfield_check",
        "not_for_precision_at_k": True,
    }
    for i in range(4)
]


@pytest.fixture
def subfield_pool(repo):
    paths.LABELS.mkdir(parents=True, exist_ok=True)
    path = paths.LABELS / "subfield_check_pool.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in SUBFIELD_POOL) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_the_subfield_check_is_a_fourth_frame(repo):
    """Drawn by subfield — not a ranking, not a band. Four frames, four questions."""
    from pipeline.labeling import PROBE_FACETS, RANKED_FACETS, sampling_of

    assert sampling_of("subfield_check") == "subfield_check"
    assert "subfield_check" in PROBE_FACETS
    assert "subfield_check" not in RANKED_FACETS


def test_none_of_the_four_files_can_be_pooled(repo):
    from pipeline.labeling import LabelSetMisuse, append_labels, precision_at_k

    with pytest.raises(LabelSetMisuse):
        precision_at_k("subfield_check")

    for facet, wrong in (
        ("subfield_check", "ranked_top_n"),
        ("subfield_check", "code_stratified"),
        ("relevance", "subfield_check"),
        ("code_probe", "subfield_check"),
        ("affinity_probe", "subfield_check"),
    ):
        with pytest.raises(LabelSetMisuse):
            append_labels(facet, [{"work_key": "x", "label": "keep", "sampling": wrong}])


def test_subfield_judgements_survive_a_crash(repo, subfield_pool):
    from pipeline.labeling import run_subfield_check_session

    answered = 0

    def prompt(_text=""):
        nonlocal answered
        if answered >= 2:
            raise KeyboardInterrupt("terminal died")
        answered += 1
        return "k"

    with pytest.raises(KeyboardInterrupt):
        run_subfield_check_session(prompt=prompt, printer=lambda *a: None)

    rows = load_labels("subfield_check")
    assert len(rows) == 2
    # The subfield rides on the row: it is what the file exists to answer about.
    assert all(r["subfield"] in {"1408", "2208", "2306", "3312"} for r in rows)


def test_the_subfield_session_resumes(repo, subfield_pool):
    from pipeline.labeling import run_subfield_check_session, subfield_check_pool as pool

    run_subfield_check_session(prompt=_answers("k", "quit"), printer=lambda *a: None)
    judged = {r["work_key"] for r in load_labels("subfield_check")}

    assert len(judged) == 1
    assert judged.isdisjoint(r["work_key"] for r in pool())


def test_the_subfield_code_is_shown_on_every_item(repo, subfield_pool):
    """It is the thing being judged, so it cannot be off-screen."""
    from pipeline.labeling import run_subfield_check_session

    said: list = []
    run_subfield_check_session(
        prompt=_answers("k", "k", "quit"),
        printer=lambda *a: said.append(" ".join(map(str, a))),
    )

    headers = [line for line in said if line.startswith("[")]
    assert len(headers) >= 2
    for line in headers:
        assert "subfield" in line
        assert any(code in line for code in ("1408", "2208", "2306", "3312"))


def test_a_row_records_what_the_labeller_could_see(repo, subfield_pool):
    """A verdict reached on a title alone must be tellable from one reached on a
    summary — seven of the twenty real items have no abstract at all."""
    from pipeline.labeling import run_subfield_check_session

    run_subfield_check_session(prompt=_answers("k", "quit"), printer=lambda *a: None)
    row = load_labels("subfield_check")[0]

    assert "shown" in row
    assert row["sampling"] == "subfield_check"


def test_a_missing_body_is_stated_rather_than_hidden(repo):
    from pipeline.labeling import subfield_check_body

    body, basis = subfield_check_body(None)

    assert body == ""
    assert "not on disk" in basis
