"""A judgement file outside version control has to say so (1H, C4).

`held_review.jsonl` — 122 judgements, the largest single labelling session this
project has produced — sat untracked for three weeks. It was never ignored:
`.gitignore` excludes `runs/*`, re-includes `runs/labels/`, then excludes only
`runs/labels/*_pool.jsonl`, so a judgement file is meant to be tracked. It
simply was not added, and **nothing in the project said so**.

The cost was not the risk of losing it. It was that D196 went on being cited as
*"the window is unmeasured"* for three weeks after the window had been measured,
because the answer was in a file no analysis looked at.

These tests pin the two cheap facts and, more importantly, the two silences:
pool files are not judgements, and a missing `git` is not an alarm.
"""

from __future__ import annotations

import json
import subprocess

from pipeline import paths


def _write(name: str, rows: list[dict]) -> None:
    paths.LABELS.mkdir(parents=True, exist_ok=True)
    (paths.LABELS / name).write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8", newline="\n",
    )


def _row(at: str = "2026-08-20T00:00:00+00:00") -> dict:
    return {
        "date": "2026-08-20",
        "work_key": "arxiv:2608.1",
        "label": "keep",
        "sampling": "held_review",
        "labelled_at": at,
    }


def test_an_untracked_judgement_file_is_named(repo, monkeypatch):
    """The 1H case, reproduced: a judgement file git does not know about."""
    import pipeline.labeling as lab

    _write("held_review.jsonl", [_row()])

    class _Done:
        returncode = 0
        stdout = "runs/labels/relevance.jsonl\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Done())

    assert lab.label_file_health()["untracked"] == ["held_review.jsonl"]


def test_a_tracked_file_is_not_named(repo, monkeypatch):
    import pipeline.labeling as lab

    _write("held_review.jsonl", [_row()])

    class _Done:
        returncode = 0
        stdout = "runs/labels/held_review.jsonl\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Done())

    assert lab.label_file_health()["untracked"] == []


def test_pool_files_are_not_judgements(repo, monkeypatch):
    """`*_pool.jsonl` is a regenerable candidate list and is ignored on purpose.

    Naming one would train the reader to ignore the line, which is the only way
    a warning this small can fail.
    """
    import pipeline.labeling as lab

    _write("code_probe_pool.jsonl", [{"work_key": "arxiv:1"}])

    class _Done:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Done())

    health = lab.label_file_health()
    assert health["untracked"] == []
    assert health["files"] == 0


def test_no_git_is_not_an_alarm(repo, monkeypatch):
    """Quiet and open. An unanswerable question is not a missing file.

    `git` may be absent, this may be an unpacked tarball, the call may time out.
    None of that is evidence that a judgement is unversioned, and a status line
    is not worth an exception.
    """
    import pipeline.labeling as lab

    _write("held_review.jsonl", [_row()])

    def _boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", _boom)

    assert lab.label_file_health()["untracked"] == []


def test_the_last_judgement_date_is_reported(repo):
    """The second fact: how long since anyone judged anything."""
    import pipeline.labeling as lab

    _write("held_review.jsonl", [_row("2026-08-20T00:00:00+00:00")])
    _write("relevance.jsonl", [{
        "date": "2026-08-11",
        "work_key": "doi:10.1/a",
        "label": "keep",
        "sampling": "ranked_top_n",
        "labelled_at": "2026-08-13T00:00:00+00:00",
    }])

    health = lab.label_file_health()

    # The newest across every facet, not per file.
    assert health["last_judged"] == "2026-08-20"
    assert health["days_since_last_judgement"] is not None


def test_no_labels_at_all_is_not_an_alarm_either(repo):
    import pipeline.labeling as lab

    health = lab.label_file_health()

    assert health["files"] == 0
    assert health["untracked"] == []
    assert health["last_judged"] is None
    assert health["days_since_last_judgement"] is None


def test_status_stays_silent_when_everything_is_tracked(repo, monkeypatch):
    from typer.testing import CliRunner

    import pipeline.labeling as lab
    from pipeline.cli import app

    monkeypatch.setattr(lab, "label_file_health", lambda: {
        "files": 1, "untracked": [], "last_judged": "2026-08-20",
        "days_since_last_judgement": 3,
    })

    assert "[LABELS]" not in CliRunner().invoke(app, ["status"]).stdout


def test_status_names_the_file_when_one_is_untracked(repo, monkeypatch):
    from typer.testing import CliRunner

    import pipeline.labeling as lab
    from pipeline.cli import app

    monkeypatch.setattr(lab, "label_file_health", lambda: {
        "files": 2, "untracked": ["held_review.jsonl"],
        "last_judged": "2026-08-20", "days_since_last_judgement": 26,
    })

    out = CliRunner().invoke(app, ["status"]).stdout

    assert "[LABELS]" in out
    assert "held_review.jsonl" in out
