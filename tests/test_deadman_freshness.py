"""The deadman's freshness arithmetic (1E, B).

On 2026-09-07 the deadman mailed *"No run has been recorded since 2026-09-06
(38h ago)"* about a pipeline that had finished normally fifteen hours earlier
and would run again seven hours later. Nothing was broken except the
measurement and the margin:

* it measured age from the newest row's **date at midnight UTC**, and a run
  that covers a day lands 22:50–23:26 UTC *on* that day, so a perfectly healthy
  archive read 33 hours old at the 09:00 slot;
* the limit was 36, leaving three hours of slack for a watchdog whose own cron
  is routinely hours late. This repository's `daily` arrives at +2:19 every
  day. The deadman was about five hours late that morning.

**A watchdog less punctual than the thing it watches is not a watchdog.**

These tests exist because that arithmetic lived inside a YAML string and could
not be run with a fabricated clock. It is a script now, and `NOW_EPOCH` makes
every scenario below a replay rather than an argument.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / ".github/scripts/deadman-freshness.sh"
WORKFLOW = ROOT / ".github/workflows/deadman.yml"


def _find_bash() -> str | None:
    """A bash that actually runs.

    On Windows `shutil.which("bash")` finds `System32\\bash.exe` — the WSL
    launcher — before Git's, and without a distribution installed it fails with
    a UTF-16 RPC error rather than an exception. So candidates are *tried*, not
    looked up.
    """
    candidates = [
        p for p in (
            shutil.which("bash"),
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            "/usr/bin/bash",
            "/bin/bash",
        ) if p
    ]
    for path in candidates:
        if not Path(path).exists():
            continue
        try:
            probe = subprocess.run(
                [path, "-c", "echo ok"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=30,
            )
        except OSError:
            continue
        if probe.returncode == 0 and probe.stdout.strip() == "ok":
            return path
    return None


BASH = _find_bash()

pytestmark = pytest.mark.skipif(
    BASH is None, reason="the freshness check is a shell script and needs bash"
)


def _max_age() -> int:
    """The limit the workflow actually ships. Never a number copied in here."""
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return int(doc["jobs"]["deadman"]["env"]["MAX_AGE_HOURS"])


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _row(log_dir: Path, day: str, recorded_at: str | None) -> None:
    body = {"date": day, "status": "published", "published": 3}
    if recorded_at is not None:
        body["recorded_at"] = recorded_at
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{day}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run(log_dir: Path, now: datetime, max_age: int | None = None):
    return subprocess.run(
        [BASH, str(SCRIPT)],
        capture_output=True,
        text=True,
        # Explicit, because the default is the console codepage and this
        # machine's is cp949. The script prints em dashes.
        encoding="utf-8",
        errors="replace",
        env={
            "PATH": __import__("os").environ.get("PATH", ""),
            "RUNS_LOG_DIR": str(log_dir),
            "MAX_AGE_HOURS": str(max_age if max_age is not None else _max_age()),
            "NOW_EPOCH": str(int(now.timestamp())),
        },
    )


# --------------------------------------------------------------------------
# The morning it cried wolf
# --------------------------------------------------------------------------


def test_the_2026_09_07_false_alarm_does_not_fire(tmp_path):
    """The exact readings from that morning, replayed.

    The 09-06 run recorded at 22:54:40 UTC. The deadman fired at about 14:00 on
    09-07 — roughly five hours behind its 09:00 slot — and the old measurement
    made that 38 hours. The pipeline was healthy: the 09-07 run went out
    normally seven hours later.
    """
    _row(tmp_path, "2026-09-06", "2026-09-06T22:54:40+00:00")

    result = _run(tmp_path, _utc("2026-09-07T14:00:00+00:00"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "15h old" in result.stdout


def test_it_would_have_fired_under_the_old_measurement(tmp_path):
    """The counter-check, so this file proves the fix rather than asserting it.

    Same row, same clock, the old rule: midnight on the row's own date, limit
    36. 38 hours, red. If this ever stops being true the scenario above has
    stopped being the one that broke.
    """
    row_midnight = _utc("2026-09-06T00:00:00+00:00")
    alarm_time = _utc("2026-09-07T14:00:00+00:00")

    old_age = (alarm_time - row_midnight).total_seconds() / 3600
    assert round(old_age) == 38
    assert old_age > 36


# --------------------------------------------------------------------------
# The healthy morning, and how late the watchdog may be
# --------------------------------------------------------------------------


@pytest.mark.parametrize("recorded_at", [
    "2026-09-07T22:48:54+00:00",   # earliest observed
    "2026-09-07T23:26:57+00:00",   # latest observed
])
def test_an_ordinary_morning_is_quiet(tmp_path, recorded_at):
    _row(tmp_path, "2026-09-07", recorded_at)
    result = _run(tmp_path, _utc("2026-09-08T09:00:00+00:00"))
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_watchdog_may_now_be_most_of_a_day_late(tmp_path):
    """What the change actually bought, stated as a number.

    Slack was three hours. It is about twenty: the row lands at ~23:00 and the
    limit is `MAX_AGE_HOURS` from then, so this job can be very late indeed and
    still not accuse a working pipeline.
    """
    _row(tmp_path, "2026-09-07", "2026-09-07T23:00:00+00:00")
    slot = _utc("2026-09-08T09:00:00+00:00")

    late = _run(tmp_path, slot + timedelta(hours=15))
    assert late.returncode == 0, late.stdout + late.stderr

    # And it is a limit, not an absence of one.
    much_later = _run(tmp_path, slot + timedelta(hours=30))
    assert much_later.returncode == 1


# --------------------------------------------------------------------------
# A missed day still gets caught, at the same slot as before
# --------------------------------------------------------------------------


def test_a_missed_slot_is_caught_at_the_very_next_check(tmp_path):
    """The thing that must not have been traded away.

    The 21:00 run on 09-08 produces nothing. The 09:00 check on 09-09 sees a
    row written at 23:00 on 09-07 — 34 hours — and goes red. The old rule went
    red at the same check, from a different number (57h against 36). Detection
    is unchanged; only the false-alarm margin moved.
    """
    _row(tmp_path, "2026-09-07", "2026-09-07T23:00:00+00:00")

    healthy = _run(tmp_path, _utc("2026-09-08T09:00:00+00:00"))
    assert healthy.returncode == 0, healthy.stdout + healthy.stderr

    after_a_missed_slot = _run(tmp_path, _utc("2026-09-09T09:00:00+00:00"))
    assert after_a_missed_slot.returncode == 1
    assert "::error::" in after_a_missed_slot.stdout


def test_the_real_missed_day_of_2026_08_21_still_fires(tmp_path):
    """Replayed from the archive: 08-20's run did not record until 08-21 21:42,
    so the 09:00 check that morning was looking at 08-19's row from 21:39.
    35 hours. That alarm was correct and has to stay correct."""
    _row(tmp_path, "2026-08-19", "2026-08-19T21:39:25+00:00")

    result = _run(tmp_path, _utc("2026-08-21T09:00:00+00:00"))

    assert result.returncode == 1
    assert "35h old" in result.stdout


def test_the_limit_sits_between_the_two_populations():
    """The limit is not a taste. Healthy mornings and a missed slot are two
    separated clouds of numbers, and it has to be between them.

    Healthy: 12h from the daily slot to this one, minus the daily's own
    lateness (1.8–2.5h), plus this job's. A missed slot adds 24.
    """
    worst_healthy = 12 - 1.8 + 6      # this job six hours late
    best_detection = 12 - 2.5 + 24    # this job on time, daily as late as ever

    limit = _max_age()
    assert worst_healthy < limit < best_detection, (worst_healthy, best_detection)


# --------------------------------------------------------------------------
# The fallback, and the case the whole job exists for
# --------------------------------------------------------------------------


def test_a_row_without_recorded_at_falls_back_and_says_so(tmp_path):
    """Every row in the archive has `recorded_at` and `Outcome.as_dict()`
    always writes one. If that ever stops being true the measurement degrades
    to the old midnight rule — which over-reads the age, so it errs toward
    alarming — and it has to be **audible**. A fallback nobody can see is how
    you stop finding out what made an alarm fire."""
    _row(tmp_path, "2026-09-07", None)

    result = _run(tmp_path, _utc("2026-09-08T09:00:00+00:00"))

    assert "::warning::" in result.stdout
    assert "recorded_at" in result.stdout
    assert "33h old" in result.stdout
    assert result.returncode == 1, "33h from midnight is over the 30h limit"


def test_an_empty_archive_is_an_error_not_a_fresh_one(tmp_path):
    result = _run(tmp_path / "nothing", _utc("2026-09-08T09:00:00+00:00"))
    assert result.returncode == 1
    assert "never recorded a run" in result.stdout


# --------------------------------------------------------------------------
# The workflow and the script have to stay attached
# --------------------------------------------------------------------------


def test_the_workflow_runs_this_script():
    """Extracting the arithmetic is only worth it while the workflow uses it."""
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = doc["jobs"]["deadman"]["steps"]
    runs = " ".join(s.get("run", "") for s in steps)
    assert ".github/scripts/deadman-freshness.sh" in runs


def test_the_gap_check_still_runs_too():
    """Question 2 is a different question and neither replaces the other.

    Freshness asks whether the archive is still being written to; the gap check
    asks whether a day inside the horizon has no row at all. 2026-08-26 is the
    proof that the first cannot answer the second: its newest row was *newer*
    than it should have been.
    """
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    runs = " ".join(s.get("run", "") for s in doc["jobs"]["deadman"]["steps"])
    assert "uc missing-days" in runs
