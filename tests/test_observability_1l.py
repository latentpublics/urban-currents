"""What an unattended run leaves behind (phase 1L).

Four investigations dead-ended on one fact: `runs/` lives on the CI runner and
is thrown away. These tests pin the facts that now survive it, and — more
importantly — pin the thing that must **not** change while they do.

🔴 **L0 is the load-bearing test in this file.** arXiv failing must not cost the
day its journal papers. `outcome.silent_sources`' own docstring says why:
*"Journal items collected on a day arXiv was unreachable are real papers, and
withholding them would trade a partial issue for none."* Recording the failure
is a change to what we know; it is not a change to what a reader receives.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pipeline.metrics import Run
from pipeline.outcome import PUBLISHED, decide, failed_sources, record, load_log

DAY = date(2026, 9, 13)


def _run(repo, stages: dict[str, str] | None = None) -> Run:
    run = Run.for_date(DAY)
    run.metrics.stages.update(
        stages
        or {
            "collect.arxiv": "OK",
            "collect.openalex": "OK",
            "collect": "OK",
            "summarize": "OK",
            "classify": "OK",
            "select": "OK",
            "issue": "OK",
        }
    )
    return run


# --------------------------------------------------------------------------
# L0 — the verdict does not move
# --------------------------------------------------------------------------


def test_a_failed_arxiv_day_still_publishes(repo):
    """🔴 The regression guard for L0.

    The nightly investigation proposed raising an exception so the guard would
    fire. Half of that is right — record the failure — and the other half would
    have deleted four real issues (09-07, 09-12, 09-13, 09-14), journal papers
    and all, because `looked()` refusing makes `decide()` return
    `not_published`.
    """
    run = _run(repo)
    run.observe_source("collect.arxiv", windows=1, windows_failed=1, failed_all=True)
    run.source_failure("collect.arxiv", "window 2026-09-13..2026-09-13: RuntimeError: boom")

    outcome = decide(run, DAY, published_count=3)

    assert outcome.status == PUBLISHED, "arXiv failing must not delete the day"
    assert outcome.published == 3
    # ...and the fact is on the record all the same.
    assert outcome.failed_sources == ["collect.arxiv"]


def test_the_stage_verdict_is_untouched(repo):
    """The fact sits beside the verdict and never becomes one."""
    run = _run(repo)
    run.observe_source("collect.arxiv", failed_all=True)
    run.source_failure("collect.arxiv", "RuntimeError: boom")

    assert run.metrics.stages["collect.arxiv"] == "OK"
    assert decide(run, DAY, published_count=1).status == PUBLISHED


# --------------------------------------------------------------------------
# L1 — silence and failure are different fields
# --------------------------------------------------------------------------


def test_failure_is_not_written_down_as_silence(repo):
    """The whole point of L1.

    `silent_sources` means "reported OK and the count was zero". A dead window
    also produces zero with the stage still OK, so it was being filed under a
    sentence that says the opposite of what happened.
    """
    run = _run(repo)
    run.observe_source("collect.arxiv", windows=1, windows_failed=1, failed_all=True)

    outcome = decide(run, DAY, published_count=2, window_days=1)

    assert outcome.failed_sources == ["collect.arxiv"]
    assert outcome.silent_sources == [], "a failure must not appear as silence"


def test_silence_is_not_written_down_as_failure(repo):
    """And the reverse. A source that worked and found nothing is not broken."""
    run = _run(repo)
    run.observe_source("collect.arxiv", windows=1, windows_failed=0, failed_all=False)
    run.count("arxiv_fetched", 0)
    run.count("openalex_fetched", 12)

    outcome = decide(run, DAY, published_count=2, window_days=7)

    assert outcome.failed_sources == []
    assert "collect.arxiv" in outcome.silent_sources


def test_one_bad_window_in_a_backfill_is_not_a_failed_source(repo):
    """Thin, not blind. `failed_all` is every window, not any window."""
    run = _run(repo)
    run.observe_source("collect.arxiv", windows=13, windows_failed=1, failed_all=False)

    assert failed_sources(run) == []


# --------------------------------------------------------------------------
# L2 — 200-with-nothing is distinguishable from never-answered
# --------------------------------------------------------------------------


def test_a_query_that_matched_nothing_is_readable_afterwards(repo):
    run = _run(repo)
    run.observe_source(
        "collect.arxiv", http_status=200, total_results=0, windows=1, windows_failed=0
    )

    obs = decide(run, DAY, published_count=1).source_observations["collect.arxiv"]

    assert obs["http_status"] == 200
    assert obs["total_results"] == 0
    assert obs["windows_failed"] == 0


def test_a_request_that_never_answered_is_readable_afterwards(repo):
    run = _run(repo)
    run.observe_source(
        "collect.arxiv", http_status=503, windows=1, windows_failed=1, failed_all=True
    )

    obs = decide(run, DAY, published_count=1).source_observations["collect.arxiv"]

    assert obs["http_status"] == 503
    assert obs["windows_failed"] == 1


# --------------------------------------------------------------------------
# L3 — the reason outlives the runner
# --------------------------------------------------------------------------


def test_the_failure_reason_reaches_the_committed_log(repo):
    """D261's gap. The reason used to exist only in Actions logs, for 90 days."""
    run = _run(repo)
    run.source_failure(
        "collect.arxiv", "window 2026-09-13..2026-09-13: ReadTimeout: timed out"
    )

    record(decide(run, DAY, published_count=1))
    row = load_log(DAY)

    assert "ReadTimeout" in row["source_failures"]["collect.arxiv"][0]


def test_a_credential_never_reaches_the_log(repo, monkeypatch):
    """🔴 1L would have published the Springer key.

    `abstracts.py` passes it as a query parameter and reports failures with the
    exception's own text, which for `requests` carries the URL. Harmless while
    that string died in gitignored `runs/`; L3 writes it into `content/` in a
    public repo and L4 uploads it.
    """
    monkeypatch.setenv("SPRINGER_API_KEY", "s3cr3t-key-value-12345")
    run = _run(repo)
    run.error(
        "springer 10.1/a: HTTPError: 401 for url: "
        "https://api.springernature.com/meta/v2/json?q=doi:10.1/a"
        "&api_key=s3cr3t-key-value-12345&p=1"
    )
    run.source_failure("collect.arxiv", "key=s3cr3t-key-value-12345 leaked here")

    record(decide(run, DAY, published_count=1))
    blob = (Path(str(repo)) / "content" / "runs_log" / f"{DAY}.json").read_text(
        encoding="utf-8"
    )

    assert "s3cr3t-key-value-12345" not in blob
    assert "[REDACTED]" in "".join(run.metrics.errors)


# --------------------------------------------------------------------------
# L5 — held counts reach the repository, and absent is not zero
# --------------------------------------------------------------------------


def test_the_held_counts_reach_the_committed_log(repo):
    run = _run(repo)
    run.count("held_withheld", 0)
    run.count("held_near_miss", 43)

    record(decide(run, DAY, published_count=7))
    row = load_log(DAY)

    assert row["held_withheld"] == 0
    assert row["held_near_miss"] == 43


def test_a_row_that_never_counted_is_none_not_zero(repo):
    """🔴 The measured-zero trap, in the field added to close it.

    A day that reached selection and held nothing writes 0. A day that never
    reached selection writes null. Filling the second with 0 would rebuild the
    exact ambiguity `content/held/`'s missing files had (1J §3-1).
    """
    run = _run(repo)

    outcome = decide(run, DAY, published_count=0)

    assert outcome.held_withheld is None
    assert outcome.held_near_miss is None

    record(outcome)
    assert load_log(DAY)["held_withheld"] is None


def test_zero_held_and_unknown_held_are_different_rows(repo):
    counted = _run(repo)
    counted.count("held_withheld", 0)
    uncounted = Run.for_date(date(2026, 9, 12))
    uncounted.metrics.stages.update(counted.metrics.stages)

    assert decide(counted, DAY, 1).held_withheld == 0
    assert decide(uncounted, date(2026, 9, 12), 1).held_withheld is None


def test_the_counts_are_the_same_on_a_second_run(repo):
    """Idempotency: the same day twice is the same numbers."""
    first = _run(repo)
    first.count("held_withheld", 1)
    first.count("held_near_miss", 37)
    a = decide(first, DAY, published_count=5)

    second = _run(repo)
    second.count("held_withheld", 1)
    second.count("held_near_miss", 37)
    b = decide(second, DAY, published_count=5)

    assert (a.held_withheld, a.held_near_miss) == (b.held_withheld, b.held_near_miss)


# --------------------------------------------------------------------------
# L6 — the commit message carries the slot date
# --------------------------------------------------------------------------


def _daily_yml() -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / ".github" / "workflows" / "daily.yml").read_text(encoding="utf-8")


def test_the_commit_message_uses_the_slot_date():
    """`32e9d69` is the specimen: the 09-14 run committed as "2026-09-15 run".

    This file warns about the midnight trap in four other places. The step that
    says *what was done* was the one that still used the wall clock.
    """
    step = _daily_yml().split("- name: Commit content", 1)[1]
    step = step.split("- name:", 1)[0]

    assert 'git commit -m "content: ${day} run"' in step
    assert "SLOT_DATE: ${{ steps.slot.outputs.date }}" in step
    assert 'day="${SLOT_DATE:-$(date -u +%Y-%m-%d)}"' in step
    assert 'git commit -m "content: $(date -u +%Y-%m-%d) run"' not in step


# --------------------------------------------------------------------------
# L4 — the run directory is uploaded, without the parts that could carry one
# --------------------------------------------------------------------------


def test_the_run_directory_is_kept_as_an_artifact():
    text = _daily_yml()
    step = text.split("- name: Keep the run directory", 1)[1].split("- name:", 1)[0]

    assert "actions/upload-artifact@v4" in step
    assert "runs/run_*/metrics.json" in step
    assert "runs/run_*/stages/" in step
    # The runs worth reading are the ones that failed.
    assert "always()" in step
    # Finite, per L4.
    assert "retention-days: 90" in step


def test_the_artifact_excludes_raw_and_rendered_mail():
    """raw/ could not be cleared for CONTACT_EMAIL or SPRINGER_API_KEY — neither
    is set locally, so the audit covering 71 runs could not see them. Nothing in
    L1-L5 needs raw/, so it stays out rather than being argued about."""
    step = _daily_yml().split("- name: Keep the run directory", 1)[1].split("- name:", 1)[0]
    paths = [line.strip() for line in step.splitlines() if line.strip().startswith("runs/")]

    assert paths, "the step must list explicit paths, not the whole run dir"
    assert not any("raw" in p for p in paths)
    assert not any(p.endswith(".html") for p in paths)


@pytest.mark.parametrize("field", ["failed_sources", "source_failures", "held_withheld"])
def test_the_new_fields_are_in_every_row(repo, field):
    run = _run(repo)
    record(decide(run, DAY, published_count=1))

    assert field in load_log(DAY)
