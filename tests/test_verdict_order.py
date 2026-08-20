"""When the verdict is reached, and what it is allowed to know (phase 0V).

0U made four stages required before a day may be published, and one of them —
`issue` — is written **after** the verdict is reached. `uc daily` therefore
refused every day, on every date, for the reason that the artefact it was about
to produce did not exist yet. 661 tests and a 9/9 verification were green while
it did so, because no test ran `run_daily` with its real stage order: the
fixtures stamped every required stage OK inside the collect stub.

So these tests are about **order**, not about any one stage:

  * a healthy day, run through the actual sequence, ends with an issue on disk;
  * nothing is required of the verdict that has not happened when it is reached;
  * the count the verdict uses is the count the issue actually published.

The last one is U5's mirror. `decide()` was given the size of the *selection*,
while the issue publishes the selection minus whatever appeared on an earlier
date — so on a day with nothing new, the row said `published` over an issue of
zero items. The old code got this right by accident, through the `quiet_day`
overwrite that U5 correctly removed.

No network and no keys: the collector is the built-in fixture and the model is
a stub.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from pipeline import daily as daily_mod
from pipeline import paths, run_stages, store
from pipeline.llm import LLMResponse
from pipeline.outcome import PUBLISHED, QUIET, load_log

SUMMARY = {
    "what": "The authors fit a gradient-boosted model to 41,000 trips and report "
            "a 12% reduction in predicted travel time error.",
    "why": "Travel-time prediction is the input to every accessibility measure "
           "downstream, and the error term is rarely reported at all.",
    "caveats": None,
    "geographic_scope": "single_city",
    "data_available": False,
    "methods": ["gradient boosting"],
    "data": ["travel survey"],
    "tools": [],
    "places": ["Seoul"],
}


@pytest.fixture
def working_day(monkeypatch):
    """A `uc daily` that can succeed: fixture collector, stubbed model.

    Only the two edges of the pipeline are replaced — where the items come from
    and what the model says. Everything between them, including the order the
    stages run in and where the verdict is reached, is the real thing, because
    that order is what is under test.
    """

    # Bound before the patch. Reading `run_stages.stage_collect` inside the
    # stub would read the stub — 0U wrote it that way and the resulting
    # RecursionError surfaced as `collect: FAILED`, which is exactly what
    # those tests were asserting, so they passed while collecting nothing.
    real_collect = run_stages.stage_collect

    def collect(run, d, backfill_from=None, **kw):
        items = real_collect(run, d, fixture=True)
        # The fixture stands in for the two collectors, so it reports what they
        # report. `looked()` requires both sources by name, and a stub that
        # skips saying so is a stub that can never produce a publishable day.
        run.metrics.stages["collect.arxiv"] = "OK"
        run.metrics.stages["collect.openalex"] = "OK"
        run.count("openalex_fetched", 0)
        return items

    def caller(system: str, user: str) -> LLMResponse:
        return LLMResponse(text=json.dumps(SUMMARY), input_tokens=900,
                           output_tokens=200, model="gemini-3.5-flash")

    from pipeline.summarize import run as summarize_run

    real_client = summarize_run.LLMClient
    monkeypatch.setattr(run_stages, "stage_collect", collect)
    monkeypatch.setattr(
        summarize_run, "LLMClient",
        lambda *a, **kw: real_client(*a, caller=caller, **kw),
    )
    monkeypatch.setenv("UC_ALERT_RECIPIENT", "yjun@example.org")
    monkeypatch.setenv("UC_PREVIEW_RECIPIENT", "reader@example.org")


def _issue_file(d: date):
    return paths.CONTENT / "issues" / f"{d}.json"


# --------------------------------------------------------------------------
# The regression itself
# --------------------------------------------------------------------------


def test_a_healthy_day_writes_an_issue(repo, sample_date, working_day):
    """The one assertion 0U had nowhere: a good day publishes.

    Before the fix this fails with `not_published` and the reason
    "issue did not run" — the verdict refusing the day because the stage it
    was about to run had not run yet.
    """
    result = daily_mod.run_daily(d=sample_date, use_llm=True)

    assert result["status"] == PUBLISHED, result["reasons"]
    assert _issue_file(sample_date).exists()

    issue = store.load_issue(sample_date)
    assert issue is not None
    assert len(issue.items) == result["published"] > 0

    log = load_log(sample_date)
    assert log["status"] == PUBLISHED
    assert log["published"] == len(issue.items)


def test_the_verdict_never_asks_about_a_stage_that_has_not_run(
    repo, sample_date, working_day, monkeypatch
):
    """The general form, so the next stage added to the list cannot repeat it.

    Whatever `decide()` is asked to require at the moment it is called, the run
    must already have a status for. This is the invariant `issue` broke, and it
    catches it without naming any particular stage.
    """
    seen: list[tuple[tuple[str, ...], dict]] = []

    from pipeline import outcome as outcome_mod

    real_decide = outcome_mod.decide

    def spy(run, d, published_count, **kw):
        required = kw.get("stages") or outcome_mod.REQUIRED_STAGES
        seen.append((tuple(required), dict(run.metrics.stages)))
        return real_decide(run, d, published_count, **kw)

    monkeypatch.setattr(outcome_mod, "decide", spy)
    monkeypatch.setattr(daily_mod, "decide", spy)

    daily_mod.run_daily(d=sample_date, use_llm=True)

    assert seen, "decide() was never called"
    for required, stages in seen:
        missing = [s for s in required if s not in stages]
        assert not missing, (
            f"decide() required {missing} at a point in the run where "
            f"they had not run; stages so far: {sorted(stages)}"
        )


def test_the_issue_stage_is_asked_about_after_it_has_run(repo, sample_date, working_day):
    """`issue` is still required — the fix is when it is asked, not whether."""
    from pipeline.outcome import PRE_ISSUE_STAGES, REQUIRED_STAGES

    assert "issue" in REQUIRED_STAGES
    assert "issue" not in PRE_ISSUE_STAGES
    assert set(PRE_ISSUE_STAGES) < set(REQUIRED_STAGES)

    result = daily_mod.run_daily(d=sample_date, use_llm=True)

    assert result["status"] == PUBLISHED
    # Read back from the run's own metrics: `issue` has a status by the time
    # the day is called published, which is the whole content of the fix.
    from pipeline.metrics import Run

    assert Run.for_date(sample_date).metrics.stages["issue"] == "OK"


def test_a_failing_issue_stage_leaves_a_row_and_no_publication(
    repo, sample_date, working_day, monkeypatch
):
    """The other direction: the artefact fails and the day is not called good.

    `stage_issue` used to sit outside `_guard`, so an exception in it escaped
    `run_daily` entirely — no verdict, no run-log row, no alert. A day with no
    row is indistinguishable from a day the schedule never fired, which is the
    silence this whole phase exists to end.
    """
    def explode(run, d, **kw):
        raise RuntimeError("the renderer raised on a malformed headline")

    monkeypatch.setattr(run_stages, "stage_issue", explode)
    monkeypatch.setattr(daily_mod, "stage_issue", explode, raising=False)

    result = daily_mod.run_daily(d=sample_date, use_llm=True)

    assert result["status"] == "not_published"
    assert not _issue_file(sample_date).exists()
    assert "issue" in result["failed_stages"]

    log = load_log(sample_date)
    assert log["status"] == "not_published"
    assert any("issue" in r for r in log["reasons"])


# --------------------------------------------------------------------------
# V3 — the count that decides is the count that was published
# --------------------------------------------------------------------------


def test_a_day_with_nothing_new_is_quiet_not_published(
    repo, sample_date, working_day
):
    """U5's mirror image.

    The window is seven days wide and moves by one, so most of what a run
    selects has already appeared. `decide()` was handed the size of the
    selection; the issue publishes the selection minus the published index. Run
    the same day twice and the second issue is empty — and the row said
    `published` over nothing.
    """
    first = daily_mod.run_daily(d=sample_date, use_llm=True)
    assert first["status"] == PUBLISHED and first["published"] > 0

    second_day = date(sample_date.year, sample_date.month, sample_date.day + 1)
    second = daily_mod.run_daily(d=second_day, use_llm=True)

    issue = store.load_issue(second_day)
    assert issue is not None
    assert issue.items == [], "the fixture should have nothing new on day two"

    assert second["published"] == 0
    assert second["status"] == QUIET, (
        "a day that published nothing is quiet — that is the definition U5 set"
    )
    assert load_log(second_day)["status"] == QUIET
