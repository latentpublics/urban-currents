"""``uc`` — the Phase 0 command line.

Every pipeline stage is a separate command taking ``--date``, because re-running
summarize must not require re-collecting (PRD §5). ``uc run`` chains them.
"""

from __future__ import annotations

import json
import sys
import webbrowser
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer

# Windows consoles default to a legacy code page (cp949 here), which turns an
# em dash in a status line into a crash after the work is already done.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - stream may not support it
            pass

from . import paths, run_stages, store  # noqa: E402  - after the reconfigure above
from .metrics import Run  # noqa: E402
from .models import PIPELINE_VERSION  # noqa: E402

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Urban Currents — a daily scan of urban data science research (Phase 0).",
)

DateOpt = typer.Option(None, "--date", "-d", help="Issue date, YYYY-MM-DD (default: today)")


def _date(value: Optional[str]) -> date:
    return date.fromisoformat(value) if value else date.today()


def _run(value: Optional[str]) -> Run:
    return Run.for_date(_date(value))


def _echo_stage(run: Run, name: str, n: Optional[int] = None) -> None:
    status = run.metrics.stages.get(name, "OK")
    suffix = f" — {n} items" if n is not None else ""
    typer.echo(f"[{status}] {name}{suffix}")


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------


@app.command()
def collect(
    date_: Optional[str] = DateOpt,
    source: str = typer.Option("all", help="all | arxiv | openalex"),
    backfill: int = typer.Option(0, help="Also collect the N days before --date"),
    fixture: bool = typer.Option(False, help="Use built-in sample papers, no network"),
):
    """Collect candidates from arXiv and OpenAlex. Raw responses are preserved."""
    d = _date(date_)
    run = Run.for_date(d)
    start = d - timedelta(days=backfill) if backfill else None
    items = run_stages.stage_collect(run, d, sources=source, fixture=fixture, backfill_from=start)
    _echo_stage(run, "collect", len(items))


@app.command()
def dedup(date_: Optional[str] = DateOpt):
    """Merge duplicate records (arXiv preprint vs OpenAlex journal Work)."""
    d = _date(date_)
    run = Run.for_date(d)
    items = run_stages.stage_dedup(run, d)
    _echo_stage(run, "dedup", len(items))


@app.command()
def gate(date_: Optional[str] = DateOpt):
    """Apply the volume gate to the high-volume arXiv categories."""
    run = _run(date_)
    items = run_stages.stage_gate(run)
    _echo_stage(run, "gate", len(items))


@app.command()
def enrich(
    date_: Optional[str] = DateOpt,
    source: str = typer.Option("all", help="all | crossref | springer | none"),
):
    """Recover abstracts publishers withdrew from OpenAlex.

    Springer Nature still serves, from its own free Metadata API, the abstracts
    it removed from OpenAlex in 2022 — roughly a quarter of the journal path.
    It needs `SPRINGER_API_KEY` in `.env` (register at dev.springernature.com);
    without it that half is recorded SKIPPED and the run continues.
    """
    run = _run(date_)
    picked = {
        "all": ("crossref", "springer"),
        "crossref": ("crossref",),
        "springer": ("springer",),
        "none": (),
    }.get(source)
    if picked is None:
        raise typer.BadParameter("source must be all, crossref, springer or none")
    items = run_stages.stage_enrich(run, sources=picked)
    counts = {
        k: v for k, v in run.metrics.counts.model_dump().items()
        if k.startswith("abstract_")
    }
    _echo_stage(run, "enrich", len(items))
    typer.echo(json.dumps(counts))


@app.command()
def classify(date_: Optional[str] = DateOpt):
    """Score relevance with the trained classifier (or the labelled fallback)."""
    run = _run(date_)
    items = run_stages.stage_classify(run)
    typer.echo(
        f"[{run.metrics.stages.get('classify')}] classify — {len(items)} items "
        f"({run.metrics.stages.get('classify.model')})"
    )


@app.command()
def select(
    date_: Optional[str] = DateOpt,
    threshold: Optional[float] = typer.Option(None),
    top: Optional[int] = typer.Option(None, "--top"),
):
    """Keep items above the relevance threshold, capped at the daily top-N."""
    run = _run(date_)
    items = run_stages.stage_select(run, threshold=threshold, top_n=top)
    _echo_stage(run, "select", len(items))


@app.command()
def link(
    date_: Optional[str] = DateOpt,
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM overlay candidates"),
):
    """Link entities: OpenAlex passthrough plus vocabulary-matched overlay tags."""
    run = _run(date_)
    items = run_stages.stage_link(run, use_llm=not no_llm)
    _echo_stage(run, "link", len(items))


@app.command()
def summarize(
    date_: Optional[str] = DateOpt,
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the API call entirely"),
    limit: Optional[int] = typer.Option(None, help="Cap the number of items summarised"),
):
    """Generate the two-layer summary for each selected item."""
    run = _run(date_)
    items = run_stages.stage_summarize(run, use_llm=not no_llm, limit=limit)
    _echo_stage(run, "summarize", len(items))


@app.command()
def score(date_: Optional[str] = DateOpt):
    """Compute headline scores."""
    run = _run(date_)
    items = run_stages.stage_score(run)
    _echo_stage(run, "score", len(items))


@app.command()
def issue(date_: Optional[str] = DateOpt):
    """Publish Items and the daily Issue into content/."""
    d = _date(date_)
    run = Run.for_date(d)
    iss = run_stages.stage_issue(run, d)
    typer.echo(
        f"[{run.metrics.stages.get('issue')}] issue — {len(iss.items)} items, "
        f"{'quiet day' if iss.is_quiet else 'headline: ' + (iss.headline.line or 'none cleared the bar')}"
    )


@app.command()
def preview(
    date_: Optional[str] = DateOpt,
    open_: bool = typer.Option(False, "--open", help="Open the result in a browser"),
):
    """Render the single-file HTML preview for a date."""
    d = _date(date_)
    run = Run.for_date(d)
    out = run_stages.stage_preview(run, d)
    typer.echo(f"[OK] preview — {out}")
    if open_ and out:
        webbrowser.open(Path(out).resolve().as_uri())


@app.command()
def run(
    date_: Optional[str] = DateOpt,
    source: str = typer.Option("all", help="all | arxiv | openalex"),
    fixture: bool = typer.Option(False, help="Use built-in sample papers, no network"),
    no_llm: bool = typer.Option(False, "--no-llm"),
    limit: Optional[int] = typer.Option(None, help="Cap items summarised this run"),
):
    """Run every stage for one date."""
    d = _date(date_)
    r = run_stages.run_all(
        d, sources=source, fixture=fixture, use_llm=not no_llm, summarize_limit=limit
    )
    from .stages import STAGE_ORDER

    for stage_name in STAGE_ORDER:
        typer.echo(f"[{r.metrics.stages.get(stage_name, '-')}] {stage_name}")
    typer.echo(f"metrics: {r.metrics_path}")


# --------------------------------------------------------------------------
# Review, report, and maintenance
# --------------------------------------------------------------------------


@app.command()
def review(
    date_: Optional[str] = DateOpt,
    label: Optional[str] = typer.Option(None, help="Fast labelling mode, e.g. --label relevance"),
    pending: bool = typer.Option(
        False, "--pending", help="Judge what the pipeline held while you were away"
    ),
    sample: bool = typer.Option(
        False, "--sample", help="Read a sample of what was published (use with --since)"
    ),
    since: Optional[str] = typer.Option(
        None, "--since", help="--sample: read issues published on or after this date"
    ),
    relabel: Optional[str] = typer.Option(
        None, "--relabel", help="Re-judge existing labels, e.g. --relabel weak"
    ),
    top: int = typer.Option(30, help="Items to label per day (split evenly by source)"),
    limit: int = typer.Option(30, help="--label affinity: probe size, split evenly by band"),
    days: int = typer.Option(7, help="--label affinity: days of candidates to draw from"),
):
    """Review the issue, the held queue, or a sample of what already went out.

    **There is deliberately no `--latest`.** An argument-free "show me today"
    would be a standing invitation to check every morning, and the pipeline is
    supposed to run without that. `--pending` is the command for coming back:
    it asks for no date because a week away leaves a week of held items and
    remembering which dates those were is the friction being removed.

    - ``--pending``            judge the held queue, oldest first, resumable
    - ``--sample --since D``   read a stratified sample of published cards
    - ``--label relevance``    the Q1b labelling pass on a day's candidates
    - ``--label affinity``     the affinity probe, written to a **separate**
      file; the two are different experiments and are never pooled
    - ``--relabel weak``       split the pre-M1 ``drop_weak`` rows into method
      and results, appending corrections rather than editing
    - ``--date D``             the full review of one issue
    """
    from .review import (
        run_code_probe_session,
        run_labeling_session,
        run_pending_session,
        run_probe_session,
        run_rejudge_session,
        run_review_session,
        run_sample_session,
        run_subfield_check_session,
    )

    if pending:
        run_pending_session()
        return
    if sample:
        if not since:
            typer.echo("--sample needs --since YYYY-MM-DD")
            raise typer.Exit(code=2)
        run_sample_session(_date(since))
        return
    if relabel:
        if relabel not in ("weak", "drop_weak"):
            typer.echo(f"unknown --relabel {relabel!r}; the only mode is `weak`")
            raise typer.Exit(code=2)
        run_rejudge_session()
        return

    d = _date(date_)
    if label in ("affinity", "affinity_probe"):
        dates = [d - timedelta(days=i) for i in range(days - 1, -1, -1)]
        run_probe_session(dates, per_band=max(1, limit // 3))
    elif label in ("code_probe", "code"):
        run_code_probe_session()
    elif label in ("subfield_check", "subfield"):
        run_subfield_check_session()
    elif label in ("relevance",):
        run_labeling_session(d, facet=label, top=top)
    elif label:
        # No fall-through. `--label code_probe` used to land in the relevance
        # session, which built `ranked_top_n` rows that the write guard then
        # refused — after all thirty questions had been asked. An unknown label
        # is refused here, before anything is shown, and the message says what
        # the known ones are.
        typer.echo(f"unknown --label {label!r}. Known label sets:")
        typer.echo("  relevance   the Q1b ranked sample")
        typer.echo("  affinity    the canon-affinity probe")
        typer.echo("  code_probe  code-bearing arXiv papers")
        typer.echo("  subfield_check  the four subfields the scope gate excludes")
        raise typer.Exit(code=2)
    else:
        run_review_session(d)


@app.command()
def daily(
    date_: Optional[str] = DateOpt,
    dry_run: bool = typer.Option(False, "--dry-run", help="Run everything, write no issue"),
    no_llm: bool = typer.Option(False, "--no-llm"),
    smoke: bool = typer.Option(
        False, "--smoke", help="Narrow window, few summaries — checks the install, not the day"
    ),
):
    """Run one day end to end: collect, classify, summarise, publish, render.

    The command a scheduler calls. It picks its own window, refuses to run
    twice at once, resumes what a previous attempt finished, and — the part
    that matters — writes no issue on a day it could not see.
    """
    from .daily import DailyLocked, run_daily

    try:
        result = run_daily(
            d=_date(date_) if date_ else None,
            dry_run=dry_run,
            use_llm=not no_llm,
            smoke=smoke,
        )
    except DailyLocked as e:
        typer.echo(f"[LOCKED] {e}")
        raise typer.Exit(code=75)  # EX_TEMPFAIL

    typer.echo(json.dumps(result, indent=2))
    if result["status"] == "not_published":
        raise typer.Exit(code=1)


@app.command("backfill-issues")
def backfill_issues_cmd(
    days: int = typer.Option(60, help="How many past days to fill"),
    budget: Optional[float] = typer.Option(None, help="Spend ceiling in USD"),
    commit: bool = typer.Option(True, help="git commit every `backfill.commit_every` days"),
):
    """Make the missing issues for past days, oldest first.

    The archive is five days long, which is why nearly every second-order
    measurement in the last four batches ended in "too few". Candidates for
    ninety days are already on disk; this turns them into issues.

    One-day windows, `backfilled: true`, existing issues never rewritten, a
    checkpoint after every day and a commit every ten. Stops at the spend
    ceiling and says how far it got.
    """
    import subprocess

    from .backfill_issues import backfill

    def commit_block(attempted: int, state: dict) -> None:
        if not commit:
            return
        subprocess.run(["git", "add", "content/"], cwd=str(paths.ROOT))
        subprocess.run(
            [
                "git", "commit", "-q", "-m",
                f"content: backfill {attempted} day(s), "
                f"${state.get('spend_usd', 0):.4f} spent",
            ],
            cwd=str(paths.ROOT),
        )
        typer.echo(f"[COMMIT] {attempted} days, ${state.get('spend_usd', 0):.4f}")

    result = backfill(days=days, budget_usd=budget, on_checkpoint=commit_block)
    payload = result.as_dict()
    out = paths.RUNS / "backfill_issues.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + chr(10),
        encoding="utf-8",
        newline=chr(10),
    )
    typer.echo(json.dumps({k: v for k, v in payload.items() if k != "days"}, indent=2))
    typer.echo(f"-> {out}")
    if result.stopped_on:
        typer.echo(f"[STOPPED] {result.stopped_on}")


@app.command("record-interrupted")
def record_interrupted_cmd(
    date_: Optional[str] = DateOpt,
    reason: str = typer.Option(
        "the job ended without a verdict", help="What to record as the cause"
    ),
):
    """Write a run-log row for a day whose run was killed before it concluded.

    The last net. `uc daily` records its own verdict when it is given the chance
    — it watches a wall-clock budget and handles SIGTERM — but a SIGKILL or a
    vanished runner leaves nothing at all, and a day with no row looks exactly
    like a day the schedule never fired on.

    Refuses to overwrite a real verdict: if the pipeline managed to conclude
    anything, that conclusion is better than this one.
    """
    from .outcome import load_log, record_interrupted

    d = _date(date_)
    existing = load_log(d)
    if existing:
        typer.echo(f"[OK] {d} already has a verdict: {existing.get('status')}")
        return

    path = record_interrupted(d, reason)
    typer.echo(f"[RECORDED] {d} interrupted — {path}")


@app.command("catch-up")
def catch_up_cmd(
    limit: Optional[int] = typer.Option(None, help="Retry at most this many days"),
):
    """Retry the days the pipeline could not see, oldest first.

    Bounded by `daily.catch_up_days`. A day past that horizon stays missed and
    stays in the log saying so — asking again cannot recover a window whose
    sources have moved on.
    """
    from .daily import DailyLocked, catch_up

    try:
        results = catch_up(limit=limit)
    except DailyLocked as e:
        typer.echo(f"[LOCKED] {e}")
        raise typer.Exit(code=75)

    if not results:
        typer.echo("[OK] nothing to catch up on")
        return
    for row in results:
        typer.echo(f"[{row['status'].upper()}] {row['date']}")
    typer.echo(json.dumps(results, indent=2))


@app.command()
def status():
    """Where things stand. The first command to run after being away.

    Answers "is anything wrong" before "what happened": the last successful run,
    the dates with no issue, what has been spent, and the window the next run
    will cover.
    """
    from .notify import status as read_status

    state = read_status()
    typer.echo(json.dumps(state, indent=2))

    stalled = state.get("interrupted_dates") or []
    if stalled:
        typer.echo(
            f"\n[INTERRUPTED] {len(stalled)} day(s) were killed before reaching "
            f"a verdict: {', '.join(stalled[:5])}"
        )
        typer.echo(
            "              Not a failure — the run did not fit the time it was "
            "given.\n"
            "              Check daily.max_minutes against the workflow timeout."
        )

    held = state.get("held") or {}
    waiting = held.get("waiting") or 0
    if waiting:
        typer.echo(
            f"\n[WAITING] {waiting} held item(s) need a judgement — uc review --pending"
        )
        # Which rule is doing the holding, not just how much is held. As of 0Q
        # the subfield deny-list is empty, so one rule can quietly own the whole
        # withheld queue — and a rule that is the only thing withholding
        # anything deserves to be looked at, not averaged into a total.
        inert = set(held.get("inert_rules") or [])
        for rule, bucket in (held.get("by_rule") or {}).items():
            typer.echo(
                f"            {rule:<16} withheld {bucket['withheld']:>4}   "
                f"near miss {bucket['near_miss']:>4}"
                + ("   (inert — cannot hold anything new)" if rule in inert else "")
            )
        alone = held.get("withheld_by_one_rule")
        if alone:
            typer.echo(
                f"\n[ONE RULE] every withheld item comes from {alone!r}. "
                f"The withheld queue is that rule, and nothing else."
            )

    # ★ Two things the daily workflow greps for, and one a person needs (0U).
    #
    # `daily.yml`'s dry-run guard has been dead since H3: it looks in
    # `uc status` for `SKIPPED|FAILED` and `uc status` never printed stage
    # statuses, so **neither branch was reachable**. Printing them here revives
    # it, and the alerting line gives the new U2 guard something to read.
    alerting = state.get("alerting") or {}
    if alerting:
        reach = "reach a person" if alerting.get("reaches_a_person") else "reach nobody"
        typer.echo(
            f"\n[ALERTS] backend {alerting.get('backend')!r} — "
            f"alerts {reach} "
            f"({alerting.get('alert_recipients', 0)} recipient(s) configured)"
        )
        if not alerting.get("reaches_a_person"):
            typer.echo(
                "         A failing day would be silent except for the GitHub "
                "workflow run going red. Choosing a provider is a separate "
                "decision — see docs/OPERATIONS.md."
            )
        skipped = alerting.get("last_run_skipped")
        failed = alerting.get("last_run_failed") or []
        if skipped or failed:
            typer.echo(
                f"         last run {alerting.get('last_run_date')}: "
                + (f"FAILED {', '.join(failed)} " if failed else "")
                + (f"SKIPPED {', '.join(skipped)}" if skipped else "")
            )
        elif skipped is None and alerting.get("last_run_date"):
            # Not the same as "nothing was skipped": rows written before 0U do
            # not carry the field at all.
            typer.echo(
                f"         last run {alerting.get('last_run_date')}: "
                f"skipped stages not recorded (row predates 0U)"
            )

    canon = state.get("canon") or {}
    if canon.get("pending"):
        typer.echo(
            f"\n[CANON] citation base {canon['resolved']:,} resolved, "
            f"{canon['pending']:,} pending"
            + (f", {canon['unresolvable']:,} unanswerable"
               if canon.get("unresolvable") else "")
            + f" ({canon['share_resolved']:.0%} done)"
        )
        typer.echo(
            "        Fills automatically at the end of each daily run. "
            "A number that only grows means the budget is too small."
        )

    missing = state["unpublished_dates"]
    if missing:
        typer.echo(
            f"\n[ATTENTION] {len(missing)} date(s) with no issue: "
            f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}"
        )
        typer.echo("            uc catch-up  — retry the ones still in range")
        raise typer.Exit(code=1)


@app.command()
def weekly(
    send: bool = typer.Option(False, "--send", help="Mail it to UC_ALERT_RECIPIENT"),
):
    """Seven days of outcomes, spend and sends — printed, or mailed with --send."""
    from .notify import notify_weekly, weekly_body, weekly_summary

    if send:
        result = notify_weekly()
        typer.echo(weekly_body(result["summary"]))
        typer.echo(f"[{result['status'].upper()}]")
        # ★ A send that did not land exits non-zero (0U, U9).
        #
        # This always returned 0, so the weekly job went green whatever
        # happened. That is fine while the mail is the signal — but the mail
        # currently reaches nobody, so **the job's own success is the signal**,
        # and a signal that is always green is not one.
        #
        # `alert_undeliverable` counts as a failure here for exactly that
        # reason: the summary was written into a runner that is about to be
        # destroyed, and nothing about that should look like success.
        # `weekly_sent` and nothing else (0V, V4). The list used to include
        # `sent`, which is what `notify_weekly` returned **whatever backend it
        # used** — so the guard could never fire and the heartbeat this job is
        # supposed to be was green every Sunday. `notify_weekly` now answers
        # the same question `notify_failure` does.
        if result.get("status") != "weekly_sent":
            raise typer.Exit(code=1)
        return
    typer.echo(weekly_body(weekly_summary()))


@app.command()
def site(
    review: bool = typer.Option(True, help="Also write docs/design-review.html"),
):
    """Build the home and archive pages from `content/`, plus the review file.

    Static HTML only — no Astro, no deploy, no external request. Every number in
    the chrome is measured from the archive at build time, because a service
    whose own description disagrees with its own data is doing the thing this
    one exists not to do.
    """
    from .render.site import build_design_review, build_site

    for key, value in build_site().items():
        typer.echo(f"[OK] {key:<9} — {value}")
    if review:
        typer.echo(f"[OK] {'review':<9} — {build_design_review()}")


@app.command("export-labeling-set")
def export_labeling_set_cmd(
    date_: Optional[str] = DateOpt,
    days: int = typer.Option(1, help="Export this many days, ending at --date"),
    out: Optional[str] = typer.Option(None, help="Output path (default runs/labeling-set.json)"),
):
    """Bundle everything a labelling session needs into one file.

    Labelling happens wherever YJUN is; the state for a day is spread across
    several stage files. Moving it was a hand-built tar once, which worked and
    is not a procedure. `uc import-labeling-set` reads it back.
    """
    from .labeling import export_labeling_set

    end = _date(date_)
    dates = [end - timedelta(days=i) for i in range(days - 1, -1, -1)]
    target = Path(out) if out else (paths.RUNS / "labeling-set.json")
    typer.echo(json.dumps(export_labeling_set(dates, target), indent=2))


@app.command("import-labeling-set")
def import_labeling_set_cmd(
    path: str = typer.Argument(..., help="The file written by export-labeling-set"),
):
    """Restore an exported labelling set into this machine's run directories."""
    from .labeling import import_labeling_set

    typer.echo(json.dumps(import_labeling_set(Path(path)), indent=2))


@app.command("prepare-probe")
def prepare_probe_cmd(
    date_: Optional[str] = DateOpt,
    days: int = typer.Option(7, help="Days of candidates to draw the probe from"),
    limit: int = typer.Option(30, help="Probe size, split evenly by affinity band"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Pick the sample without summarising"),
):
    """Summarise the affinity probe's picks so they judge as fast as the ranked pass.

    Writes `runs/labels/affinity_probe_pool.jsonl` only. No stage output and no
    part of `content/` is touched — the probe must not change what the pipeline
    would have produced for these dates.
    """
    from .labeling import prepare_probe

    d = _date(date_)
    dates = [d - timedelta(days=i) for i in range(days - 1, -1, -1)]
    typer.echo(
        json.dumps(
            prepare_probe(dates, per_band=max(1, limit // 3), summarize=not no_llm),
            indent=2,
        )
    )


@app.command("prepare-labeling")
def prepare_labeling(
    date_: Optional[str] = DateOpt,
    days: int = typer.Option(1, help="Prepare this many days, ending at --date"),
    per_source: int = typer.Option(15, help="Sample size per source per day"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Build the pool without summarising"),
):
    """Make sure every item in the labelling sample has a summary.

    The published issue carries 24 items but the labelling sample is 30 drawn
    from a wider pool, so some candidates would otherwise reach the labeller
    with only an abstract — roughly three times slower to judge.
    """
    from .labeling import prepare_day

    end = _date(date_)
    results = []
    for i in range(days - 1, -1, -1):
        d = end - timedelta(days=i)
        r = prepare_day(d, per_source=per_source, summarize=not no_llm)
        results.append(r)
        typer.echo(json.dumps(r))
    ready = sum(r.get("with_summary", 0) for r in results)
    total = sum(r.get("sample", 0) for r in results)
    typer.echo(f"\n[OK] {ready}/{total} labelling items have a summary")


@app.command()
def labels(
    facet: str = typer.Option("relevance", help="Which label file to summarise"),
    k: int = typer.Option(10, help="k for precision@k"),
):
    """Summarise collected labels: precision@k, score bands, drop reasons.

    Everything here is per source and measurement only. Neither
    `classifier.threshold` nor `selection.slots` is derived from it — those are
    YJUN's calls, and one labelled day is not enough to make them.

    **One label file at a time, and the summary follows the sampling.** The
    ranked file gets precision@k; the affinity probe gets band keep rates. There
    is no `--facet all`, because the only thing pooling them could produce is a
    number that looks like precision and is not one.
    """
    from .calibrate import arxiv_candidates_by_floor
    from .labeling import PROBE_FACETS, keep_rate_by_standard, probe_summary

    # The labelling bar changed on 2026-08-19 (T8). Shown first and shown split,
    # because a keep rate averaged across that line is the keep rate of a
    # mixture of two standards.
    split = keep_rate_by_standard(facet) if facet not in PROBE_FACETS else None
    if split:
        typer.echo(json.dumps({"labelling_standards": split}, indent=2))

    if facet in PROBE_FACETS:
        typer.echo(json.dumps(probe_summary(facet), indent=2))
        return

    from .labeling import precision_at_k

    out = precision_at_k(facet=facet, k=k)
    # The other half of the threshold question: precision says how good the
    # items above a floor are, this says whether there are enough of them.
    out["arxiv_candidates_by_floor"] = arxiv_candidates_by_floor()
    typer.echo(json.dumps(out, indent=2))


@app.command()
def report(
    out: Optional[str] = typer.Option(None, help="Output path (default docs/phase0-report.md)"),
):
    """Aggregate every run into docs/phase0-report.md — Phase 0's real output."""
    from .report import build_report

    path = build_report(Path(out) if out else None)
    typer.echo(f"[OK] report — {path}")


@app.command()
def validate():
    """Validate every file under content/ against the pydantic schema."""
    from .validate import validate_content

    result = validate_content()
    for line in result.lines:
        typer.echo(line)
    raise typer.Exit(code=0 if result.ok else 1)


@app.command()
def schema():
    """Regenerate pipeline/schemas/*.json from the pydantic models."""
    from .models import Entity, Issue, Item

    paths.SCHEMAS.mkdir(parents=True, exist_ok=True)
    for name, model in (("item", Item), ("issue", Issue), ("entity", Entity)):
        target = paths.SCHEMAS / f"{name}.schema.json"
        store.write_text_atomic(
            target,
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
        )
        typer.echo(f"[OK] {target}")


@app.command()
def graph():
    """Rebuild content/graph/edges.jsonl and content/entities/ from Items."""
    from .graph.build import build_edges
    from .linking.pipeline import rebuild_entity_nodes

    n_nodes = rebuild_entity_nodes()
    n_edges = build_edges()
    typer.echo(f"[OK] graph — {n_nodes} entity nodes, {n_edges} edges")


@app.command()
def promote(
    date_: Optional[str] = DateOpt,
    since: Optional[str] = typer.Option(None, help="Only consider issues from this date on"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Recover abstracts without summarising"),
):
    """Retry enrichment for unreadable items and promote any that gain an abstract.

    Published issues are immutable: a past issue's `unreadable` list is never
    edited. A promoted item joins the next run's candidate pool, the way a
    preprint that becomes a journal article does.
    """
    from .promote import promote as run_promote

    result = run_promote(
        _date(date_),
        since=date.fromisoformat(since) if since else None,
        use_llm=not no_llm,
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def citations(
    rebuild: bool = typer.Option(True, help="Rebuild the reference base first"),
    neighbours: int = typer.Option(3, help="Top coupled neighbours to sample per item"),
):
    """Build the citation layer: reference base, internal cites, coupling.

    All build outputs under `content/graph/`. Nothing here is published or
    rendered; `runs/coupling_neighbours.json` is a sample for eyeballing quality.
    """
    from .graph.citation import (
        build_coupling,
        build_reference_base,
        internal_citation_edges,
        top_neighbours,
    )

    if rebuild:
        typer.echo(json.dumps({"references": build_reference_base()}))
    internal = internal_citation_edges()
    coupling = build_coupling()
    typer.echo(json.dumps({"cites_internal": len(internal), "coupling": coupling}))

    out = paths.RUNS / "coupling_neighbours.json"
    out.write_text(
        json.dumps(top_neighbours(k=neighbours), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    typer.echo(f"[OK] citations — neighbours sample: {out}")


@app.command("accumulate-canon")
def accumulate_canon(
    date_: Optional[str] = DateOpt,
    max_ids: Optional[int] = typer.Option(None, help="Override the budget-derived id cap"),
):
    """Fold a day into the reference base and resolve what the budget allows.

    Runs automatically as the last step of `uc daily` since 0T; this command is
    for running it by hand or for catching a queue up out of band.

    Takes at most `canon.daily_budget_fraction` of the OpenAlex day budget,
    minus what the run has already spent, and stops after
    `canon.max_seconds_per_run` or when the day's deadline is close — whichever
    comes first. Measured: 50 ids a request, $0.0001 and 1.7s a request, so
    **time is the binding limit and money is not**.

    Unresolved IDs wait in `runs/state/canon_pending.jsonl` and go first
    tomorrow; the queue length is recorded in metrics, because a queue that only
    grows means the budget is too small.
    """
    from .graph.daily_canon import accumulate_day

    result = accumulate_day(_date(date_), max_ids=max_ids)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def canon(
    top: Optional[int] = typer.Option(None, help="How many in-scope candidates to keep"),
    mode: Optional[str] = typer.Option(None, help="Scope rule: venue | subfield | both"),
):
    """Rank the works this archive keeps citing. Selection only, never published."""
    from .graph.canon import build_candidates

    kwargs: dict = {}
    if top:
        kwargs["top_n"] = top
    if mode:
        kwargs["mode"] = mode
    result = build_candidates(**kwargs)
    typer.echo(json.dumps(
        {k: v for k, v in result.items() if k not in ("candidates", "out_of_scope_examples")},
        indent=2,
    ))


@app.command()
def centrality(
    metapath: str = typer.Option("method-method", help="method-method | data-method | researcher-researcher"),
    window: int = typer.Option(90, help="Days of archive to include"),
    min_degree: int = typer.Option(3, help="Drop nodes below this degree first"),
    stability_check: bool = typer.Option(True, "--stability/--no-stability"),
):
    """Betweenness and degree over a metapath projection. Analysis only."""
    from .graph.centrality import (
        centrality as compute,
        project,
        researcher_projection,
        stability as measure_stability,
    )

    weights = (
        researcher_projection(window)
        if metapath == "researcher-researcher"
        else project(metapath, days=window)
    )
    result = compute(weights, min_degree=min_degree)
    payload = {
        "metapath": metapath,
        "window_days": window,
        "nodes": result["nodes"],
        "edges": result["edges"],
        "nodes_after_floor": result.get("nodes_after_floor"),
        "betweenness_top": result["betweenness"][:10],
        "degree_top": result["degree"][:10],
    }
    if stability_check:
        payload["stability"] = measure_stability(metapath, min_degree=min_degree)
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def backfill(
    days: int = typer.Option(90, help="Days to look back from --date"),
    date_: Optional[str] = DateOpt,
    source: str = typer.Option("all", help="all | arxiv | openalex"),
    max_pages: Optional[int] = typer.Option(None, help="Cap pages per source (testing)"),
):
    """Collect, gate, classify and score a date range. Does NOT summarise.

    Defaults to both sources: a threshold calibrated on arXiv alone does not
    describe a population that is half journal articles scoring 1.0 by
    membership.
    """
    from .calibrate import run_backfill

    meta = run_backfill(_date(date_), days=days, sources=source, max_pages=max_pages)
    typer.echo(json.dumps(meta, indent=2))


@app.command()
def calibrate(
    apply: bool = typer.Option(False, "--apply", help="Write the threshold into config/scoring.yaml"),
    target_low: float = typer.Option(0.30),
    target_high: float = typer.Option(0.50),
):
    """Pick the quiet-day threshold from the backfill score distribution (Q3)."""
    from .calibrate import calibrate_threshold

    result = calibrate_threshold(target_low, target_high, apply=apply)
    typer.echo(json.dumps(result, indent=2))


@app.command("gate-recall")
def gate_recall(
    sample: int = typer.Option(200, help="Items to draw from the rejected set"),
    date_: Optional[str] = DateOpt,
):
    """Measure how much the keyword gate throws away (PRD §5.3, once per Phase 0)."""
    from .gate_recall import measure_gate_recall

    result = measure_gate_recall(sample=sample, run_date=_date(date_))
    typer.echo(json.dumps(result, indent=2))


@app.command()
def version():
    """Print the pipeline version."""
    typer.echo(PIPELINE_VERSION)


if __name__ == "__main__":  # pragma: no cover
    app()
