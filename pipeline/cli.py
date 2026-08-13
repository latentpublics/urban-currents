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

from . import paths, run_stages, store
from .metrics import Run
from .models import PIPELINE_VERSION

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
        f"{'quiet day' if iss.quiet_day else 'headline: ' + (iss.headline.line or '')}"
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
    top: int = typer.Option(30, help="Items to label per day (split evenly by source)"),
):
    """Review a day's issue, recording elapsed time and every edited field path.

    With ``--label relevance`` it runs the Q1b labelling pass instead: a
    stratified sample of the day's candidates, resumable, with a reason on every
    drop.
    """
    from .review import run_labeling_session, run_review_session

    d = _date(date_)
    if label:
        run_labeling_session(d, facet=label, top=top)
    else:
        run_review_session(d)


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
    """Summarise collected labels: precision@k and drop reasons, per source."""
    from .labeling import precision_at_k

    typer.echo(json.dumps(precision_at_k(facet=facet, k=k), indent=2))


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
