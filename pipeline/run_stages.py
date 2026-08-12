"""Stage orchestration. ``cli.py`` is a thin typer wrapper over these.

Each stage reads the previous stage's JSONL and writes its own, so any stage can
be re-run alone (PRD §5). Re-running summarize must never require re-collecting.

Every stage records OK / SKIPPED / FAILED in ``metrics.stages``. A missing API
key produces SKIPPED and the run continues — a partial issue beats no issue.
"""

from __future__ import annotations

import traceback
from datetime import date
from typing import Optional

from . import store
from .config import cfg
from .dedup.merge import merge_candidates
from .filters.classifier import score_items
from .filters.gate import Gate, apply_gate
from .metrics import Run
from .models import Headline, Issue, Item, ScanMeta
from .render.preview import write_preview
from .score.headline import headline_line, pick_headline, score_all
from .signals import apply_badges, apply_rule_signals
from .stages import read_input, read_stage, write_stage


class StageSkipped(RuntimeError):
    """Raised when a stage cannot run (no key, no model) but the run should go on."""


def _guard(run: Run, name: str, fn):
    """Run a stage, converting its failure into a recorded status.

    A stage that reports its own status (SKIPPED, PARTIAL) keeps it; ``_guard``
    only fills in OK when the stage said nothing.
    """
    before = run.metrics.stages.get(name)
    try:
        result = fn()
        if run.metrics.stages.get(name) == before:
            run.stage(name, "OK")
        return result
    except StageSkipped as e:
        run.stage(name, "SKIPPED")
        run.error(f"{name}: SKIPPED — {e}")
        return None
    except Exception as e:  # noqa: BLE001 - one stage failing must not kill the run
        run.stage(name, "FAILED")
        run.error(f"{name}: FAILED — {type(e).__name__}: {e}")
        run.metrics.errors.append(traceback.format_exc(limit=3))
        return None
    finally:
        run.save()


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------


def stage_collect(
    run: Run,
    d: date,
    sources: str = "all",
    fixture: bool = False,
    backfill_from: Optional[date] = None,
) -> list[Item]:
    """arXiv + OpenAlex collection. Raw responses are preserved verbatim."""
    items: list[Item] = []

    if fixture:
        from .collectors.fixtures import fixture_items

        items = fixture_items(d)
        run.count("arxiv_fetched", len(items))
        with run.timed("collect_s"):
            pass
        write_stage(run, "collect", items)
        run.stage("collect", "OK")
        run.save()
        return items

    with run.timed("collect_s"):
        if sources in ("all", "arxiv"):
            from .collectors.arxiv import ArxivCollector

            def _arxiv():
                c = ArxivCollector(run)
                return c.collect(d, backfill_from=backfill_from)

            got = _guard(run, "collect.arxiv", _arxiv) or []
            run.count("arxiv_fetched", len(got))
            items.extend(got)

        if sources in ("all", "openalex"):
            from .collectors.openalex import OpenAlexCollector

            def _openalex():
                c = OpenAlexCollector(run)
                return c.collect_journals(d, backfill_from=backfill_from)

            got = _guard(run, "collect.openalex", _openalex) or []
            run.count("openalex_fetched", len(got))
            items.extend(got)

            # Enrichment pass: fill ids.openalex / graph / topics on arXiv items.
            # Best-effort by design — arXiv indexing lags by days (PRD §5.1).
            arxiv_items = [it for it in items if it.ids.arxiv and not it.ids.openalex]
            if arxiv_items:

                def _enrich():
                    c = OpenAlexCollector(run)
                    return c.enrich(arxiv_items)

                _guard(run, "collect.enrich", _enrich)

    if not items:
        # arXiv's submittedDate index lags: a query for "yesterday" can legitimately
        # return zero while the same query a day later returns hundreds. Silence
        # here reads as "a quiet day" when it actually means "come back later".
        run.error(
            f"collect: zero candidates for {d}. arXiv indexes submissions with a "
            f"lag — re-run this date tomorrow before treating it as a quiet day."
        )
        run.stage("collect", "EMPTY")
    else:
        run.stage("collect", "OK")

    write_stage(run, "collect", items)
    run.save()
    return items


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------


def stage_dedup(run: Run, d: date) -> list[Item]:
    candidates = read_stage(run, "collect")
    result = merge_candidates(candidates, run_date=d)
    run.count("after_dedup", len(result.items))
    setattr(run.metrics.counts, "merged_away", result.merged_away)
    write_stage(run, "dedup", result.items)
    run.stage("dedup", "OK")
    run.save()
    return result.items


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------


def stage_gate(run: Run) -> list[Item]:
    items = read_input(run, "gate")
    kept, dropped = apply_gate(items, Gate.from_vocab())
    run.count("after_gate", len(kept))
    for it, reason in dropped:
        run.append_jsonl(
            "gate_dropped.jsonl",
            {"work_key": it.work_key, "reason": reason, "title": it.bibliography.title},
        )
    write_stage(run, "gate", kept)
    run.stage("gate", "OK")
    run.save()
    return kept


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------


def stage_classify(run: Run) -> list[Item]:
    items = read_input(run, "classify")
    with run.timed("classify_s"):
        pred = score_items(items)
    run.count("classified", len(items))
    run.metrics.stages["classify.model"] = pred.version
    write_stage(run, "classify", items)
    run.stage("classify", "OK")
    run.save()
    return items


# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------


def stage_select(run: Run, threshold: Optional[float] = None, top_n: Optional[int] = None) -> list[Item]:
    items = read_input(run, "select")
    thr = cfg("classifier.threshold", 0.5) if threshold is None else threshold
    n = int(cfg("classifier.select_top_n", 24) if top_n is None else top_n)
    above = [it for it in items if it.scores.relevance >= thr]
    selected = _select_with_source_quota(above, n, float(cfg("classifier.arxiv_min_share", 0.5)))
    for it in selected:
        apply_rule_signals(it)
        apply_badges(it)
    run.count("selected", len(selected))
    write_stage(run, "select", selected)
    run.stage("select", "OK")
    run.save()
    return selected


def _select_with_source_quota(candidates: list[Item], n: int, arxiv_min_share: float) -> list[Item]:
    """Fill the daily list with a floor on arXiv items.

    The classifier is trained on whitelist-journal articles, so a journal article
    scores ~0.99 close to by construction — it came from a journal on the list.
    Ranking a mixed day purely by that score fills every slot with journal items
    (measured 2026-08-11: 23 of 24) and pushes out the preprints, which are the
    part of the day that is actually new. A floor, not a fixed split: if arXiv
    has fewer qualifying items than its quota, journals take the remainder.
    """
    by_score = sorted(candidates, key=lambda it: (-it.scores.relevance, it.work_key))
    if n <= 0 or arxiv_min_share <= 0:
        return by_score[:n]

    arxiv = [it for it in by_score if it.ids.arxiv]
    other = [it for it in by_score if not it.ids.arxiv]
    quota = min(len(arxiv), int(round(n * arxiv_min_share)))

    picked = arxiv[:quota]
    picked += other[: n - len(picked)]
    if len(picked) < n:  # journals ran out; give the slack back to arXiv
        picked += arxiv[quota : quota + (n - len(picked))]
    return sorted(picked, key=lambda it: (-it.scores.relevance, it.work_key))


# --------------------------------------------------------------------------
# link
# --------------------------------------------------------------------------


def stage_link(run: Run, use_llm: bool = True) -> list[Item]:
    items = read_input(run, "link")
    from .linking.pipeline import link_items

    with run.timed("link_s"):
        stats = link_items(items, run, use_llm=use_llm)
    run.metrics.linking.topics_from_openalex = stats.get("topics_from_openalex", 0)
    run.metrics.linking.unmatched_methods = stats.get("unmatched_methods", 0)
    run.metrics.linking.unmatched_data = stats.get("unmatched_data", 0)
    write_stage(run, "link", items)
    run.stage("link", stats.get("status", "OK"))
    run.save()
    return items


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------


def stage_summarize(run: Run, use_llm: bool = True, limit: Optional[int] = None) -> list[Item]:
    items = read_input(run, "summarize")
    from .summarize.run import summarize_items

    with run.timed("summarize_s"):
        stats = summarize_items(items, run, use_llm=use_llm, limit=limit)
    run.count("summarized", stats.get("summarized", 0))
    write_stage(run, "summarize", items)
    run.stage("summarize", stats.get("status", "OK"))
    run.save()
    return items


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def stage_score(run: Run) -> list[Item]:
    items = read_input(run, "score")
    # Exclude the items being scored. On a re-run they are already in content/,
    # and counting their own tags as "previously seen" would drive novelty to
    # zero and change every headline score — breaking idempotency (PRD §9).
    seen = _seen_entity_ids(exclude={it.work_key for it in items})
    score_all(items, seen)
    write_stage(run, "score", items)
    run.stage("score", "OK")
    run.save()
    return items


def _seen_entity_ids(exclude: Optional[set[str]] = None) -> set[str]:
    """Overlay entity IDs already in the archive — the basis of the novelty term."""
    exclude = exclude or set()
    seen: set[str] = set()
    for it in store.iter_items():
        if it.work_key in exclude:
            continue
        for e in it.entities.methods + it.entities.data + it.entities.tools:
            seen.add(e.id)
    return seen


# --------------------------------------------------------------------------
# issue
# --------------------------------------------------------------------------


def stage_issue(run: Run, d: date) -> Issue:
    """Publish Items and the Issue.

    An Item that already appeared in an issue on an *earlier* date is not a new
    publication (PRD §5.2): its status is updated and a ``status_changes`` line
    is recorded. Matching on Item existence alone would make a second run of the
    same day publish an empty issue.
    """
    items = read_input(run, "issue")
    already = store.published_index()
    today = str(d)

    publish: list[Item] = []
    status_changes = []

    for it in items:
        prior_date = already.get(it.work_key)
        existing = store.load_item(it.work_key)
        if existing is not None:
            before = existing.publication_status.state
            from .dedup.merge import _merge_pair

            it = _merge_pair(existing.model_copy(deep=True), it)
            # Badges were computed at `select`; a state change after that would
            # otherwise leave a published paper still wearing a preprint badge.
            apply_badges(it)
            after = it.publication_status.state
            if prior_date and prior_date != today and before != after:
                from .models import StatusChange

                status_changes.append(
                    StatusChange(
                        work_key=it.work_key,
                        **{"from": before},
                        to=after,
                        journal=it.publication_status.journal,
                    )
                )
        if prior_date and prior_date != today:
            # Already carried by an earlier issue: update the Item, do not re-publish.
            store.save_item(it, today=d)
            continue
        publish.append(it)

    for it in publish:
        store.save_item(it, today=d)

    headline_item = pick_headline(publish)
    scan = ScanMeta(
        arxiv_categories=len(cfg("arxiv.categories", []) or []),
        journals=_journal_count(),
        candidates_scanned=int(run.metrics.counts.arxiv_fetched)
        + int(run.metrics.counts.openalex_fetched),
        candidates_after_gate=int(run.metrics.counts.after_gate),
        items_published=len(publish),
        minutes_saved_estimate=len(publish) * int(cfg("review.minutes_saved_per_item", 7)),
    )
    issue = Issue(
        date=d,
        headline=Headline(
            present=headline_item is not None,
            work_key=headline_item.work_key if headline_item else None,
            line=headline_line(headline_item) if headline_item else None,
        ),
        quiet_day=headline_item is None,
        scan_meta=scan,
        items=sorted(it.work_key for it in publish),
        status_changes=status_changes,
        run_id=run.run_id,
    )
    store.save_issue(issue)
    run.count("published", len(publish))
    write_stage(run, "issue", publish)
    run.stage("issue", "OK")
    run.save()
    return issue


def _journal_count() -> int:
    from .config import journals_vocab

    return len((journals_vocab().get("sources") or []))


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------


def stage_preview(run: Run, d: date):
    issue = store.load_issue(d)
    if issue is None:
        raise StageSkipped(f"no issue for {d}; run `uc issue --date {d}` first")
    items = [it for it in (store.load_item(k) for k in issue.items) if it is not None]
    out = write_preview(issue, items, run.dir / "preview.html")
    run.stage("preview", "OK")
    run.save()
    return out


# --------------------------------------------------------------------------
# full run
# --------------------------------------------------------------------------


def run_all(
    d: date,
    sources: str = "all",
    fixture: bool = False,
    use_llm: bool = True,
    summarize_limit: Optional[int] = None,
) -> Run:
    run = Run.for_date(d)
    _guard(run, "collect", lambda: stage_collect(run, d, sources=sources, fixture=fixture))
    _guard(run, "dedup", lambda: stage_dedup(run, d))
    _guard(run, "gate", lambda: stage_gate(run))
    _guard(run, "classify", lambda: stage_classify(run))
    _guard(run, "select", lambda: stage_select(run))
    _guard(run, "link", lambda: stage_link(run, use_llm=use_llm))
    _guard(run, "summarize", lambda: stage_summarize(run, use_llm=use_llm, limit=summarize_limit))
    _guard(run, "score", lambda: stage_score(run))
    _guard(run, "issue", lambda: stage_issue(run, d))
    _guard(run, "preview", lambda: stage_preview(run, d))
    run.save()
    return run
