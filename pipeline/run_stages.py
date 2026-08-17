"""Stage orchestration. ``cli.py`` is a thin typer wrapper over these.

Each stage reads the previous stage's JSONL and writes its own, so any stage can
be re-run alone (PRD §5). Re-running summarize must never require re-collecting.

Every stage records OK / SKIPPED / FAILED in ``metrics.stages``. A missing API
key produces SKIPPED and the run continues — a partial issue beats no issue.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import date
from typing import Optional

from . import store
from .collectors.abstracts import publisher_of
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
# enrich
# --------------------------------------------------------------------------


def stage_enrich(
    run: Run, sources: tuple[str, ...] = ("crossref", "springer"), enricher=None
) -> list[Item]:
    """Recover abstracts publishers withdrew from OpenAlex (phase 0c, P1/P2).

    Only whitelist-journal candidates are asked about: an arXiv record always
    carries its own abstract, so a request for one is a request wasted. Runs
    before `classify` so a recovered abstract is in front of the classifier and
    the journal ranking rather than behind them.
    """
    from .collectors.abstracts import enrich_abstracts, needs_abstract

    items = read_input(run, "enrich")
    targets = [it for it in items if _is_whitelist_journal(it) and needs_abstract(it)]

    if not targets:
        # Still stamp provenance, so `abstract_source` is a fact about every
        # item rather than only about the ones this stage happened to touch.
        enrich_abstracts(items, run, sources=())
        write_stage(run, "enrich", items)
        run.stage("enrich", "OK")
        run.save()
        return items

    counts = enrich_abstracts(targets, run, enricher=enricher, sources=sources)
    for it in items:
        if it.provenance.abstract_source == "none" and not needs_abstract(it):
            it.provenance.abstract_source = "openalex"

    # Counted over every candidate, not over the ones this stage asked about.
    # `enrich_abstracts` only sees the items that needed help, so its `openalex`
    # tally is 0 by construction — reporting that as "OpenAlex supplied none"
    # would be the same implicit-population mistake the calibration had.
    run.count("abstract_attempted", counts["attempted"])
    run.count("abstract_candidates", len(items))
    for name in ("crossref", "springer_api"):
        run.count(f"abstract_{name}", counts[name])
    run.count(
        "abstract_openalex",
        sum(1 for it in items if it.provenance.abstract_source == "openalex"),
    )
    run.count(
        "abstract_none",
        sum(1 for it in items if it.provenance.abstract_source == "none"),
    )

    from .config import secret

    if "springer" in sources and not secret("SPRINGER_API_KEY"):
        # A missing key is a SKIPPED sub-step, not a failure: the run continues
        # and the recoverable items simply stay unreadable for now.
        run.stage("enrich.springer", "SKIPPED")

    write_stage(run, "enrich", items)
    run.stage("enrich", "OK")
    run.save()
    return items


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------


def stage_classify(run: Run) -> list[Item]:
    """Score relevance for arXiv candidates only.

    A whitelist-journal article is relevant by membership — that is what putting
    the journal on the list asserted. Running the classifier over it adds no
    information (it returns ~0.99 because it was trained on those journals) and
    costs an embedding, so those items are assigned 1.0 with an explicit
    ``classifier_version`` of ``whitelist-membership`` rather than a model
    number that would look like a prediction.
    """
    items = read_input(run, "classify")
    journal_items = [it for it in items if _is_whitelist_journal(it)]
    arxiv_items = [it for it in items if not _is_whitelist_journal(it)]

    for it in journal_items:
        it.scores.relevance = 1.0
        it.scores.components.relevance = 1.0
        it.provenance.classifier_version = "whitelist-membership"

    with run.timed("classify_s"):
        pred = score_items(arxiv_items)

    run.count("classified", len(arxiv_items))
    setattr(run.metrics.counts, "classify_skipped_journal", len(journal_items))
    run.metrics.stages["classify.model"] = pred.version
    write_stage(run, "classify", items)
    run.stage("classify", "OK")
    run.save()
    return items


# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------


_WHITELIST_IDS: dict[int, set[str]] = {}


def _whitelist_source_ids() -> set[str]:
    """Included whitelist source IDs, memoised against the parsed document.

    `journals_vocab()` is already mtime-cached; this avoids rebuilding the set
    itself once per item, which a backfill does tens of thousands of times.
    """
    from .config import journals_vocab

    doc = journals_vocab()
    key = id(doc)
    cached = _WHITELIST_IDS.get(key)
    if cached is None:
        _WHITELIST_IDS.clear()
        cached = {
            s["id"]
            for s in (doc.get("sources") or [])
            if s.get("id") and s.get("include", True)
        }
        _WHITELIST_IDS[key] = cached
    return cached


def _is_whitelist_journal(item: Item) -> bool:
    src = item.bibliography.primary_location.source_id
    return bool(src) and src in _whitelist_source_ids()


def journal_rank_score(item: Item) -> float:
    """PLACEHOLDER ranking for the journal path.

    There is no signal here yet for the question that actually matters —
    "is this the *kind* of paper we cover?" *Cities* publishes qualitative case
    studies, theory pieces and policy commentary alongside the data-and-method
    work this digest is about, and the relevance classifier cannot tell them
    apart because it was trained on those journals wholesale.

    So this ranks on the score components that do not depend on relevance:
    artifact completeness, novelty, and cluster multiplicity. That is a weak
    proxy and it is labelled as one rather than dressed up.

    **Replace once Q1b labels exist.** The `q` drop reason ("urban research but
    not our kind") in `uc review --label relevance` is being collected precisely
    to train the classifier that belongs here.
    """
    c = item.scores.components
    return round(
        0.5 * c.artifact_completeness + 0.3 * c.novelty + 0.2 * c.source_multiplicity, 4
    )


UNREADABLE_STAGE = "unreadable"


def has_abstract(item: Item) -> bool:
    return bool((item.bibliography.abstract or "").strip())


@dataclass(frozen=True)
class SlotPolicy:
    """How many items each path may publish, and how good an arXiv one must be.

    **This is a publication policy, not a candidacy one.** `classifier.threshold`
    still decides what enters the day's candidate pool, and the labelling sample
    is drawn from that pool — so nothing here can move a precision@k figure. The
    120 labels stand on the candidate ranking, and the candidate ranking is
    untouched.

    The measurement that produced these numbers (90 relevance labels, 3 days):

    | arXiv floor | labelled precision | n  | median candidates/day (90-day backfill) |
    |------------:|-------------------:|---:|----------------------------------------:|
    | 0.35        | 0.489              | 45 | 18   |
    | 0.50        | 0.556              | 36 | 11   |
    | 0.70        | 0.722              | 18 | 6    |
    | **0.80**    | **1.000**          | 9  | **3** |
    | 0.90        | 1.000              | 6  | 1.5  |

    The band between 0.70 and 0.80 is 4 keeps out of 9 — 0.44, barely better
    than the 0.35-0.70 mass at 0.33, and it is the whole reason the 0.70 floor
    measures 0.72. Above 0.80 every labelled item was a keep. Nine of nine is
    not proof: the rule of three puts the 95% lower bound near 0.67, so the
    honest claim is "no observed failures in nine", not "perfect".

    So the arXiv path stops filling instead of reaching down. A day with three
    arXiv items is the correct output of a day with three good arXiv preprints.
    """

    arxiv_floor: float
    arxiv_max: int
    journal_base: int
    journal_max: int

    @classmethod
    def from_config(cls) -> "SlotPolicy":
        return cls(
            arxiv_floor=float(cfg("selection.arxiv.floor", 0.80)),
            arxiv_max=int(cfg("selection.arxiv.max", 12)),
            journal_base=int(cfg("selection.slots.journal", 12)),
            journal_max=int(cfg("selection.journal.max", 15)),
        )

    @classmethod
    def even_split(cls, top_n: int) -> "SlotPolicy":
        """`--top N` splits evenly and keeps the floor — an override of size, not of standard."""
        half = int(top_n) // 2
        return cls(
            arxiv_floor=float(cfg("selection.arxiv.floor", 0.80)),
            arxiv_max=int(top_n) - half,
            journal_base=half,
            journal_max=half,
        )


def fill_slots(
    items: list[Item],
    threshold: float,
    journal_slots: Optional[int] = None,
    arxiv_slots: Optional[int] = None,
    policy: Optional[SlotPolicy] = None,
) -> tuple[list[Item], list[Item]]:
    """Which items fill a day's two sets of slots, ranked and lent.

    Pure: no Run, no stage files, no scoring — the items must already carry
    their scores. `stage_select` uses it to publish a day; the backfill uses it
    to work out which of a day's candidates *would* have published, which is the
    population the quiet-day threshold is calibrated on. One rule, one place.

    An item with no abstract is not eligible for a journal slot. Its card would
    be a title and nothing else, and it publishes in `Also published today`
    instead. Enforced here rather than in `stage_select` so the backfill's idea
    of what would publish keeps matching what does.

    **The lending is one-directional now.** It used to run both ways so the day
    always totalled 24, which meant a thin arXiv day was patched by reaching
    further down the arXiv ranking — into the 33%-keep region the labels found.
    A short day is now allowed to be short: the journal path may expand to
    `journal_max`, and beyond that the day simply publishes fewer items.
    """
    if policy is None:
        policy = SlotPolicy.from_config()
    if journal_slots is not None or arxiv_slots is not None:
        # Explicit slot counts still work — the backfill and the tests use them.
        policy = SlotPolicy(
            arxiv_floor=policy.arxiv_floor,
            arxiv_max=arxiv_slots if arxiv_slots is not None else policy.arxiv_max,
            journal_base=journal_slots if journal_slots is not None else policy.journal_base,
            journal_max=max(
                journal_slots if journal_slots is not None else policy.journal_base,
                policy.journal_max,
            ),
        )

    journal_pool = sorted(
        (it for it in items if _is_whitelist_journal(it) and has_abstract(it)),
        key=lambda it: (-journal_rank_score(it), it.work_key),
    )
    # Candidacy is `threshold`; publication is the floor. The floor is never
    # below the threshold — a lower one would publish items the day never
    # collected as candidates.
    floor = max(float(threshold), policy.arxiv_floor)
    arxiv_pool = sorted(
        (
            it
            for it in items
            if not _is_whitelist_journal(it) and it.scores.relevance >= floor
        ),
        key=lambda it: (-it.scores.relevance, it.work_key),
    )

    arxiv_taken = arxiv_pool[: policy.arxiv_max]
    journal_slots_now = min(
        policy.journal_max,
        policy.journal_base + max(0, policy.arxiv_max - len(arxiv_taken)),
    )
    journal_taken = journal_pool[:journal_slots_now]
    return journal_taken, arxiv_taken


def stage_select(
    run: Run, threshold: Optional[float] = None, top_n: Optional[int] = None
) -> list[Item]:
    """Fill the day's list from two independent entry paths (roadmap §2.1).

    | path    | entry                        | ranking                  |
    |---------|------------------------------|--------------------------|
    | journal | whitelist membership          | placeholder, see above   |
    | arxiv   | classifier probability >= thr | probability              |

    Phase 0 put both through one classifier and then imposed an arXiv quota to
    stop journal articles taking every slot (D14). The quota treated a symptom:
    a whitelist article scores ~0.99 nearly by construction, so the classifier
    added no information on that side and its score could not rank within it.
    Separate paths make the quota unnecessary — each path owns its slots.
    """
    items = read_input(run, "select")
    thr = cfg("classifier.threshold", 0.35) if threshold is None else threshold

    policy = SlotPolicy.even_split(top_n) if top_n is not None else SlotPolicy.from_config()

    # An item no source could give an abstract for cannot be summarised, so its
    # card would carry a title and nothing else. Measured on the five prepared
    # days: 5-10 of 24 published cards were in that state. They are not dropped
    # and not ranked last — they leave the slot competition entirely and publish
    # in `Also published today`, where the facts we do have are stated and
    # nothing is invented. Ranking them last was the earlier half-measure; two
    # mechanisms for one decision is one too many.
    unreadable = [
        it for it in items if _is_whitelist_journal(it) and not has_abstract(it)
    ]
    write_stage(run, UNREADABLE_STAGE, unreadable)
    run.count("unreadable", len(unreadable))

    journal_pool = [it for it in items if _is_whitelist_journal(it) and has_abstract(it)]
    arxiv_pool = [
        it for it in items if not _is_whitelist_journal(it) and it.scores.relevance >= thr
    ]

    for it in journal_pool:
        apply_rule_signals(it)
        apply_badges(it)
    for it in arxiv_pool:
        apply_rule_signals(it)
        apply_badges(it)

    from .score.headline import score_item

    seen = _seen_entity_ids(exclude={it.work_key for it in items})
    for it in journal_pool:
        score_item(it, seen)

    journal_taken, arxiv_taken = fill_slots(items, thr, policy=policy)
    selected = journal_taken + arxiv_taken
    selected.sort(key=lambda it: (-it.scores.headline, -it.scores.relevance, it.work_key))

    setattr(run.metrics.counts, "journal_candidates", len(journal_pool))
    setattr(run.metrics.counts, "arxiv_candidates", len(arxiv_pool))
    setattr(run.metrics.counts, "selected_journal", len(journal_taken))
    setattr(run.metrics.counts, "selected_arxiv", len(arxiv_taken))
    # The arXiv path being under its ceiling is the policy working, not a
    # shortage: above the floor is all there was. Recorded as a count, not an
    # error, so the day's own log stops calling a correct outcome a problem.
    setattr(run.metrics.counts, "arxiv_above_floor", len(arxiv_taken))
    run.metrics.timing.setdefault("arxiv_floor", policy.arxiv_floor)
    if len(journal_taken) < policy.journal_base:
        run.error(
            f"select: short day — journal {len(journal_taken)}/{policy.journal_base}, "
            f"arxiv {len(arxiv_taken)} above the {policy.arxiv_floor} floor"
        )

    run.count("selected", len(selected))
    write_stage(run, "select", selected)
    run.stage("select", "OK")
    run.save()
    return selected


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


def reconcile_places_status(items: list[Item]) -> int:
    """Distinguish "no place to find" from "we did not find one" (P4-2).

    `link_places` returns `unspecified` whenever it resolves nothing, which
    collapses two different facts. `PlacesStatus` has carried `not_applicable`
    since the schema was written and nothing ever set it, so an item the LLM
    judged to have no study area at all — `signals.geographic_scope =
    not_applicable` — was recorded identically to one whose city we simply could
    not resolve.

    It has to happen here rather than in `link`, because `link` runs before
    `summarize` and the scope is not known until the summary call returns.

    Places is a de-prioritised axis (PRD §2, v1.1), which is exactly why this
    matters: the field is being filled now and read later. A wrong value written
    today is an archive nobody can trust when the axis is revived.
    """
    changed = 0
    for item in items:
        scope = item.signals.geographic_scope
        if (
            scope is not None
            and scope.value == "not_applicable"
            and not item.entities.places
            and item.entities.places_status != "not_applicable"
        ):
            item.entities.places_status = "not_applicable"
            changed += 1
    return changed


def stage_summarize(run: Run, use_llm: bool = True, limit: Optional[int] = None) -> list[Item]:
    items = read_input(run, "summarize")
    from .summarize.run import summarize_items

    with run.timed("summarize_s"):
        stats = summarize_items(items, run, use_llm=use_llm, limit=limit)
    run.count("summarized", stats.get("summarized", 0))
    reconcile_places_status(items)
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


def _restore_run_outputs(merged: Item, fresh: Item) -> None:
    """Let this run's derived fields win over the stored copy's.

    `_merge_pair` exists to fold an arXiv preprint and an OpenAlex Work into one
    record, so it accumulates: the base keeps what it has and takes only what it
    lacks. That is right for identity and bibliography, and wrong for anything a
    stage just computed. Folding the archive in as `base` meant a re-run's
    summary, signals and scores were all discarded in favour of the stored ones.

    Found by bumping the summarize prompt to 0.4.0: 121 summaries were
    regenerated, the stage file carried every one of them, and `content/` still
    said `summarize/papers@0.3.0`. A prompt version could never reach a
    published item, which makes `uc summarize --date` pointless for one.

    Idempotency is unaffected (PRD §9): a second run of the same date recomputes
    the same values from the same cache, so the bytes still match.
    """
    if fresh.summary.en and fresh.summary.en.what:
        merged.summary = fresh.summary
        merged.provenance.llm = fresh.provenance.llm
        merged.signals = fresh.signals
    # The overlay is this run's reading of the abstract, and scores are computed
    # against today's archive. Both are outputs, not accumulated facts.
    for facet in ("methods", "data", "tools"):
        if getattr(fresh.entities, facet):
            setattr(merged.entities, facet, getattr(fresh.entities, facet))
    if fresh.entities.places or fresh.entities.places_status != "not_attempted":
        merged.entities.places = fresh.entities.places
        merged.entities.places_status = fresh.entities.places_status
    merged.scores = fresh.scores
    merged.badges = fresh.badges
    merged.provenance.abstract_source = fresh.provenance.abstract_source


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

            fresh = it
            it = _merge_pair(existing.model_copy(deep=True), it)
            _restore_run_outputs(it, fresh)
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

    # `Also published today` (P3). These appeared in a tracked journal and no
    # source could give us an abstract, so no card can be written about them.
    # They are stored as ordinary Items — same work_key, no summary — so that
    # the day an abstract turns up, the same record is promoted to a real card
    # rather than published twice. `published_index` reads `issue.items` only,
    # which is what makes that promotion possible.
    unreadable = [
        it for it in read_stage(run, UNREADABLE_STAGE)
        if not already.get(it.work_key)
    ]
    for it in unreadable:
        store.save_item(it, today=d)

    by_publisher: dict[str, int] = {}
    for it in unreadable:
        name = publisher_of(it)
        by_publisher[name] = by_publisher.get(name, 0) + 1

    headline_item = pick_headline(publish)
    scan = ScanMeta(
        arxiv_categories=len(cfg("arxiv.categories", []) or []),
        journals=_journal_count(),
        candidates_scanned=int(run.metrics.counts.arxiv_fetched)
        + int(run.metrics.counts.openalex_fetched),
        candidates_after_gate=int(run.metrics.counts.after_gate),
        items_published=len(publish),
        minutes_saved_estimate=len(publish) * int(cfg("review.minutes_saved_per_item", 7)),
        unreadable_count=len(unreadable),
        unreadable_by_publisher=dict(sorted(by_publisher.items())),
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
        unreadable=sorted(it.work_key for it in unreadable),
        status_changes=status_changes,
        run_id=run.run_id,
    )
    store.save_issue(issue)
    run.count("published", len(publish))
    run.count("unreadable_published", len(unreadable))
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
    unreadable = [
        it for it in (store.load_item(k) for k in issue.unreadable) if it is not None
    ]
    out = write_preview(issue, items, run.dir / "preview.html", unreadable=unreadable)
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
    enrich: bool = True,
) -> Run:
    run = Run.for_date(d)
    _guard(run, "collect", lambda: stage_collect(run, d, sources=sources, fixture=fixture))
    _guard(run, "dedup", lambda: stage_dedup(run, d))
    _guard(run, "gate", lambda: stage_gate(run))
    _guard(
        run,
        "enrich",
        lambda: stage_enrich(run, sources=("crossref", "springer") if enrich else ()),
    )
    _guard(run, "classify", lambda: stage_classify(run))
    _guard(run, "select", lambda: stage_select(run))
    _guard(run, "link", lambda: stage_link(run, use_llm=use_llm))
    _guard(run, "summarize", lambda: stage_summarize(run, use_llm=use_llm, limit=summarize_limit))
    _guard(run, "score", lambda: stage_score(run))
    _guard(run, "issue", lambda: stage_issue(run, d))
    _guard(run, "preview", lambda: stage_preview(run, d))
    run.save()
    return run
