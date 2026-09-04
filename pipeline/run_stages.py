"""Stage orchestration. ``cli.py`` is a thin typer wrapper over these.

Each stage reads the previous stage's JSONL and writes its own, so any stage can
be re-run alone (PRD §5). Re-running summarize must never require re-collecting.

Every stage records OK / SKIPPED / FAILED in ``metrics.stages``. A missing API
key produces SKIPPED and the run continues — a partial issue beats no issue.
"""

from __future__ import annotations

import re
import time
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
from .score.headline import pick_headline, score_all
from .signals import apply_badges, apply_rule_signals
# Defined in `pipeline/skips.py` so the collectors and the LLM client can
# inherit it without importing this module (hotfix H2). Re-exported because
# every stage already imports it from here.
from .skips import StageSkipped  # noqa: F401
from .stages import read_input, read_stage, write_stage


def _guard(run: Run, name: str, fn):
    """Run a stage, converting its failure into a recorded status.

    A stage that reports its own status (SKIPPED, PARTIAL) keeps it; ``_guard``
    only fills in OK when the stage said nothing.

    It also prints a line as each stage starts and finishes. Without that, a run
    that stops making progress is indistinguishable from a run that is working:
    a backfilled day sat in one stage for sixteen minutes and the only way to
    tell which stage was to read the metrics file and infer it from what was
    missing. With PYTHONUNBUFFERED set in CI, these lines arrive as they happen.
    """
    before = run.metrics.stages.get(name)
    started = time.monotonic()
    # Bound before the try: the `finally` reports it, and a stage that raises
    # would otherwise turn its own exception into a NameError in the handler.
    result = None
    print(f"[stage] {name} ...", flush=True)
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
        elapsed = time.monotonic() - started
        n = len(result) if isinstance(result, (list, tuple)) else ""
        print(
            f"[stage] {name} {run.metrics.stages.get(name, '?')} "
            f"{elapsed:.1f}s {n}".rstrip(),
            flush=True,
        )
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
    enrich_arxiv: bool = True,
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
            # Skipped by the backfill. This is one OpenAlex lookup per unmatched
            # preprint, and it hung for sixteen minutes on a single backfilled
            # day - sixty days of that is ten hours for a pass the stage itself
            # calls best-effort. The cost is that backfilled arXiv items may
            # lack `referenced_works`; the journal path is unaffected, and that
            # is what canon affinity and most coupling stand on.
            if arxiv_items and enrich_arxiv:

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
        #
        # ★ And it says how many (1A, C1). This sub-step has been SKIPPED on 16
        # of 18 days, which is a line that stops being read — the failure mode
        # the alerting work keeps running into, arriving in the run log
        # instead. A bare SKIPPED cannot be told apart from a sub-step nobody
        # needs; the count can. Measured over the 16 days to 2026-09-02: 252 of
        # 1,546 unreadable items were Springer-owned, 16%, which is the size of
        # what the key would recover and the reason the stage is kept rather
        # than removed.
        from .collectors.abstracts import SPRINGER_PREFIXES

        recoverable = sum(
            1
            for it in targets
            if (it.ids.doi or "").split("/")[0] in SPRINGER_PREFIXES
        )
        run.count("abstract_springer_unasked", recoverable)
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

    # ★ A fallback to the keyword heuristic is not an OK day (0X, X1).
    #
    # `selection.arxiv.floor` is 0.80 and was calibrated against the trained
    # model. The heuristic scores a typical abstract between 0.05 and 0.4, so
    # falling back does not lower the arXiv path's quality — it **removes**
    # the path, and the issue becomes journal-only without saying so. DEGRADED
    # rather than FAILED because the stage did run and its journal half is
    # sound; `looked()` reads it and refuses the day, which is the point.
    if pred.fallback_reason:
        run.error(
            f"classify: fell back to {pred.version} — {pred.fallback_reason}. "
            f"The arXiv floor of {cfg('selection.arxiv.floor', 0.80)} was "
            f"calibrated against the trained model, so this day's arXiv path "
            f"is effectively empty rather than merely worse."
        )
        run.stage("classify", "DEGRADED")
    else:
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

    One part of that question is already answerable without a classifier: three
    of the nine labelled `q` drops are book reviews, and a book review announces
    itself in its title. Those sink to the bottom rather than being removed, so
    a thin journal day still publishes what a tracked journal actually printed.
    """
    from .filters.book_review import demotion
    from .filters.correction import demotion as correction_demotion

    c = item.scores.components
    base = 0.5 * c.artifact_completeness + 0.3 * c.novelty + 0.2 * c.source_multiplicity
    # Two genres, two modules, one mechanism. A corrigendum is not a paper at
    # all (0P Q3) and sinks for the same reason and by the same amount.
    title = item.bibliography.title
    return round(base * demotion(title) * correction_demotion(title), 4)


UNREADABLE_STAGE = "unreadable"

# ★ Journal apparatus: things a journal issues that are not papers (1D, D1b).
#
# A DOI and a place in a tracked journal are not enough to make something an
# article. IJURR's `Issue Information` and `Cover Image` reached
# `Also published today`, and a correction notice reached the 2026-06-16 issue
# as a card whose summary reads *"a formal correction ... updating the
# acknowledgments section"*. The summariser had understood exactly what it was
# and the pipeline published it anyway, because nothing asked the question.
#
# **Both rules were measured against the whole archive before being chosen, and
# the loosest rule that suggested itself was rejected.** Over 2,670 items:
#
#   exact front-matter title      40 matches,  0 of them ever published
#   correction-notice shape       29 matches,  2 of them published
#   no authors at all             52 matches,  2 published  <- REJECTED
#
# The third would also have removed "Cities, not rural areas, power the digital
# infrastructure of the USA" and three more Nature Cities pieces, which carry no
# byline and are real writing. Absence of a byline is a fact about the metadata,
# not about the thing.
_APPARATUS_TITLES = frozenset({
    "issue information", "editorial board", "front matter", "back matter",
    "contents", "table of contents", "cover image", "issue cover",
    "title page", "copyright", "index", "masthead",
    "acknowledgements to reviewers", "acknowledgments to reviewers",
})

# `Correction to '...'`, `Corrigendum To: ...`, `Erratum: ...`, or the bare word.
# ★ Anchored on the *shape* of a notice, not on the first word alone: a paper
# called "Correction of GPS drift in ..." is a paper. No such title is in the
# archive today, so the two rules currently catch the same 29 items — the
# narrower one is chosen for the day one arrives.
_CORRECTION = re.compile(
    r"^(correction|corrigendum|erratum|retraction|withdrawal)"
    r"(\s+(to|for)\b|\s*[:\u2018\u201c\"]|\s*$)",
    re.I,
)


def is_journal_apparatus(item: Item) -> bool:
    """Front matter, back matter, or a correction notice — not a paper."""
    title = (item.bibliography.title or "").strip()
    if not title:
        return False
    if title.lower() in _APPARATUS_TITLES:
        return True
    return bool(_CORRECTION.match(title))


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

    # ★ Removed before either path sees them (1D, D1b), so a non-paper cannot
    # arrive as a card *or* in `Also published today`. Counted rather than
    # dropped silently: a filter nobody can see the size of is the next thing
    # to go wrong quietly.
    apparatus = [it for it in items if is_journal_apparatus(it)]
    if apparatus:
        items = [it for it in items if not is_journal_apparatus(it)]
    run.count("journal_apparatus_dropped", len(apparatus))

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

    # Nobody is checking before this goes out, so the doubtful ones are held
    # rather than published (M2-2). Withholding leaves a hole in the day and
    # that is the intended outcome — a slot filled with something we are unsure
    # of is worth less than a shorter issue, and the reader cannot tell the two
    # apart. Near-misses are collected here too; they cost the issue nothing and
    # they are the labelling queue.
    from . import held as held_queue

    suspicions = []
    for it in journal_taken:
        s = held_queue.inspect(it, "journal", selected=True, floor=policy.arxiv_floor)
        if s:
            suspicions.append(s)
    for it in arxiv_taken:
        s = held_queue.inspect(it, "arxiv", selected=True, floor=policy.arxiv_floor)
        if s:
            suspicions.append(s)
    for it in arxiv_pool:
        if it in arxiv_taken:
            continue
        s = held_queue.inspect(it, "arxiv", selected=False, floor=policy.arxiv_floor)
        if s:
            suspicions.append(s)

    withheld_keys = {s.work_key for s in suspicions if s.kind == held_queue.WITHHELD}
    journal_taken = [it for it in journal_taken if it.work_key not in withheld_keys]
    arxiv_taken = [it for it in arxiv_taken if it.work_key not in withheld_keys]

    selected = journal_taken + arxiv_taken
    selected.sort(key=lambda it: (-it.scores.headline, -it.scores.relevance, it.work_key))

    held_queue.record(run.metrics.date, suspicions, published=len(selected))
    warning = held_queue.over_warn_threshold(len(selected), len(withheld_keys))
    if warning:
        run.error(warning)
    run.count("held_withheld", len(withheld_keys))
    run.count("held_near_miss", len(suspicions) - len(withheld_keys))

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


def stage_issue(run: Run, d: date, use_llm: bool = True) -> Issue:
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

    # An issue is immutable once published (`Issue` docstring, PRD §5.2), and a
    # selection-policy change does not reach back through it. Re-running
    # 2026-08-11 after V1-1 cut its list from 24 to 18 and left 18 items in
    # `content/items/` that no issue referenced — published papers with nowhere
    # to have been published.
    #
    # The list is restored to what it was, not merged with what the new policy
    # would pick: a union grows a past issue every time the day is re-run, which
    # is the same violation from the other direction. The Items themselves are
    # still refreshed — a status change or a recovered abstract updates the
    # record — but which papers that day carried is settled.
    #
    # An unreadable item that has since gained an abstract is the one thing that
    # may join, because that promotion is what `Also published today` exists to
    # allow and it changes the item's presentation rather than the day's list.
    prior_issue = store.load_issue(d)
    if prior_issue:
        restored: list[Item] = []
        for work_key in prior_issue.items:
            item = next((it for it in publish if it.work_key == work_key), None)
            if item is None:
                item = store.load_item(work_key)
            if item is not None:
                restored.append(item)
        promoted = [
            it for it in publish
            if it.work_key in set(prior_issue.unreadable) and has_abstract(it)
        ]
        publish = restored + [
            it for it in promoted if it.work_key not in {r.work_key for r in restored}
        ]

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
    # The synthesis layer. Built after the publish list is settled, because it
    # is a statement about what this issue contains — and failing softly,
    # because a day's issue must not be lost to a missing LLM key or a network
    # blip in a section that is an addition to it.
    synthesis = None
    try:
        from .synthesis import build as build_synthesis

        synthesis = build_synthesis(d, publish, len(unreadable))
    except Exception as e:  # noqa: BLE001
        run.error(f"synthesis: {type(e).__name__}: {e}")

    # One call per issue, and never fatal — see `pipeline/summarize/headline.py`
    # for why this is the most carefully fenced LLM call in the pipeline.
    #
    # ★ The threshold now chooses the **shape**, not whether a line exists
    # (0Z-B, B0). With a standout paper the line is about that paper; without
    # one it is about the day's papers, because a day of evenly solid work is
    # not a day with nothing to say. `headline_form` carries the measurement
    # that forced the change.
    from .score.headline import best_candidate, headline_form

    lead = headline_item or best_candidate(publish)
    form = headline_form(publish)

    headline_text, headline_basis = (None, None)
    if lead is not None:
        from .summarize.headline import write_headline

        headline_text, headline_basis = write_headline(
            lead, use_llm=use_llm, others=publish if form == "day" else None
        )
        run.count(f"headline_{headline_basis.split(':')[0]}", 1)
        run.metrics.stages["headline.form"] = form
        if not headline_basis.startswith("llm"):
            run.error(f"headline: {headline_basis}")

    issue = Issue(
        date=d,
        headline=Headline(
            # True whenever a line was written. Before 0Z-B this tracked "an
            # item cleared the bar", which is now `form == "lead"` and is
            # recorded in the run's metrics rather than in the issue.
            present=headline_text is not None,
            work_key=lead.work_key if lead else None,
            line=headline_text,
            basis=headline_basis,
        ),
        # Written as it always was, and read by nobody: see the field's own
        # comment in `models.py`. `Issue.is_quiet` is what the renderers use.
        # Still keyed on the threshold rather than on whether a line was
        # written, so the field keeps the one meaning it has always had
        # ("no item cleared the bar") instead of acquiring a third (0Z-B).
        quiet_day=headline_item is None,
        scan_meta=scan,
        items=sorted(it.work_key for it in publish),
        unreadable=sorted(it.work_key for it in unreadable),
        status_changes=status_changes,
        synthesis=synthesis,
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
    """Journals we actually query — `include: true` only (phase 0k, X0-1).

    This counted every entry in `journals.yaml`, which is 159 against the 96 we
    collect from: the other 63 are kept with `include: false` so the review that
    excluded them stays auditable. Every issue from phase 0 to 0j therefore said
    159, overstating our scope by 65%.

    **Issues already published keep 159.** They are immutable, and rewriting
    them to correct a number would be a worse lie than the number. The archive
    will hold both values, which is the honest state: the figure describes what
    the pipeline did on the day it ran.
    """
    from .config import journals_vocab

    return sum(
        1 for s in (journals_vocab().get("sources") or []) if s.get("include", True)
    )


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
    # The email edition, from the same render. Written beside the preview so
    # the two can never be produced from different inputs, and so a diff
    # between them is always a formatting diff.
    from .render.preview import email_subject, write_email

    write_email(issue, items, run.dir / "email.html", unreadable=unreadable)
    run.metrics.timing.setdefault("email_subject", 0.0)
    setattr(run.metrics, "email_subject", email_subject(issue))
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
    _guard(run, "issue", lambda: stage_issue(run, d, use_llm=use_llm))
    _guard(run, "preview", lambda: stage_preview(run, d))

    # ★ `uc run` gets the same verdict check `uc daily` has (0U, U4).
    #
    # It had none. `uc daily` refuses a day whose stages did not hold up, and
    # `uc run` wrote the issue regardless — so the one command with fewer
    # safeguards was the one a person reaches for when something has already
    # gone wrong and they are debugging it by hand.
    #
    # This does **not** record an outcome or send anything; `uc run` is not the
    # scheduled path and must not start writing the run log. It states the
    # verdict so the operator sees it, and names the reasons.
    from .outcome import looked

    ok, reasons = looked(run)
    if not ok:
        for reason in reasons:
            run.error(f"outcome: {reason}")
        print(
            "\n[NOT PUBLISHABLE] this run would not have been published:\n  - "
            + "\n  - ".join(reasons)
            + "\n  The issue file was still written; `uc daily` would have "
              "refused it. Do not treat it as a day's output."
        )

    run.save()
    return run
