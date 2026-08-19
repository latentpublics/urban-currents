"""``uc review --label relevance`` — the Q1b labelling pass.

This is not only a measurement instrument. The 150 labels it collects are also
**the training set for a classifier that does not exist yet** (roadmap §2.2,
§2.3): the one that answers "is this the *kind* of paper we cover?" — a question
nothing in the pipeline currently answers, and the reason the journal path ranks
on a placeholder.

Three design choices follow from that, and they matter more than they look:

1. **Stratified by source.** 15 arXiv + 15 journal a day, drawn from the
   *candidate pool* rather than the published 24. precision@10 measures the
   ranking, so sampling only what already cleared the publication cut would
   measure the cut instead. And one blended precision hides which of the two
   entry paths is failing.
2. **Every drop carries a reason.** ``n`` (not urban research) is a classifier
   error. ``q`` (urban research, but not our kind) is an unanswered coverage
   question. A single precision number mixes the two, and a drop with no reason
   cannot train anything.
3. **The stored row is the training example.** Everything needed to reproduce
   and learn from a judgement is written with it.

150 labels cannot be collected twice, so the format is fixed before any are.

The full review mode (``uc review --date``) is untouched: Q4 depends on its
timing, and it is not this module's business.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

from . import paths, store
from .config import cfg
from .metrics import Run, utcnow
from .models import Item

# Five verdicts and a skip. `w` was one label doing two jobs (M1).
#
# The 0k report read the 15 `drop_weak` rows on a topic/method axis and YJUN
# corrected it: **topic mismatch is not what `drop_weak` means** — a paper
# outside the field gets `n`, and one inside the field but not our kind gets
# `q`. `drop_weak` is for a paper squarely in scope whose *work* is thin, and
# that thinness comes in two kinds:
#
#   `m`  the method is weak     — how the work was done
#   `r`  the argument is weak   — what the work claims
#
# **`r` was called `drop_weak_results` until 0Q, and the name was wrong.** The
# labeller who made all fifteen judgements corrected it: what was meant is that
# the *claim the paper wants to make* is too narrow, or too thin for what was
# measured — and that **is** visible in an abstract, "초록으로 판단하기엔
# 부족하지만, 그래도 판단해볼 수 있는" — imperfectly, but visibly.
#
# That correction matters more than a rename. The old name carried a claim that
# this axis is **unlearnable from an abstract**, and 0P §5 acted on it by
# dropping these rows from a gate evaluation, which lifted journal precision@10
# from 0.660 to 0.700. The premise has been withdrawn by the person who
# supplied it, so the rows stay in and **Q1b is 0.600 / 0.660** (D204).
#
# Both are abstract-visible. They are two different axes, not a visible one and
# an invisible one: `m` is how the work was done, `r` is what it claims.
LABEL_KEYS: dict[str, str] = {
    "k": "keep",
    "n": "drop_not_urban",
    "q": "drop_not_our_kind",
    "m": "drop_weak_method",
    # The key stays `r` — it reads as the r of "argument" as easily as of
    # "results", and changing a keystroke people have in their fingers to record
    # a rename would cost a mislabel to save nothing.
    "r": "drop_weak_arguments",
    "s": "skip",
}

# `w` is retained as an input alias only, and it is deliberately NOT in
# LABEL_KEYS: a key that still produces the merged label would let the split be
# undone by muscle memory. Typing it asks which kind rather than accepting it.
LEGACY_WEAK_KEY = "w"

# The historical merged label. Rows written before M1 keep it; nothing writes it
# any more. It stays in DROP_LABELS so old files still aggregate correctly.
LEGACY_WEAK_LABEL = "drop_weak"

# `drop_weak_results` was renamed to `drop_weak_arguments` in 0Q (R1). The
# migration moved every row this repo holds, but the old string stays readable
# for the same reason `drop_weak` does: an export taken before the rename, or a
# file restored from a backup, must still aggregate rather than silently drop
# out of `weak`. **Nothing writes it.**
LEGACY_RESULTS_LABEL = "drop_weak_results"

WEAK_LABELS = (
    "drop_weak_method",
    "drop_weak_arguments",
    LEGACY_RESULTS_LABEL,
    LEGACY_WEAK_LABEL,
)

DROP_LABELS = ("drop_not_urban", "drop_not_our_kind") + WEAK_LABELS


def is_weak(label: str) -> bool:
    """Any of the three weak labels.

    Aggregates group them: precision@k is unchanged by the split, because they
    are all still drops. **The split is diagnostic information, not a change to
    the metric** — reporting a different precision after a relabelling would mean
    the metric had been moved rather than measured.

    That cuts both ways, and 0P got it wrong in the other direction: it removed
    the `results` rows from a gate evaluation on the grounds that they were
    unlearnable, and reported the higher number that followed. Neither the split
    nor the rename moves Q1b. It is **0.600 arXiv / 0.660 journal**.
    """
    return label in WEAK_LABELS

# Classifier score bands for the precision table (P5). Narrow at the top because
# that is where the selection threshold could plausibly move to, and where a
# handful of labels changes the answer.
SCORE_BANDS = ((0.95, 1.01), (0.90, 0.95), (0.70, 0.90), (0.35, 0.70))

# A day whose top-k is only partly labelled cannot produce a precision figure,
# only a guess with the same shape as one. `scripts/journal_gate.py` has drawn
# that line at 8 of 10 since N4; `precision_at_k` did not, and averaged a
# three-label day into the mean with the same weight as a full one — so a thin
# day moved the headline number while looking like evidence.
#
# Days below the bar are counted and named in `unmeasured_days` rather than
# dropped silently. **A day we could not measure is not a day that scored
# badly**, and the two have to stay tellable apart.
MIN_TOP_K_COVERAGE = 8

LABEL_PROMPT = (
    "   [k]eep / [n]ot urban / not our kind [q] / weak [m]ethod / weak a[r]gument"
    " / [s]kip: "
)
# `m` and `r` are both about a paper squarely in scope, so the boundary between
# them has to be readable on screen or the distinction is only in this file.
# **`m` is how the work was done; `r` is what it claims.** The wording of both
# was rewritten together in 0Q, because sharpening one without the other just
# moves the ambiguity.
LABEL_LEGEND = (
    "  k  keep — worth publishing as a card\n"
    "  n  not urban research at all            (classifier error)\n"
    "  q  urban research, not the kind we cover (qualitative case study,\n"
    "                                           theory, policy commentary)\n"
    "  m  our kind, but the METHOD is weak     — HOW it was done: thin data,\n"
    "                                           no baseline, n too small to\n"
    "                                           support what is claimed\n"
    "  r  our kind, but the ARGUMENT is weak   — WHAT it claims: the claim it\n"
    "                                           wants to make is too narrow,\n"
    "                                           or too thin for what was\n"
    "                                           measured\n"
    "  s  skip — undecided, offer it again next time"
)

# Shown once when someone types the old key.
LEGACY_WEAK_HINT = (
    "   'w' is now two labels — [m] weak method, [r] weak results. "
    "Which was it?"
)


# Label files whose sampling is a ranked top-N per source, which is what
# precision@k is defined over. Anything else is a different experiment wearing
# the same row shape.
RANKED_FACETS = frozenset({"relevance"})

# Label files sampled some other way. They may not be pooled with the ranked
# ones, and `precision_at_k` refuses them outright rather than returning a
# number that looks fine.
#
# Five frames now, and they answer five questions:
#   relevance      ranked top-N per source  — what precision@k is defined over
#   affinity_probe band-stratified over canon affinity
#   code_probe     band-stratified over relevance, among code-bearing arXiv
#   subfield_check five per subfield, over the four the scope gate excludes
#   held_review    whatever a suspicion rule stopped — drawn by rule, not rank
#
# Any two of them concatenated give a figure that looks reasonable and means
# nothing, which is why the guard refuses by name rather than hoping nobody
# points a summariser at the wrong file (phase 0L, N2).
PROBE_FACETS = frozenset(
    {"affinity_probe", "code_probe", "subfield_check", "held_review"}
)

# What each file's rows must declare. A row carries its own frame so a file that
# was mixed by hand — `cat a.jsonl >> b.jsonl` — is still detectable afterwards.
SAMPLING_OF_FACET = {
    "relevance": "ranked_top_n",
    "affinity_probe": "band_stratified",
    "code_probe": "code_stratified",
    # Drawn by subfield, five from each of the four the scope gate excludes.
    # Not a ranking, not a band — a fourth question with a fourth frame, and
    # pooling it with any of the others would average two different populations.
    "subfield_check": "subfield_check",
    # ★ The held queue (0P, Q4). `uc review --pending` used to write its
    # judgements straight into `relevance.jsonl` with `rank=0`, which is the
    # pooling this whole registry exists to prevent — and worse than the usual
    # case, because `precision_at_k` sorts a day by rank and takes the head, so
    # every held judgement would have sorted **above rank 1** and occupied the
    # top of its day's top-ten window. The queue is drawn by which rule stopped
    # an item, not by where the ranking put it; it is its own question.
    #
    # It had not fired yet: `relevance.jsonl` holds no rank-0 row, so nobody had
    # run `--pending` to the end before this was caught.
    "held_review": "held_review",
}


class LabelSetMisuse(RuntimeError):
    """Raised when a probe label set is used where a ranked one is required."""


def can_record(row: dict) -> bool:
    """Could a judgement on this row be written?

    **The invariant this hotfix installs: anything shown to a person must be
    recordable.** Three times a labelling session has asked for judgements it
    could not store, and every time the write path required something the
    display path did not — so the screen looked fine while the answers went
    nowhere.

    Checking it is cheap and it belongs *before* the first question, so a row
    that cannot be recorded is never shown rather than shown and dropped.
    """
    return bool(row.get("work_key")) and bool(row.get("date"))


class LabelWriteFailed(RuntimeError):
    """A person answered and the answer did not reach disk.

    Deliberately loud. Three times now a labelling session has dropped
    judgements without a word on screen, and each time the person who answered
    found out only when the file was missing. **A session that cannot record an
    answer must stop**, so the next twenty answers are not given to a hole.
    """


def _weak_total(reasons) -> int:
    """The three weak labels counted as one.

    Aggregates report `weak` as a single number so the split cannot move a
    published metric: precision@k is defined over keep-vs-drop, and all three
    are drops. The breakdown rides alongside as `weak_detail`.
    """
    return sum(reasons.get(name, 0) for name in WEAK_LABELS)


def _weak_detail(reasons) -> dict:
    """The breakdown under `weak`.

    `arguments` counts the old `drop_weak_results` string too, so a file written
    before the 0Q rename reports under the name the category actually has rather
    than vanishing from the breakdown while still counting in the total.
    """
    return {
        "method": reasons.get("drop_weak_method", 0),
        "arguments": (
            reasons.get("drop_weak_arguments", 0)
            + reasons.get(LEGACY_RESULTS_LABEL, 0)
        ),
        "unsplit": reasons.get(LEGACY_WEAK_LABEL, 0),
    }


def _ask_label(prompt, printer) -> Optional[str]:
    """One keystroke, or None if the session was stopped.

    The old `w` is caught rather than accepted: it was one key for two verdicts,
    and silently mapping it to either would put a guess in the label file. It
    re-asks instead, which costs one keystroke on the rows where the distinction
    is the whole point.
    """
    while True:
        answer = (prompt(LABEL_PROMPT) or "").strip().lower()
        if answer in ("quit", "exit"):
            return None
        key = answer[:1]
        if key in LABEL_KEYS:
            return key
        if key == LEGACY_WEAK_KEY:
            printer(LEGACY_WEAK_HINT)
            continue
        return "s"


def labels_path(facet: str = "relevance") -> Path:
    paths.LABELS.mkdir(parents=True, exist_ok=True)
    return paths.LABELS / f"{facet}.jsonl"


def load_labels(facet: str = "relevance") -> list[dict]:
    """Read one label file, and refuse it if it holds rows drawn two ways.

    The write guard stops the pipeline from mixing them; this stops a file that
    was mixed some other way — a hand-edit, a `cat a.jsonl >> b.jsonl` — from
    being summarised as though it were one sample.
    """
    path = labels_path(facet)
    if not path.exists():
        return []
    expected = sampling_of(facet)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("sampling", "ranked_top_n") != expected:
            raise LabelSetMisuse(
                f"{path} contains a {row.get('sampling')!r} row but is a "
                f"{expected!r} label set — the two cannot be summarised together"
            )
        out.append(row)
    return out


def labelled_keys(facet: str = "relevance") -> set[tuple[str, str]]:
    """(date, work_key) pairs already labelled — the basis of resuming."""
    return {(r.get("date", ""), r.get("work_key", "")) for r in load_labels(facet)}


def sampling_of(facet: str) -> str:
    """How a label file was drawn. The one fact that decides what it can measure."""
    if facet in SAMPLING_OF_FACET:
        return SAMPLING_OF_FACET[facet]
    if facet in PROBE_FACETS:
        return "band_stratified"
    return "ranked_top_n"


def append_labels(facet: str, rows: Iterable[dict]) -> int:
    """Append judgements, refusing rows drawn a different way than the file.

    The guard is on the write, not only on the read. Two files with the same row
    shape are one careless append away from becoming one file, and by the time a
    ranked row sits inside the probe there is nothing left to detect it with —
    the mixing is invisible in the data and shows up only as a precision figure
    that is quietly wrong.
    """
    expected = sampling_of(facet)
    checked = []
    for row in rows:
        found = row.get("sampling", "ranked_top_n")
        if found != expected:
            raise LabelSetMisuse(
                f"refusing to write a {found!r} row into {facet!r} "
                f"({expected!r}); these are different experiments and their "
                f"labels are not interchangeable"
            )
        checked.append(row)

    path = labels_path(facet)
    n = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in checked:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def append_one(facet: str, row: dict) -> int:
    """Write one judgement, now.

    **A session used to collect its rows in a list and write them once at the
    end.** One exception, one Ctrl-C, one guard raised on the last item, and the
    whole sitting was gone — which is what happened to a completed
    `code_probe` pass: thirty judgements made, thirty judgements lost, and the
    prompt had told the labeller "everything answered so far is saved".

    This is the same shape as the review timer in 0i, which only recorded
    sessions that ran to completion and so dropped precisely the sessions of
    people who were busy. That was moved into a `finally`. **Labels need more
    than that: a timing can be measured again and a person's judgement cannot.**

    JSONL is append-only and there is no reason to batch. One line per verdict,
    flushed before the next question is asked.
    """
    return append_labels(facet, [row])


def assert_writable(facet: str, sampling: str) -> None:
    """Check the sampling frame **before** the first item is shown.

    The guard did its job and refused rows from the wrong frame — after asking
    all thirty questions. The same fact was knowable before the first one, and
    finding out at the end costs a session; finding out at the start costs
    nothing.
    """
    expected = sampling_of(facet)
    if sampling == expected:
        return
    raise LabelSetMisuse(
        f"this session writes {sampling!r} rows but {facet!r} is a {expected!r} "
        f"label set — nothing was asked and nothing was written. "
        f"{_suggest_session(facet)}"
    )


def _suggest_session(facet: str) -> str:
    """Say what to run instead, rather than only what went wrong."""
    return {
        "relevance": "Use `uc review --label relevance`.",
        "affinity_probe": "Use `uc review --label affinity`.",
        "code_probe": "Use `uc review --label code_probe`.",
        "held_review": "Use `uc review --pending`.",
        "subfield_check": "Use `uc review --label subfield_check`.",
    }.get(facet, f"No session writes {facet!r}; the known ones are relevance, "
                 f"affinity and code_probe.")


def item_source(item: Item) -> str:
    from .run_stages import _is_whitelist_journal

    return "journal" if _is_whitelist_journal(item) else "arxiv"


def stratified_sample(
    d: date, per_source: int = 15, threshold: Optional[float] = None
) -> list[tuple[Item, str, int]]:
    """The day's labelling sample: top-N per source from the candidate pool.

    Returns (item, source, rank_within_source). Ranks are per source because
    precision@10 is reported per source.
    """
    from .run_stages import journal_rank_score
    from .stages import read_stage

    run = Run.for_date(d)
    # `labeling_pool` is the classify pool with summaries attached (written by
    # `uc prepare-labeling`). Fall back to `classify` — the raw candidate pool —
    # so the sample is still correct before preparation has run, just without
    # summaries on screen.
    items = (
        read_stage(run, "labeling_pool")
        or read_stage(run, "classify")
        or read_stage(run, "select")
    )
    thr = float(cfg("classifier.threshold", 0.35) if threshold is None else threshold)

    # An item with no abstract cannot be summarised and cannot be judged on
    # anything but its title. These labels are training data, and a label guessed
    # from a title is noise — fewer clean labels beat more dirty ones. Measured
    # 2026-08-05: 6 of 30 sampled items had no abstract, all journal-side
    # (several Elsevier titles expose none to OpenAlex).
    items = [it for it in items if (it.bibliography.abstract or "").strip()]

    journal = [it for it in items if item_source(it) == "journal"]
    arxiv = [
        it for it in items if item_source(it) == "arxiv" and it.scores.relevance >= thr
    ]

    journal.sort(key=lambda it: (-journal_rank_score(it), it.work_key))
    arxiv.sort(key=lambda it: (-it.scores.relevance, it.work_key))

    out: list[tuple[Item, str, int]] = []
    for source, pool in (("arxiv", arxiv), ("journal", journal)):
        for rank, item in enumerate(pool[:per_source], start=1):
            out.append((item, source, rank))
    return out


def _render(item: Item, source: str, rank: int, position: str) -> str:
    lines = [
        f"\n{position}  [{source} #{rank}]  score {item.scores.relevance:.3f}",
        f"  {item.bibliography.title}",
    ]
    loc = item.bibliography.primary_location.source_name
    if loc:
        lines.append(f"  {loc}")
    en = item.summary.en
    if en and en.what:
        # The summary is why this takes 15 minutes rather than 45.
        lines.append(f"\n  WHAT: {en.what}")
        if en.why:
            lines.append(f"  WHY : {en.why}")
    else:
        lines.append(f"\n  (no summary) {(item.bibliography.abstract or '')[:400]}")
    tags = [e.label for e in item.entities.methods + item.entities.data][:6]
    if tags:
        lines.append(f"  tags: {', '.join(tags)}")
    return "\n".join(lines)


def label_row(
    item: Item, source: str, rank: int, label: str, d: date, threshold: float
) -> dict[str, Any]:
    """One stored judgement — and one training example.

    Carries enough to reproduce the ranking that produced it and to learn from
    the outcome, because this file is the only source for the "our kind of
    paper?" classifier.
    """
    return {
        "date": str(d),
        "work_key": item.work_key,
        "source": source,
        "rank": rank,
        "label": label,
        "score": round(item.scores.relevance, 4),
        "title": item.bibliography.title,
        "has_summary": bool(item.summary.en and item.summary.en.what),
        "classifier_version": item.provenance.classifier_version,
        "model_version": cfg("classifier.model_version"),
        "threshold": threshold,
        # Stated on the row, not inferred from the filename: this is a ranked
        # top-N draw, which is what makes precision@k defined over it.
        "sampling": "ranked_top_n",
        "labelled_at": utcnow().isoformat(),
    }


def held_review_row(
    row: dict, label: str, item: Optional[Item] = None
) -> dict[str, Any]:
    """One judgement on a held item, carrying **why it was held**.

    That is the whole point of the file: these labels exist to tell us whether
    the rule that stopped an item was right, so `rule`, `kind` and the rule's
    own `detail` travel with the verdict. Without them the row is just another
    opinion about a paper and answers nothing about the queue.

    There is no `rank`, because the queue has no ranking — an item is here
    because a rule flagged it.

    ## The held row is the source, and the item is enrichment (0Q hotfix, G2)

    This used to take `item` first and require it. **A withheld item was never
    published, so by definition it has no file in `content/items/`** — 116 of
    the 118 withheld rows have none — and the caller skipped every row where the
    lookup came back empty. Twenty-five judgements were asked for and thrown
    away.

    The held row already carries `work_key`, `date`, `rule`, `kind`, `detail`,
    `title` and `source`. **Everything a label needs is in it.** The item adds a
    relevance score and a summary flag when it happens to exist, and adds
    nothing that the row cannot do without.

    `shown` records which of the two the labeller actually had, for the same
    reason `subfield_check` records whether there was a summary: **a judgement
    made on less must not be indistinguishable from one made on more.**
    """
    out: dict[str, Any] = {
        "date": row["date"],
        "work_key": row["work_key"],
        "source": row.get("source") or "journal",
        "label": label,
        "title": row.get("title", ""),
        "rule": row.get("rule"),
        "kind": row.get("kind"),
        "why_held": row.get("detail"),
        "score": row.get("score"),
        "shown": "held_row",
        "model_version": cfg("classifier.model_version"),
        "sampling": "held_review",
        "labelled_at": utcnow().isoformat(),
        # Drawn by rule, never by rank. No precision@k is defined over it.
        "not_for_precision_at_k": True,
    }
    if item is not None:
        out["shown"] = "held_row+item"
        out["title"] = item.bibliography.title or out["title"]
        out["score"] = round(item.scores.relevance, 4)
        out["has_summary"] = bool(item.summary.en and item.summary.en.what)
        out["classifier_version"] = item.provenance.classifier_version
    return out


def run_labeling_session(
    d: date,
    facet: str = "relevance",
    top: int = 30,
    prompt=None,
    threshold: Optional[float] = None,
    printer=print,
) -> dict[str, Any]:
    """Label one day. Resumable: anything already labelled for this date is skipped."""
    if prompt is None:  # pragma: no cover - interactive
        def prompt(message: str) -> str:
            return input(message).strip().lower()

    per_source = max(1, top // 2)
    thr = float(cfg("classifier.threshold", 0.35) if threshold is None else threshold)
    sample = stratified_sample(d, per_source=per_source, threshold=thr)
    if not sample:
        printer(f"no classified candidates for {d}; run `uc classify --date {d}` first")
        return {"labelled": 0, "remaining": 0, "counts": {}}

    done = labelled_keys(facet)
    todo = [(it, s, r) for (it, s, r) in sample if (str(d), it.work_key) not in done]

    printer(f"\nlabelling {d} — {len(todo)} of {len(sample)} remaining")
    if len(todo) < len(sample):
        printer(f"({len(sample) - len(todo)} already labelled; resuming)")
    printer(LABEL_LEGEND)
    printer("  type 'quit' to stop — everything answered so far is saved")

    assert_writable(facet, sampling_of(facet))

    started = time.monotonic()
    n = 0
    counts: Counter = Counter()
    stopped = False

    for i, (item, source, rank) in enumerate(todo, start=1):
        printer(_render(item, source, rank, f"{i}/{len(todo)}"))
        key = _ask_label(prompt, printer)
        if key is None:
            stopped = True
            break
        label = LABEL_KEYS[key]
        counts[label] += 1
        if label == "skip":
            continue
        # Written before the next question is asked, not collected for the end.
        n += append_one(facet, label_row(item, source, rank, label, d, thr))

    elapsed = time.monotonic() - started
    run = Run.for_date(d)
    run.metrics.timing["label_s"] = round(
        run.metrics.timing.get("label_s", 0.0) + elapsed, 1
    )
    run.save()

    remaining = len(todo) - n
    printer(
        f"\nwrote {n} labels to {labels_path(facet)} in {elapsed / 60:.1f} min "
        f"({dict(counts)})"
    )
    if remaining > 0:
        printer(
            f"{remaining} left for {d} — re-run the same command to continue"
            + (" (stopped early)" if stopped else "")
        )
    return {
        "labelled": n,
        "remaining": max(0, remaining),
        "counts": dict(counts),
        "stopped_early": stopped,
    }


# --------------------------------------------------------------------------
# Aggregation — always per source (roadmap §2.3)
# --------------------------------------------------------------------------


def precision_at_k(facet: str = "relevance", k: int = 10) -> dict:
    """Q1b, reported per source with the drop reasons kept apart.

    A blended precision@10 hides which entry path is failing, and merging the
    drop reasons hides whether the problem is the classifier (`n`) or an
    unanswered coverage question (`q`). Both distinctions are the reason this
    tool exists, so neither is collapsed here.

    **Probe label sets are refused, not merged.** `precision@k` means "of the top
    k the ranking offered, how many were worth publishing", so it is only defined
    over a ranked top-N sample. `affinity_probe.jsonl` is drawn across affinity
    bands on purpose — it deliberately over-samples the bottom, which is exactly
    what would make a precision figure computed over it meaningless. The rows
    have the same shape, which is why this refuses by name instead of hoping
    nobody points it here.
    """
    if facet in PROBE_FACETS:
        raise LabelSetMisuse(
            f"{facet!r} is a band-stratified probe, not a ranked sample; "
            f"precision@k is undefined over it. Use `probe_summary({facet!r})`."
        )
    if facet not in RANKED_FACETS:
        raise LabelSetMisuse(
            f"{facet!r} is not a known ranked label set "
            f"(expected one of {sorted(RANKED_FACETS)})"
        )
    # Newest judgement per item. A re-judged row is appended rather than edited
    # (M1), so without this the same paper would be counted twice — once under
    # the old verdict and once under the new one — and n would drift upward
    # every time someone corrected something.
    rows = superseded(load_labels(facet))
    if not rows:
        return {
            "n_labels": 0,
            "k": k,
            "days_labelled": 0,
            "by_source": {},
            "note": "no labels yet — run `uc review --label relevance --date …`",
        }

    by_source: dict[str, dict] = {}
    # The same default on both sides. They disagreed: the group set used
    # `r.get("source", "unknown")` and the filter used `r.get("source")`, so a
    # row with no `source` created an "unknown" group that then matched nothing
    # and divided by zero. A malformed row should land in a named group and be
    # visible, not take the metric down.
    def _source_of(row: dict) -> str:
        return row.get("source") or "unknown"

    for source in sorted({_source_of(r) for r in rows}):
        srows = [r for r in rows if _source_of(r) == source]
        by_day: dict[str, list[dict]] = {}
        for r in srows:
            by_day.setdefault(r.get("date", ""), []).append(r)

        per_day = []
        unmeasured: list[dict] = []
        for _day, day_rows in sorted(by_day.items()):
            day_rows.sort(key=lambda r: r.get("rank", 0))
            topk = day_rows[:k]
            # Coverage is how much of the top-k window carries a judgement, not
            # how many rows the day has: a day labelled only below rank 10 tells
            # us nothing about the top 10.
            covered = sum(1 for r in day_rows if (r.get("rank") or 0) and r["rank"] <= k)
            if covered < min(MIN_TOP_K_COVERAGE, k):
                unmeasured.append({"date": _day, "labelled_in_top_k": covered})
                continue
            if topk:
                per_day.append(sum(1 for r in topk if r["label"] == "keep") / len(topk))

        # Precision at every depth, not just at k. The daily list gives each
        # path a fixed number of slots, so what decides whether the path is
        # usable is how far down its ranking precision survives — a path that
        # holds 1.0 to rank 4 and 0.5 by rank 12 is not failing at ranking, it
        # is being asked for more items than it has.
        depth: list[float] = []
        max_depth = max((len(v) for v in by_day.values()), default=0)
        for pos in range(1, max_depth + 1):
            at_pos = []
            for day_rows in by_day.values():
                head = sorted(day_rows, key=lambda r: r.get("rank", 0))[:pos]
                if len(head) == pos:
                    at_pos.append(sum(1 for r in head if r["label"] == "keep") / pos)
            if at_pos:
                depth.append(round(sum(at_pos) / len(at_pos), 4))

        # Precision by classifier score band (P5). This is the table that
        # answers "where should the threshold go" when enough labels exist;
        # a single precision@10 cannot, because it averages over a range in
        # which the classifier's confidence varies by 60 points.
        #
        # The journal path has no bands: every whitelist article scores exactly
        # 1.0 by membership (N4), so its distribution is one point and the drop
        # reasons are the only thing that varies. Reported as such rather than
        # rendered as a table with one row.
        scores = [float(r.get("score", 0.0)) for r in srows]
        single_valued = len(set(round(s, 4) for s in scores)) <= 1
        bands: list[dict] = []
        if not single_valued:
            for low, high in SCORE_BANDS:
                in_band = [r for r in srows if low <= float(r.get("score", 0.0)) < high]
                if not in_band:
                    continue
                kept = sum(1 for r in in_band if r["label"] == "keep")
                band_reasons = Counter(
                    r["label"] for r in in_band if r["label"] in DROP_LABELS
                )
                bands.append({
                    "band": f"{low}-{high}" if high <= 1.0 else f">={low}",
                    "n": len(in_band),
                    "keep_rate": round(kept / len(in_band), 4),
                    "drop_reasons": {
                        "not_urban": band_reasons.get("drop_not_urban", 0),
                        "not_our_kind": band_reasons.get("drop_not_our_kind", 0),
                        "weak": _weak_total(band_reasons),
                    },
                    "weak_detail": _weak_detail(band_reasons),
                })

        reasons = Counter(r["label"] for r in srows if r["label"] in DROP_LABELS)
        n_drops = sum(reasons.values())

        def share(name: str) -> Optional[float]:
            return round(reasons.get(name, 0) / n_drops, 3) if n_drops else None

        by_source[source] = {
            "n_labels": len(srows),
            "days": len(per_day),
            "days_unmeasured": len(unmeasured),
            "unmeasured_days": unmeasured,
            f"precision_at_{k}": (
                round(sum(per_day) / len(per_day), 4) if per_day else None
            ),
            "per_day": [round(p, 4) for p in per_day],
            "score_bands": bands,
            "score_is_single_valued": single_valued,
            "precision_by_depth": depth,
            "depth_holding_0.7": max(
                (i + 1 for i, p in enumerate(depth) if p >= 0.7), default=0
            ),
            "keep_rate": round(
                sum(1 for r in srows if r["label"] == "keep") / len(srows), 4
            ),
            "drop_reasons": {
                "not_urban": reasons.get("drop_not_urban", 0),
                "not_our_kind": reasons.get("drop_not_our_kind", 0),
                "weak": _weak_total(reasons),
            },
            "weak_detail": _weak_detail(reasons),
            "drop_reason_share": {
                "not_urban": share("drop_not_urban"),
                "not_our_kind": share("drop_not_our_kind"),
                "weak": (
                    round(_weak_total(reasons) / n_drops, 3) if n_drops else None
                ),
            },
        }

    return {
        "n_labels": len(rows),
        "k": k,
        "days_labelled": len({r.get("date") for r in rows}),
        "by_source": by_source,
        "summaries_available": round(
            sum(1 for r in rows if r.get("has_summary")) / len(rows), 3
        ),
    }


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------

LABELING_POOL_STAGE = "labeling_pool"


def prepare_day(
    d: date,
    per_source: int = 15,
    threshold: Optional[float] = None,
    summarize: bool = True,
    client=None,
) -> dict[str, Any]:
    """Give every item in the day's labelling sample a summary.

    The published issue carries 24 items; the labelling sample is 30 drawn from
    a wider pool, so a handful each day would otherwise reach the labeller with
    no summary. That is not a cosmetic gap — labelling from abstracts is roughly
    three times slower, which is the difference between Q4 being measurable in
    15 minutes a day and not.

    Writes ``stages/labeling_pool.jsonl``, which ``stratified_sample`` prefers.
    """
    from .stages import read_stage, write_stage
    from .summarize.run import summarize_items

    run = Run.for_date(d)
    pool_items = read_stage(run, "classify")
    if not pool_items:
        return {"date": str(d), "status": "NO_CANDIDATES", "sample": 0}

    without_abstract = sum(
        1 for it in pool_items if not (it.bibliography.abstract or "").strip()
    )

    sample = stratified_sample(d, per_source=per_source, threshold=threshold)
    sample_keys = {it.work_key for it, _, _ in sample}

    # Summaries already produced for the published issue are reused rather than
    # regenerated — same cache key, but this avoids even the lookup.
    published = {it.work_key: it for it in read_stage(run, "summarize")}
    by_key = {it.work_key: it for it in pool_items}
    for key, done in published.items():
        if key in by_key and done.summary.en:
            by_key[key].summary = done.summary
            by_key[key].signals = done.signals
            by_key[key].provenance = done.provenance

    needs = [
        by_key[k]
        for k in sorted(sample_keys)
        if k in by_key and not (by_key[k].summary.en and by_key[k].summary.en.what)
    ]

    stats: dict[str, Any] = {"needed": len(needs), "summarized": 0}
    if summarize and needs:
        stats.update(summarize_items(needs, run, client=client))

    write_stage(run, LABELING_POOL_STAGE, list(by_key.values()))

    with_summary = sum(
        1 for k in sample_keys if by_key.get(k) and by_key[k].summary.en
        and by_key[k].summary.en.what
    )
    return {
        "date": str(d),
        "status": "OK",
        "sample": len(sample),
        "with_summary": with_summary,
        "missing_summary": len(sample_keys) - with_summary,
        # Recorded rather than hidden: it is a coverage bias in the label set.
        "candidates_without_abstract": without_abstract,
        **{k: v for k, v in stats.items() if k in ("needed", "summarized", "failures", "status")},
    }


# --------------------------------------------------------------------------
# Affinity probe (phase 0h, U0-2)
# --------------------------------------------------------------------------

AFFINITY_BANDS = ("high", "mid", "zero")


def affinity_pool(
    dates: list[date], exclude_labelled: bool = True, require_refs: bool = True
) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Unlabelled journal candidates with their canon affinity.

    **Journal path only.** `canon_affinity` needs a reference list and only 5%
    of arXiv items have one, so an arXiv probe would measure the absence of data
    rather than the signal. Stated here and in the probe file.

    **Items with no reference list at all are excluded** (`require_refs`), and
    that is not a detail. Measured over the five prepared days, 21 of the 82
    zero-affinity candidates have no references in our base — their affinity is
    zero because we hold nothing to score, not because they cite nothing
    canonical. Left in, a fifth of the zero band would be a coverage gap wearing
    the costume of a negative result, and a low keep rate there would read as
    "the signal works" when it measured nothing. The count is reported.
    """
    from .graph.citation import load_reference_base
    from .metrics import Run
    from .run_stages import _is_whitelist_journal, has_abstract
    from .stages import read_stage

    scripts_dir = str(paths.ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from journal_metrics import canon_affinity, canon_sets, cites_canon  # type: ignore

    foundation, _ = canon_sets()
    refs = {
        r["work_key"]: (r.get("referenced_works") or []) for r in load_reference_base()
    }
    already = {r["work_key"] for r in load_labels("relevance")} if exclude_labelled else set()
    already |= {r["work_key"] for r in load_labels("affinity_probe")}

    pool: dict[str, dict] = {}
    no_refs: set[str] = set()
    for d in dates:
        for item in read_stage(Run.for_date(d), "classify") or []:
            if not (_is_whitelist_journal(item) and has_abstract(item)):
                continue
            if item.work_key in already or item.work_key in pool:
                continue
            item_refs = refs.get(item.work_key, [])
            if not item_refs and require_refs:
                no_refs.add(item.work_key)
                continue
            pool[item.work_key] = {
                "item": item,
                "date": str(d),
                "canon_affinity": canon_affinity(item_refs, foundation),
                # The binary the probe says carries the signal. Stored beside
                # the continuous value rather than replacing it, because the
                # probe's own bands were drawn on the continuous one and a row
                # has to stay re-scorable against the sampling that produced it.
                "cites_canon": cites_canon(item_refs, foundation),
                "canon_hits": sum(1 for r in item_refs if r in foundation),
                "refs_total": len(item_refs),
            }
    # Returned, not logged: a pool smaller than the candidate count is a
    # population change, and the caller has to be able to report what it lost.
    return pool, {"no_references": sorted(no_refs)}


def affinity_bands(pool: dict[str, dict], high_cut: Optional[float] = None) -> dict:
    """Split the pool into high / mid / zero, and say where the cut came from.

    The positive values are heavily right-skewed — measured over the five
    prepared days, 53 positives running 1.03 to 28.8 with a median of 2.88 — so
    the interesting boundary is not the midpoint of the range. The upper third
    of the positive mass is where the values start to separate; below it they
    cluster between 2 and 3.6 and are not distinguishable by eye or by score.
    """
    positives = sorted(v["canon_affinity"] for v in pool.values() if v["canon_affinity"] > 0)
    if high_cut is None:
        high_cut = positives[int(0.66 * (len(positives) - 1))] if positives else 0.0

    bands: dict[str, list[str]] = {b: [] for b in AFFINITY_BANDS}
    for key, row in pool.items():
        value = row["canon_affinity"]
        band = "zero" if value == 0 else ("high" if value >= high_cut else "mid")
        bands[band].append(key)
    return {
        "high_cut": round(float(high_cut), 4),
        "high_cut_basis": "66th percentile of non-zero affinity in the pool",
        "positives": len(positives),
        "bands": bands,
        "sizes": {b: len(v) for b, v in bands.items()},
    }


def affinity_probe_sample(
    dates: list[date], per_band: int = 10, seed: int = 42
) -> tuple[list[tuple[Item, str, int]], dict]:
    """A band-stratified probe sample: `per_band` from high / mid / zero.

    The ranked sample answers "of the top the ranking offered, how many were
    good", which is precision@k. It cannot answer "does a high affinity actually
    mean keep", because it never shows the bottom of the distribution. This does
    the opposite on purpose — equal draws from each band — which is exactly why
    its labels must never be pooled with the ranked ones.
    """
    import random

    pool, excluded = affinity_pool(dates)
    spec = affinity_bands(pool)
    spec["excluded_no_references"] = len(excluded["no_references"])
    rng = random.Random(seed)

    picked: list[tuple[Item, str, int]] = []
    per_band_keys: dict[str, list[str]] = {}
    for band in AFFINITY_BANDS:
        keys = sorted(spec["bands"][band])
        rng.shuffle(keys)
        chosen = keys[:per_band]
        per_band_keys[band] = chosen
        for rank, key in enumerate(chosen, start=1):
            picked.append((pool[key]["item"], band, rank))

    spec["sampled"] = {b: len(v) for b, v in per_band_keys.items()}
    spec["pool_size"] = len(pool)
    spec["detail"] = {
        key: {k: v for k, v in pool[key].items() if k != "item"}
        for keys in per_band_keys.values()
        for key in keys
    }
    return picked, spec


def probe_row(
    item: Item, band: str, rank: int, label: str, detail: dict, venue_prior: Optional[float]
) -> dict[str, Any]:
    """One probe judgement, carrying everything needed to re-score the signal.

    The band and the affinity are stored with the label so this file alone can
    re-evaluate `canon_affinity` later without recomputing a pool that will have
    changed by then.
    """
    return {
        "work_key": item.work_key,
        "title": item.bibliography.title,
        "date": detail.get("date", ""),
        "label": LABEL_KEYS.get(label, label),
        "band": band,
        "rank_in_band": rank,
        "canon_affinity": detail.get("canon_affinity"),
        "canon_hits": detail.get("canon_hits"),
        "refs_total": detail.get("refs_total"),
        "venue_prior": venue_prior,
        "source": "journal",
        "sampling": "band_stratified",
        "labelled_at": utcnow().isoformat(),
        # Stated in every row: this file is not a ranked sample and no
        # precision@k may be computed from it.
        "not_for_precision_at_k": True,
    }


def probe_summary(facet: str = "affinity_probe") -> dict[str, Any]:
    """Keep rate by affinity band — what a probe can answer and precision cannot.

    Deliberately not called `precision_at_k`. The question here is whether the
    signal separates, and the answer is a comparison of keep rates across bands,
    not a figure about the top of a ranking.
    """
    if facet in RANKED_FACETS:
        raise LabelSetMisuse(
            f"{facet!r} is a ranked sample; use `precision_at_k({facet!r})`"
        )
    rows = superseded(load_labels(facet))
    if not rows:
        return {
            "n_labels": 0,
            "by_band": {},
            "note": "no probe labels yet — run `uc review --label affinity`",
        }

    by_band: dict[str, dict] = {}
    for band in AFFINITY_BANDS:
        in_band = [r for r in rows if r.get("band") == band]
        if not in_band:
            continue
        kept = sum(1 for r in in_band if r.get("label") == "keep")
        reasons = Counter(r["label"] for r in in_band if r.get("label") in DROP_LABELS)
        by_band[band] = {
            "n": len(in_band),
            "keep_rate": round(kept / len(in_band), 4),
            "drop_reasons": {
                "not_urban": reasons.get("drop_not_urban", 0),
                "not_our_kind": reasons.get("drop_not_our_kind", 0),
                "weak": _weak_total(reasons),
            },
            "median_affinity": round(
                sorted(r.get("canon_affinity") or 0 for r in in_band)[len(in_band) // 2], 4
            ),
        }
    return {
        "n_labels": len(rows),
        "sampling": "band_stratified over canon_affinity — journal path only",
        "population_note": (
            "arXiv is excluded: canon_affinity needs a reference list and only "
            "5% of arXiv items have one, so an arXiv probe would measure missing "
            "data rather than the signal."
        ),
        "not_comparable_with": "relevance.jsonl (ranked top-N sampling)",
        "by_band": by_band,
    }


def venue_prior_map() -> dict[str, Optional[float]]:
    """`prestige_pct_in_subfield` by source id, from the 0g metrics run.

    Recorded on each probe row rather than recomputed later: the percentile is
    relative to the subfield population as it stood when the labels were taken,
    and that population moves.
    """
    path = paths.ROOT / "runs/journal_metrics.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        s["id"]: s.get("prestige_pct_in_subfield") for s in data.get("sources", [])
    }


def _render_probe(item: Item, band: str, detail: dict, position: str) -> str:
    """Same shape as the ranked renderer, minus the rank and the score.

    The band is deliberately *not* shown. The probe asks whether affinity
    predicts the judgement, so telling the labeller which band an item came from
    would be handing them the answer.
    """
    lines = [
        f"\n{'-' * 72}",
        f"  {position}  [{detail.get('date', '')}]  journal",
        f"  {item.bibliography.title}",
    ]
    en = item.summary.en
    if en and en.what:
        lines.append(f"\n  WHAT: {en.what}")
        if en.why:
            lines.append(f"  WHY : {en.why}")
    else:
        lines.append(f"\n  (no summary) {(item.bibliography.abstract or '')[:400]}")
    tags = [e.label for e in item.entities.methods + item.entities.data][:6]
    if tags:
        lines.append(f"  tags: {', '.join(tags)}")
    return "\n".join(lines)


def run_probe_session(
    dates: list[date],
    per_band: int = 10,
    facet: str = "affinity_probe",
    prompt=None,
    printer=print,
) -> dict[str, Any]:
    """Label the band-stratified affinity probe. Resumable, like the ranked pass.

    Writes to `runs/labels/affinity_probe.jsonl` and nowhere else. The file it
    writes cannot be fed to `precision_at_k` — see `LabelSetMisuse` — because the
    sampling here is equal draws per band, which is the opposite of a ranked
    top-N and would silently corrupt any precision figure it entered.
    """
    if facet in RANKED_FACETS:
        raise LabelSetMisuse(
            f"{facet!r} is the ranked label set; the probe must not write into it"
        )
    if prompt is None:  # pragma: no cover - interactive
        def prompt(message: str) -> str:
            return input(message).strip().lower()

    picked, spec = affinity_probe_sample(dates, per_band=per_band)
    if not picked:
        printer("no unlabelled journal candidates with references for those dates")
        return {"labelled": 0, "remaining": 0, "counts": {}, "spec": spec}

    # Summaries prepared for exactly these picks, if `uc prepare-probe` has run.
    prepared = {it.work_key: it for it in _load_probe_pool()}
    picked = [
        (prepared.get(it.work_key, it), band, rank) for (it, band, rank) in picked
    ]

    done = {r["work_key"] for r in load_labels(facet)}
    todo = [(it, b, r) for (it, b, r) in picked if it.work_key not in done]

    printer(
        f"\naffinity probe — {len(todo)} of {len(picked)} remaining "
        f"({spec['sampled']}, high cut {spec['high_cut']})"
    )
    if len(todo) < len(picked):
        printer(f"({len(picked) - len(todo)} already labelled; resuming)")
    printer("  journal path only — arXiv items have no reference lists to score")
    printer(LABEL_LEGEND)
    printer("  type 'quit' to stop — everything answered so far is saved")

    priors = venue_prior_map()
    assert_writable(facet, sampling_of(facet))

    started = time.monotonic()
    n = 0
    counts: Counter = Counter()
    stopped = False

    for i, (item, band, rank) in enumerate(todo, start=1):
        detail = spec["detail"][item.work_key]
        printer(_render_probe(item, band, detail, f"{i}/{len(todo)}"))
        key = _ask_label(prompt, printer)
        if key is None:
            stopped = True
            break
        if LABEL_KEYS[key] == "skip":
            counts["skip"] += 1
            continue
        counts[LABEL_KEYS[key]] += 1
        source_id = item.bibliography.primary_location.source_id
        n += append_one(
            facet, probe_row(item, band, rank, key, detail, priors.get(source_id))
        )

    
    elapsed = time.monotonic() - started
    printer(
        f"\nwrote {n} probe labels to {labels_path(facet)} in {elapsed / 60:.1f} min "
        f"({dict(counts)})"
    )
    remaining = len(todo) - n
    if remaining > 0:
        printer(
            f"{remaining} left — re-run the same command to continue"
            + (" (stopped early)" if stopped else "")
        )
    return {
        "labelled": n,
        "remaining": max(0, remaining),
        "counts": dict(counts),
        "stopped_early": stopped,
        "high_cut": spec["high_cut"],
        "sampled": spec["sampled"],
        "pool_size": spec["pool_size"],
    }


PROBE_POOL_FILE = "affinity_probe_pool.jsonl"


def probe_pool_path() -> Path:
    paths.LABELS.mkdir(parents=True, exist_ok=True)
    return paths.LABELS / PROBE_POOL_FILE


def prepare_probe(
    dates: list[date], per_band: int = 10, summarize: bool = True, client=None
) -> dict[str, Any]:
    """Summarise the probe's 30 items so they can be judged at the ranked pace.

    The ranked sample was prepared per day and only the 15+15 it drew were
    summarised. The probe draws from the whole unlabelled journal pool, so most
    of its picks arrive with an abstract and nothing else — roughly three times
    slower to judge, which is the difference between 30 labels in one sitting and
    30 labels spread over a week.

    Written to its own file, not into any stage output. The probe is a side
    experiment and must not alter what the pipeline would produce for these
    dates; `content/` and every stage file are left exactly as they were.
    """
    from .summarize.run import summarize_items

    picked, spec = affinity_probe_sample(dates, per_band=per_band)
    if not picked:
        return {"status": "NO_CANDIDATES", "sample": 0}

    items = [it for it, _, _ in picked]
    have = {it.work_key: it for it in _load_probe_pool()}
    for it in items:
        done = have.get(it.work_key)
        if done is not None and done.summary.en and done.summary.en.what:
            it.summary = done.summary
            it.signals = done.signals
            it.provenance = done.provenance

    needs = [it for it in items if not (it.summary.en and it.summary.en.what)]
    stats: dict[str, Any] = {"needed": len(needs), "summarized": 0}
    if summarize and needs:
        by_day: dict[str, list[Item]] = {}
        for it in needs:
            by_day.setdefault(spec["detail"][it.work_key]["date"], []).append(it)
        summarized = 0
        for day, group in sorted(by_day.items()):
            out = summarize_items(group, Run.for_date(date.fromisoformat(day)), client=client)
            summarized += int(out.get("summarized", 0))
        stats["summarized"] = summarized

    merged = {it.work_key: it for it in have.values()}
    merged.update({it.work_key: it for it in items})
    lines = [
        merged[k].model_dump_json(by_alias=True) for k in sorted(merged)
    ]
    probe_pool_path().write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
    )

    with_summary = sum(1 for it in items if it.summary.en and it.summary.en.what)
    return {
        "status": "OK",
        "sample": len(items),
        "with_summary": with_summary,
        "missing_summary": len(items) - with_summary,
        "bands": spec["sampled"],
        "high_cut": spec["high_cut"],
        **stats,
    }


def _load_probe_pool() -> list[Item]:
    path = probe_pool_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Item.model_validate_json(line))
    return out


# --------------------------------------------------------------------------
# Moving a labelling session between machines (phase 0j, W7)
# --------------------------------------------------------------------------

EXPORT_VERSION = "labeling-set@1"


def export_labeling_set(dates: list[date], out: Path) -> dict[str, Any]:
    """Everything needed to label a day, in one file.

    Labelling happens wherever YJUN is, and the pipeline's state for a day is
    spread across `runs/<id>/stages/*.jsonl`, the labelling pool, and the probe
    pool. Last time that move was a hand-built 8.4MB tar, which worked once and
    is not a procedure.

    Only what labelling reads is exported — the classify pool, the labelling
    pool, the summaries and the probe pool. Not the raw API responses, which are
    the bulk of a run directory and which nothing in a labelling session touches.
    """
    from .metrics import Run
    from .stages import read_stage

    payload: dict[str, Any] = {
        "version": EXPORT_VERSION,
        "exported_at": utcnow().isoformat(),
        "dates": [str(d) for d in dates],
        "days": {},
        "probe_pool": [it.model_dump(mode="json", by_alias=True) for it in _load_probe_pool()],
    }
    for d in dates:
        run = Run.for_date(d)
        day: dict[str, list] = {}
        for stage in ("classify", LABELING_POOL_STAGE, "summarize"):
            items = read_stage(run, stage) or []
            if items:
                day[stage] = [it.model_dump(mode="json", by_alias=True) for it in items]
        if day:
            payload["days"][str(d)] = day

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "path": str(out),
        "bytes": out.stat().st_size,
        "dates": payload["dates"],
        "stages": {
            d: {k: len(v) for k, v in day.items()} for d, day in payload["days"].items()
        },
        "probe_pool": len(payload["probe_pool"]),
    }


def import_labeling_set(path: Path) -> dict[str, Any]:
    """Write an exported set back into this machine's run directories.

    The round trip is the point: an export nobody has read back is a backup
    nobody has restored.
    """
    from .metrics import Run
    from .stages import write_stage

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != EXPORT_VERSION:
        raise ValueError(
            f"unknown export version {payload.get('version')!r}; expected {EXPORT_VERSION}"
        )

    restored: dict[str, dict[str, int]] = {}
    for day, stages in (payload.get("days") or {}).items():
        run = Run.for_date(date.fromisoformat(day))
        restored[day] = {}
        for stage, rows in stages.items():
            items = [Item.model_validate(r) for r in rows]
            write_stage(run, stage, items)
            restored[day][stage] = len(items)

    probe = [Item.model_validate(r) for r in (payload.get("probe_pool") or [])]
    if probe:
        probe_pool_path().write_text(
            "\n".join(it.model_dump_json(by_alias=True) for it in sorted(
                probe, key=lambda i: i.work_key
            )) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return {"dates": sorted(restored), "stages": restored, "probe_pool": len(probe)}


# --------------------------------------------------------------------------
# M1 — re-judging the rows that were labelled before the split
# --------------------------------------------------------------------------


def superseded(rows: list[dict]) -> list[dict]:
    """Collapse each (date, work_key) to its most recent judgement.

    The relabel session **appends** rather than edits: the standing rule is that
    label files are not modified, and the 15 `drop_weak` rows are evidence of
    what was judged before the split existed. So a re-judgement is a new row
    carrying `corrected_from`, and this is what makes the newest one count.

    Ordering is by `corrected_at` then `labelled_at`, both ISO strings, so a
    corrected row always sorts after the original it corrects. Rows without
    either keep file order, which is append order.
    """
    by_key: dict[tuple[str, str], dict] = {}
    for i, row in enumerate(rows):
        key = (row.get("date", ""), row.get("work_key", ""))
        stamp = (row.get("corrected_at") or "", row.get("labelled_at") or "", i)
        previous = by_key.get(key)
        if previous is None or stamp >= previous["_stamp"]:
            by_key[key] = {**row, "_stamp": stamp}
    return [{k: v for k, v in r.items() if k != "_stamp"} for r in by_key.values()]


def weak_rows_to_rejudge(facet: str = "relevance") -> list[dict]:
    """The rows still carrying the unsplit label, newest judgement first.

    Only `drop_weak`. A row already re-judged as method or results is done, and
    a row YJUN moved to `keep` (the 08-11 correction) is not weak at all.
    """
    return [
        r
        for r in superseded(load_labels(facet))
        if r.get("label") == LEGACY_WEAK_LABEL
    ]


def rejudge_row(original: dict, label: str, by: str = "YJUN") -> dict:
    """A new row that supersedes `original`, in the shape YJUN's own correction used.

    Everything from the original is carried over so the row remains a complete
    training example on its own; only the verdict and the correction history
    change. `sampling` comes along untouched, which keeps the write guard
    meaningful — a re-judged ranked row is still a ranked row.
    """
    return {
        **original,
        "label": label,
        "corrected_from": original.get("label"),
        "corrected_by": by,
        "corrected_at": utcnow().isoformat(),
    }


def run_rejudge_session(
    facet: str = "relevance",
    prompt=input,
    printer=print,
    by: str = "YJUN",
) -> dict[str, Any]:
    """Re-judge the unsplit `drop_weak` rows, one keystroke each.

    Deliberately offers only the two weak kinds plus skip. This session exists to
    split a verdict that was already made, not to reopen it — a row that should
    have been `keep` or `not_urban` is a different correction, and mixing the two
    would turn a five-minute pass into a re-labelling of the archive.
    """
    todo = weak_rows_to_rejudge(facet)
    if not todo:
        printer("nothing to re-judge — no rows carry the unsplit label")
        return {"labelled": 0, "remaining": 0, "counts": {}, "stopped_early": False}

    printer(
        f"\n{len(todo)} row(s) labelled before `drop_weak` was split.\n"
        "  m  the METHOD was weak    (thin data, no baseline, n too small)\n"
        "  r  the RESULTS were weak  (it worked, and nothing followed from it)\n"
        "  s  skip — leave it unsplit for now\n"
        "  type `quit` to stop; re-run to continue where you left off\n"
    )

    assert_writable(facet, sampling_of(facet))

    started = time.monotonic()
    counts: Counter = Counter()
    n = 0
    stopped = False

    for i, original in enumerate(todo, start=1):
        printer(
            f"\n[{i}/{len(todo)}] {original.get('date')} "
            f"{original.get('source', '?')} rank {original.get('rank', '?')}\n"
            f"  {original.get('title', '')}"
        )
        answer = (prompt("   weak [m]ethod / weak [r]esults / [s]kip: ") or "").strip().lower()
        if answer in ("quit", "exit"):
            stopped = True
            break
        key = answer[:1]
        if key not in ("m", "r"):
            counts["skip"] += 1
            continue
        label = LABEL_KEYS[key]
        counts[label] += 1
        n += append_one(facet, rejudge_row(original, label, by=by))

    
    elapsed = time.monotonic() - started
    remaining = len(todo) - n
    printer(
        f"\nwrote {n} re-judgements to {labels_path(facet)} in "
        f"{elapsed / 60:.1f} min ({dict(counts)})"
    )
    if remaining > 0:
        printer(
            f"{remaining} still unsplit — re-run the same command to continue"
            + (" (stopped early)" if stopped else "")
        )
    return {
        "labelled": n,
        "remaining": max(0, remaining),
        "counts": dict(counts),
        "stopped_early": stopped,
        "minutes": round(elapsed / 60, 2),
    }


# --------------------------------------------------------------------------
# The code probe (hotfix, F3)
# --------------------------------------------------------------------------

CODE_PROBE_POOL = "code_probe_pool.jsonl"


def code_probe_pool() -> list[dict]:
    """The prepared pool, minus anything already judged.

    A separate reader rather than a generalised `run_probe_session`. That one is
    built around `canon_affinity` bands — it computes affinity, loads venue
    priors, and writes `band_stratified` rows — and bending it to also mean
    "relevance bands over code-bearing arXiv items" would leave one function
    serving two sampling frames. **Two frames in one function is how a row ends
    up in the wrong file**, which is the failure this hotfix exists for. The
    pool is a JSONL of prepared rows, so reading it is a dozen lines.
    """
    path = paths.LABELS / CODE_PROBE_POOL
    if not path.exists():
        return []
    done = {r.get("work_key") for r in load_labels("code_probe")}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("work_key") in done:
            continue
        rows.append(row)
    return rows


def code_probe_row(pool_row: dict, label: str) -> dict[str, Any]:
    """One judgement, carrying why the item was in the pool at all.

    `score`, the band and the **origin of the code signal** travel with the
    verdict so this file alone can answer "does the classifier rank
    code-releasing papers low, and does that track what a person thinks" without
    rebuilding a pool that will have moved.
    """
    return {
        "work_key": pool_row["work_key"],
        "title": pool_row.get("title", ""),
        "date": pool_row.get("date"),
        "label": LABEL_KEYS.get(label, label),
        "band": pool_row.get("band"),
        "rank_in_band": pool_row.get("rank_in_band"),
        "score": pool_row.get("score"),
        "code_basis": pool_row.get("code_basis"),
        "source": pool_row.get("source", "arxiv"),
        "sampling": "code_stratified",
        "labelled_at": utcnow().isoformat(),
        # Stated on every row: stratified over relevance among code-bearing
        # candidates, so no precision@k may be computed from it.
        "not_for_precision_at_k": True,
    }


def run_code_probe_session(
    facet: str = "code_probe", prompt=input, printer=print
) -> dict[str, Any]:
    """Judge the code-bearing arXiv papers the classifier ranks low.

    Every verdict is written before the next question is asked, so stopping
    part-way keeps everything answered so far and re-running resumes.
    """
    assert_writable(facet, "code_stratified")

    todo = code_probe_pool()
    if not todo:
        path = paths.LABELS / CODE_PROBE_POOL
        if not path.exists():
            printer(
                f"no pool at {path} — build it with "
                f"`uv run python scripts/code_probe.py`"
            )
        else:
            printer("every item in the code probe pool has been judged")
        return {"labelled": 0, "remaining": 0, "counts": {}, "stopped_early": False}

    by_band: Counter = Counter(r.get("band") for r in todo)
    printer("")
    printer(
        f"{len(todo)} code-bearing arXiv paper(s) to judge "
        f"({dict(by_band)} by relevance band)."
    )
    printer("These are papers that release code and that the classifier ranks low.")
    printer("The question is whether it is right to.")
    printer("")
    printer(LABEL_LEGEND)
    printer("  type 'quit' to stop — every answer is written as you give it")

    started = time.monotonic()
    n = 0
    counts: Counter = Counter()
    stopped = False

    for i, row in enumerate(todo, start=1):
        printer("")
        printer(
            f"[{i}/{len(todo)}] {row.get('date')}  band {row.get('band')}  "
            f"score {row.get('score')}  code from {row.get('code_basis')}"
        )
        printer(f"  {row.get('title', '')}")
        key = _ask_label(prompt, printer)
        if key is None:
            stopped = True
            break
        label = LABEL_KEYS[key]
        counts[label] += 1
        if label == "skip":
            continue
        n += append_one(facet, code_probe_row(row, key))

    elapsed = time.monotonic() - started
    remaining = len(todo) - n - counts.get("skip", 0)
    printer("")
    printer(
        f"wrote {n} judgement(s) to {labels_path(facet)} in "
        f"{elapsed / 60:.1f} min ({dict(counts)})"
    )
    if remaining > 0:
        printer(
            f"{remaining} left — re-run `uc review --label code_probe` to continue"
            + (" (stopped early)" if stopped else "")
        )
    return {
        "labelled": n,
        "remaining": max(0, remaining),
        "counts": dict(counts),
        "stopped_early": stopped,
        "minutes": round(elapsed / 60, 2),
    }


# --------------------------------------------------------------------------
# The subfield check
# --------------------------------------------------------------------------

SUBFIELD_CHECK_POOL = "subfield_check_pool.jsonl"

# How much of an abstract to show when there is no summary. Enough to tell what
# was done to what; not so much that twenty of them become a reading session.
ABSTRACT_CHARS = 320


def subfield_check_pool() -> list[dict]:
    """The prepared pool, minus anything already judged.

    Its own reader for the same reason the code probe has one: a session that
    serves two sampling frames is how a row ends up in the wrong file.
    """
    path = paths.LABELS / SUBFIELD_CHECK_POOL
    if not path.exists():
        return []
    done = {r.get("work_key") for r in load_labels("subfield_check")}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("work_key") in done:
            continue
        rows.append(row)
    return rows


def _first_sentences(text: str, limit: int = ABSTRACT_CHARS) -> str:
    """The opening of an abstract, cut at a sentence boundary where possible."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return (cut[: stop + 1] if stop > limit // 2 else cut.rstrip() + "…")


def subfield_check_body(item: Optional[Item]) -> tuple[str, str]:
    """(what to show, where it came from).

    **The title alone is not enough to judge scope**, which is the whole reason
    this pool needed a session rather than a spreadsheet. "Technology transfer
    and innovation performance in digital industries: intracity and intercity
    linkages in China" could sit on either side of the line; the venue and a
    sentence of what was actually done decide it.

    The pool rows carry only a title, so the item is loaded for the rest. Best
    available, in order: the written summary, then the opening of the abstract,
    then nothing — and when it is nothing that is **said out loud**, because a
    judgement made on a title alone should be visibly weaker than one made on a
    summary, not silently identical to it.
    """
    if item is None:
        return "", "item not on disk"
    summary = item.summary.en if item.summary else None
    if summary and summary.what:
        return summary.what, "summary"
    if item.bibliography.abstract:
        return _first_sentences(item.bibliography.abstract), "abstract"
    return "", "no abstract — judging on title and venue alone"


def subfield_topic_label(item: Optional[Item]) -> str:
    """The paper's own primary topic, in words.

    A bare `2208` is opaque, and the judgement being asked for is precisely
    about that code. OpenAlex's topic name is the finest-grained label we hold
    and it comes from the paper itself, so it is shown beside the number on
    every item — including the seven with no abstract, where it is most of what
    there is to go on.
    """
    if item is None or not item.entities.topics:
        return ""
    topics = list(item.entities.topics)
    primary = next((t for t in topics if getattr(t, "is_primary", False)), topics[0])
    return primary.label or ""


def subfield_check_row(pool_row: dict, label: str, basis: str) -> dict[str, Any]:
    """One judgement, carrying the subfield it was about.

    `subfield` is the point of the file: these labels exist to confirm or
    overturn the four exclusions the `off_subfield` gate now enforces, and each
    of those rests on three or four observations. `shown` records what the
    labeller actually had in front of them, so a verdict reached on a title
    alone can be told apart later from one reached on a summary.
    """
    return {
        "work_key": pool_row["work_key"],
        "title": pool_row.get("title", ""),
        "date": pool_row.get("date"),
        "label": LABEL_KEYS.get(label, label),
        "subfield": pool_row.get("subfield"),
        "source": pool_row.get("source", "journal"),
        "shown": basis,
        "sampling": "subfield_check",
        "labelled_at": utcnow().isoformat(),
        # Drawn by subfield, not by rank and not by band. Its own frame, and no
        # precision@k may be computed from it.
        "not_for_precision_at_k": True,
    }


def run_subfield_check_session(
    facet: str = "subfield_check", prompt=input, printer=print
) -> dict[str, Any]:
    """Judge papers in the four subfields the scope gate now excludes.

    Every verdict is written before the next question is asked, so stopping
    part-way keeps what was answered and re-running resumes.
    """
    assert_writable(facet, "subfield_check")

    todo = subfield_check_pool()
    if not todo:
        path = paths.LABELS / SUBFIELD_CHECK_POOL
        if not path.exists():
            printer(f"no pool at {path}")
        else:
            printer("every item in the subfield check pool has been judged")
        return {"labelled": 0, "remaining": 0, "counts": {}, "stopped_early": False}

    by_sub: Counter = Counter(r.get("subfield") for r in todo)
    printer("")
    printer(f"{len(todo)} paper(s) to judge, {dict(by_sub)} by subfield.")
    printer(
        "These sit in the four subfields the scope gate now excludes. Each of "
        "those exclusions rests on three or four observations, so these "
        "judgements are what the whole rule stands on."
    )
    printer("A `keep` here is evidence the gate is too wide.")
    printer("")
    printer(LABEL_LEGEND)
    printer("  type 'quit' to stop — every answer is written as you give it")

    started = time.monotonic()
    n = 0
    counts: Counter = Counter()
    stopped = False

    for i, row in enumerate(todo, start=1):
        item = store.load_item(row["work_key"])
        body, basis = subfield_check_body(item)
        venue = ""
        if item and item.bibliography.primary_location:
            venue = item.bibliography.primary_location.source_name or ""

        topic = subfield_topic_label(item)
        printer("")
        # The subfield leads every item: it is the thing being judged, and the
        # topic name is there because a bare four-digit code is not a question
        # anyone can answer.
        printer(
            f"[{i}/{len(todo)}]  subfield {row.get('subfield')}"
            + (f" — {topic}" if topic else "")
            + f"   {row.get('date')}"
        )
        printer(f"  venue: {venue}")
        printer(f"  {row.get('title', '')}")
        if body:
            printer(f"  {basis}: {body}")
        else:
            tags = []
            if item:
                tags = [e.label for e in (item.entities.methods + item.entities.data)][:6]
            printer(f"  ({basis})")
            if tags:
                printer(f"  extracted tags: {', '.join(tags)}")

        key = _ask_label(prompt, printer)
        if key is None:
            stopped = True
            break
        label = LABEL_KEYS[key]
        counts[label] += 1
        if label == "skip":
            continue
        n += append_one(facet, subfield_check_row(row, key, basis))

    elapsed = time.monotonic() - started
    remaining = len(todo) - n - counts.get("skip", 0)
    printer("")
    printer(
        f"wrote {n} judgement(s) to {labels_path(facet)} in "
        f"{elapsed / 60:.1f} min ({dict(counts)})"
    )
    if remaining > 0:
        printer(
            f"{remaining} left — re-run `uc review --label subfield_check` to continue"
            + (" (stopped early)" if stopped else "")
        )
    return {
        "labelled": n,
        "remaining": max(0, remaining),
        "counts": dict(counts),
        "stopped_early": stopped,
        "minutes": round(elapsed / 60, 2),
    }
