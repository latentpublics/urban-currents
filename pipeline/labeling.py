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
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

from . import paths
from .config import cfg
from .metrics import Run, utcnow
from .models import Item

LABEL_KEYS: dict[str, str] = {
    "k": "keep",
    "n": "drop_not_urban",
    "q": "drop_not_our_kind",
    "w": "drop_weak",
    "s": "skip",
}
DROP_LABELS = ("drop_not_urban", "drop_not_our_kind", "drop_weak")

LABEL_PROMPT = "   [k]eep / [n]ot urban / not our kind [q] / [w]eak / [s]kip: "
LABEL_LEGEND = (
    "  k  keep — worth publishing as a card\n"
    "  n  not urban research at all            (classifier error)\n"
    "  q  urban research, not the kind we cover (qualitative case study,\n"
    "                                           theory, policy commentary)\n"
    "  w  the right kind, but weak or minor\n"
    "  s  skip — undecided, offer it again next time"
)


def labels_path(facet: str = "relevance") -> Path:
    paths.LABELS.mkdir(parents=True, exist_ok=True)
    return paths.LABELS / f"{facet}.jsonl"


def load_labels(facet: str = "relevance") -> list[dict]:
    path = labels_path(facet)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def labelled_keys(facet: str = "relevance") -> set[tuple[str, str]]:
    """(date, work_key) pairs already labelled — the basis of resuming."""
    return {(r.get("date", ""), r.get("work_key", "")) for r in load_labels(facet)}


def append_labels(facet: str, rows: Iterable[dict]) -> int:
    path = labels_path(facet)
    n = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


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
    # `classify` is the candidate pool: everything that cleared the gate and was
    # scored, before the daily slots were applied.
    items = read_stage(run, "classify") or read_stage(run, "select")
    thr = float(cfg("classifier.threshold", 0.35) if threshold is None else threshold)

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
        "labelled_at": utcnow().isoformat(),
    }


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

    started = time.monotonic()
    rows: list[dict] = []
    counts: Counter = Counter()
    stopped = False

    for i, (item, source, rank) in enumerate(todo, start=1):
        printer(_render(item, source, rank, f"{i}/{len(todo)}"))
        answer = (prompt(LABEL_PROMPT) or "").strip().lower()
        if answer in ("quit", "exit"):
            stopped = True
            break
        key = answer[:1] if answer[:1] in LABEL_KEYS else "s"
        label = LABEL_KEYS[key]
        counts[label] += 1
        if label == "skip":
            continue
        rows.append(label_row(item, source, rank, label, d, thr))

    n = append_labels(facet, rows)
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
    """
    rows = load_labels(facet)
    if not rows:
        return {
            "n_labels": 0,
            "k": k,
            "days_labelled": 0,
            "by_source": {},
            "note": "no labels yet — run `uc review --label relevance --date …`",
        }

    by_source: dict[str, dict] = {}
    for source in sorted({r.get("source", "unknown") for r in rows}):
        srows = [r for r in rows if r.get("source") == source]
        by_day: dict[str, list[dict]] = {}
        for r in srows:
            by_day.setdefault(r.get("date", ""), []).append(r)

        per_day = []
        for _day, day_rows in sorted(by_day.items()):
            day_rows.sort(key=lambda r: r.get("rank", 0))
            topk = day_rows[:k]
            if topk:
                per_day.append(sum(1 for r in topk if r["label"] == "keep") / len(topk))

        reasons = Counter(r["label"] for r in srows if r["label"] in DROP_LABELS)
        n_drops = sum(reasons.values())

        def share(name: str) -> Optional[float]:
            return round(reasons.get(name, 0) / n_drops, 3) if n_drops else None

        by_source[source] = {
            "n_labels": len(srows),
            "days": len(per_day),
            f"precision_at_{k}": (
                round(sum(per_day) / len(per_day), 4) if per_day else None
            ),
            "per_day": [round(p, 4) for p in per_day],
            "keep_rate": round(
                sum(1 for r in srows if r["label"] == "keep") / len(srows), 4
            ),
            "drop_reasons": {
                "not_urban": reasons.get("drop_not_urban", 0),
                "not_our_kind": reasons.get("drop_not_our_kind", 0),
                "weak": reasons.get("drop_weak", 0),
            },
            "drop_reason_share": {
                "not_urban": share("drop_not_urban"),
                "not_our_kind": share("drop_not_our_kind"),
                "weak": share("drop_weak"),
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
