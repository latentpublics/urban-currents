"""``uc review`` — the human checkpoint (PRD §7).

Two modes:

- **full review** — opens the day's preview, then walks the items with
  ``[a]pprove / [r]eject / [e]dit / [s]kip``. Editing opens the Item JSON in
  ``$EDITOR`` and records the field paths that changed in ``review.edits``.
- **labelling** (``--label relevance``) — a fast keep/drop pass over the top N by
  classifier score, appended to ``runs/labels/relevance.jsonl``. This is what
  produces the precision@10 number in Q1.

**The elapsed time is the point.** Q4 asks whether review fits in 15 minutes a
day, and self-reported times are always under-reported, so the clock starts and
stops here and the result lands in ``metrics.timing.review_s``.

This module is exercised by tests through the injected ``prompt`` / ``opener``
callables; the interactive path is never driven by automation.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import paths, store
from .metrics import Run
from .models import Item

Prompt = Callable[[str], str]
Opener = Callable[[Path], None]


def _default_prompt(message: str) -> str:  # pragma: no cover - interactive
    return input(message).strip().lower()


def _default_opener(path: Path) -> None:  # pragma: no cover - interactive
    webbrowser.open(path.resolve().as_uri())


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Field paths → scalar values, so an edit can be reported as a path list."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def changed_paths(before: dict, after: dict) -> list[str]:
    fb, fa = flatten(before), flatten(after)
    keys = set(fb) | set(fa)
    return sorted(k for k in keys if fb.get(k) != fa.get(k))


@dataclass
class ReviewOutcome:
    approved: int = 0
    rejected: int = 0
    edited: int = 0
    skipped: int = 0
    seconds: float = 0.0
    edits: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "approved": self.approved,
            "rejected": self.rejected,
            "edited": self.edited,
            "skipped": self.skipped,
            "seconds": round(self.seconds, 1),
            "edits": self.edits,
        }


def edit_item_in_editor(item: Item) -> Optional[Item]:  # pragma: no cover - interactive
    editor = os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "vi")
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix=item.work_key.replace(":", "_"))
    path = Path(tmp)
    try:
        os.close(fd)
        path.write_text(store.dumps(item), encoding="utf-8")
        subprocess.call([editor, str(path)])
        return Item.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  edit discarded: {type(e).__name__}: {e}")
        return None
    finally:
        path.unlink(missing_ok=True)


def run_review_session(
    d: date,
    prompt: Prompt = _default_prompt,
    opener: Opener = _default_opener,
    editor: Callable[[Item], Optional[Item]] = edit_item_in_editor,
) -> ReviewOutcome:
    issue = store.load_issue(d)
    if issue is None:
        print(f"no issue for {d}")
        return ReviewOutcome()

    run = Run.for_date(d)
    preview = run.dir / "preview.html"
    if preview.exists():
        opener(preview)
    else:
        print(f"(no preview at {preview}; run `uc preview --date {d}`)")

    outcome = ReviewOutcome()
    started = time.monotonic()

    for work_key in issue.items:
        item = store.load_item(work_key)
        if item is None:
            continue
        print(f"\n{work_key} — {item.bibliography.title}")
        if item.summary.en:
            print(f"  what: {item.summary.en.what}")
            print(f"  why:  {item.summary.en.why}")
        answer = prompt("  [a]pprove / [r]eject / [e]dit / [s]kip: ")

        if answer.startswith("a"):
            item.review.status = "approved"
            outcome.approved += 1
        elif answer.startswith("r"):
            item.review.status = "rejected"
            outcome.rejected += 1
        elif answer.startswith("e"):
            before = item.model_dump(mode="json", by_alias=True)
            edited = editor(item)
            if edited is not None:
                after = edited.model_dump(mode="json", by_alias=True)
                touched = [p for p in changed_paths(before, after) if not p.startswith("review.")]
                edited.review.status = "edited"
                # The edit log is the quiet payload of this command: which field
                # YJUN keeps rewriting is what tells Phase 1 where to fix prompts.
                edited.review.edits = sorted(set(edited.review.edits) | set(touched))
                item = edited
                outcome.edits[work_key] = touched
                outcome.edited += 1
            else:
                outcome.skipped += 1
        else:
            outcome.skipped += 1

        store.save_item(item, today=d)

    outcome.seconds = time.monotonic() - started
    run.metrics.timing["review_s"] = round(
        run.metrics.timing.get("review_s", 0.0) + outcome.seconds, 1
    )
    run.metrics.stages["review"] = "OK"
    setattr(run.metrics, "review", outcome.as_dict())
    run.save()

    print(
        f"\nreviewed {len(issue.items)} items in {outcome.seconds / 60:.1f} min "
        f"(approved {outcome.approved}, rejected {outcome.rejected}, "
        f"edited {outcome.edited}, skipped {outcome.skipped})"
    )
    return outcome


# --------------------------------------------------------------------------
# Labelling mode
# --------------------------------------------------------------------------


def labels_path(facet: str) -> Path:
    paths.LABELS.mkdir(parents=True, exist_ok=True)
    return paths.LABELS / f"{facet}.jsonl"


def candidates_for_labelling(d: date, top: int) -> list[Item]:
    """Top-N of the day's classified candidates, highest score first."""
    from .stages import read_stage

    run = Run.for_date(d)
    items = read_stage(run, "classify") or read_stage(run, "select")
    items.sort(key=lambda it: (-it.scores.relevance, it.work_key))
    return items[:top]


def append_labels(facet: str, rows: Iterable[dict]) -> int:
    path = labels_path(facet)
    n = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def run_labeling_session(
    d: date, facet: str = "relevance", top: int = 30, prompt: Prompt = _default_prompt
) -> int:
    items = candidates_for_labelling(d, top)
    if not items:
        print(f"no classified candidates for {d}; run `uc classify --date {d}` first")
        return 0

    print(f"labelling {len(items)} candidates for {d} — [k]eep / [d]rop / [s]kip")
    started = time.monotonic()
    rows = []
    for rank, item in enumerate(items, start=1):
        print(f"\n{rank}. [{item.scores.relevance:.3f}] {item.bibliography.title}")
        print(f"   {(item.bibliography.abstract or '')[:240]}")
        answer = prompt("   [k]eep / [d]rop / [s]kip: ")
        if answer.startswith("s"):
            continue
        rows.append(
            {
                "date": str(d),
                "work_key": item.work_key,
                "rank": rank,
                "score": item.scores.relevance,
                "label": "keep" if answer.startswith("k") else "drop",
                "source": "arxiv" if item.ids.arxiv else "journal",
                "classifier_version": item.provenance.classifier_version,
            }
        )

    n = append_labels(facet, rows)
    elapsed = time.monotonic() - started
    run = Run.for_date(d)
    run.metrics.timing["label_s"] = round(
        run.metrics.timing.get("label_s", 0.0) + elapsed, 1
    )
    run.save()
    print(f"\nwrote {n} labels to {labels_path(facet)} in {elapsed / 60:.1f} min")
    return n


def precision_at_k(facet: str = "relevance", k: int = 10) -> dict:
    """Q1's precision@10, computed from whatever labels exist so far."""
    path = labels_path(facet)
    if not path.exists():
        return {"labelled_days": 0, "precision_at_k": None, "k": k, "n_labels": 0}

    by_day: dict[str, list[dict]] = {}
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_day.setdefault(row["date"], []).append(row)
        total += 1

    per_day = []
    for day, rows in sorted(by_day.items()):
        rows.sort(key=lambda r: r["rank"])
        topk = rows[:k]
        if not topk:
            continue
        per_day.append(sum(1 for r in topk if r["label"] == "keep") / len(topk))

    by_source: dict[str, dict[str, int]] = {}
    for rows in by_day.values():
        for r in rows:
            b = by_source.setdefault(r.get("source", "unknown"), {"keep": 0, "drop": 0})
            b[r["label"]] = b.get(r["label"], 0) + 1

    return {
        "labelled_days": len(per_day),
        "n_labels": total,
        "k": k,
        "precision_at_k": round(sum(per_day) / len(per_day), 4) if per_day else None,
        "per_day": [round(p, 4) for p in per_day],
        "by_source": by_source,
    }
