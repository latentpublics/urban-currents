"""``uc review`` — the human checkpoint (PRD §7).

Two modes:

- **full review** — opens the day's preview, then walks the items with
  ``[a]pprove / [r]eject / [e]dit / [s]kip``. Editing opens the Item JSON in
  ``$EDITOR`` and records the field paths that changed in ``review.edits``.
- **labelling** (``--label relevance``) — lives in ``pipeline/labeling.py``.
  It is a separate module because it grew a different job: its output is the
  training set for a classifier that does not exist yet, not only a measurement.

**The elapsed time is the point.** Q4 asks whether review fits in 15 minutes a
day, and self-reported times are always under-reported, so the clock starts and
stops here and the result lands in ``metrics.timing.review_s``.

This module is exercised by tests through the injected ``prompt`` / ``opener``
callables; the interactive path is never driven by automation.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

from . import store
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


# --------------------------------------------------------------------------
# Labelling mode lives in pipeline/labeling.py — see the note above.
# Re-exported here so `uc review --label …` and existing imports keep working.
# --------------------------------------------------------------------------

from .labeling import (  # noqa: E402,F401
    labels_path,
    load_labels,
    precision_at_k,
    probe_summary,
    run_labeling_session,
    run_probe_session,
    stratified_sample,
)
