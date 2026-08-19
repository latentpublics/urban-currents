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

**And it is written however the session ends.** Recording only on a completed
walk of the day biases Q4 by exactly the sessions a busy person has — the
interrupted ones — which is why the measurement stood at zero days for eight
batches while a review had in fact been started. ``reviewed_n`` is recorded
beside the seconds so a stopped session is not read as a fast one.

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
    # How many items were actually judged, and whether the session ran out of
    # items or was stopped. 90 seconds over 3 items and 14 minutes over 24 are
    # the same number of minutes and not the same measurement.
    reviewed_n: int = 0
    total_n: int = 0
    stopped_early: bool = False

    @property
    def seconds_per_item(self) -> Optional[float]:
        if not self.reviewed_n:
            return None
        return round(self.seconds / self.reviewed_n, 1)

    def as_dict(self) -> dict:
        return {
            "approved": self.approved,
            "rejected": self.rejected,
            "edited": self.edited,
            "skipped": self.skipped,
            "seconds": round(self.seconds, 1),
            "reviewed_n": self.reviewed_n,
            "total_n": self.total_n,
            "seconds_per_item": self.seconds_per_item,
            "stopped_early": self.stopped_early,
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

    outcome = ReviewOutcome(total_n=len(issue.items))
    started = time.monotonic()

    try:
        _review_items(issue, outcome, d, prompt, editor)
    except KeyboardInterrupt:
        # Ctrl-C is how a real review ends more often than not. The elapsed time
        # up to here is a measurement, and discarding it is what left Q4 with
        # zero days after eight batches.
        outcome.stopped_early = True
        print("\n(interrupted — the time and judgements so far are saved)")
    finally:
        outcome.seconds = time.monotonic() - started
        run.metrics.timing["review_s"] = round(
            run.metrics.timing.get("review_s", 0.0) + outcome.seconds, 1
        )
        # Accumulated alongside the seconds, or the two stop describing the same
        # session as soon as a day is reviewed in more than one sitting.
        run.metrics.timing["reviewed_n"] = (
            run.metrics.timing.get("reviewed_n", 0) + outcome.reviewed_n
        )
        run.metrics.stages["review"] = "OK"
        setattr(run.metrics, "review", outcome.as_dict())
        run.save()

    per_item = outcome.seconds_per_item
    print(
        f"\nreviewed {outcome.reviewed_n} of {outcome.total_n} items in "
        f"{outcome.seconds / 60:.1f} min "
        f"({per_item if per_item is not None else '—'} s/item; "
        f"approved {outcome.approved}, rejected {outcome.rejected}, "
        f"edited {outcome.edited}, skipped {outcome.skipped})"
    )
    if outcome.reviewed_n < outcome.total_n:
        print(
            f"{outcome.total_n - outcome.reviewed_n} left for {d} — "
            f"re-run the same command; the clock adds up"
        )
    return outcome


def _review_items(
    issue,
    outcome: ReviewOutcome,
    d: date,
    prompt: Prompt,
    editor: Callable[[Item], Optional[Item]],
) -> None:
    """The loop itself, so the caller's `finally` owns the clock.

    Split out for one reason: whatever happens in here — a quit, a Ctrl-C, an
    unhandled exception — the elapsed time has already been earned and must be
    recorded. Keeping the timing in the caller makes that structural rather than
    a thing every exit path has to remember.
    """
    for work_key in issue.items:
        item = store.load_item(work_key)
        if item is None:
            continue
        print(f"\n{work_key} — {item.bibliography.title}")
        if item.summary.en:
            print(f"  what: {item.summary.en.what}")
            print(f"  why:  {item.summary.en.why}")
        answer = prompt("  [a]pprove / [r]eject / [e]dit / [s]kip / [q]uit: ")

        if answer.startswith("q"):
            outcome.stopped_early = True
            return

        outcome.reviewed_n += 1
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


# --------------------------------------------------------------------------
# Labelling mode
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Labelling mode lives in pipeline/labeling.py — see the note above.
# Re-exported here so `uc review --label …` and existing imports keep working.
# --------------------------------------------------------------------------

from .labeling import (  # noqa: E402,F401
    append_one,
    assert_writable,
    code_probe_pool,
    labels_path,
    load_labels,
    precision_at_k,
    probe_summary,
    code_probe_row,
    run_code_probe_session,
    run_labeling_session,
    run_probe_session,
    run_rejudge_session,
    run_subfield_check_session,
    stratified_sample,
    subfield_check_pool,
    superseded,
    weak_rows_to_rejudge,
)


# --------------------------------------------------------------------------
# Reviewing when you come back, rather than every morning (phase 0L, M2-3)
# --------------------------------------------------------------------------

PROGRESS_FILE = "review_progress.json"


def progress_path() -> Path:
    from . import paths

    return paths.STATE / PROGRESS_FILE


def read_progress() -> dict:
    path = progress_path()
    if not path.exists():
        return {}
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a lost bookmark is not a failure
        return {}


def write_progress(**fields) -> None:
    """Where the reviewing got to. A bookmark, not a record.

    Being able to stop in the middle is the whole design: someone back from a
    week away works through what accumulated, gets interrupted, and has to be
    able to resume without remembering a date. Losing this file costs a little
    re-reading and nothing else, so it is written best-effort.
    """
    import json

    from .metrics import utcnow

    path = progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {**read_progress(), **fields, "updated_at": utcnow().isoformat()}
    try:
        path.write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError:
        pass


# One sitting. Measured on the first real day: 7 withheld and 36 near-misses,
# so a week away is roughly 250 items and an uncapped session would be a
# afternoon's work presented as a prompt. The cap is on the sitting, not on the
# queue — nothing is dropped, and the remaining count is printed.
PENDING_BATCH = 25


def run_pending_session(
    prompt: Prompt = _default_prompt, printer=print, limit: Optional[int] = PENDING_BATCH
) -> dict:
    """Judge what the pipeline held while nobody was looking.

    This is the command that replaces the daily review. It does not ask for a
    date: a week away leaves a week of held items and **remembering which dates
    those were is exactly the friction this is meant to remove**.
    """
    from . import held as held_queue
    from .labeling import (
        LABEL_KEYS,
        _ask_label,
        append_one,
        assert_writable,
        held_review_row,
    )

    # Before the first item is shown, not after the last (F2).
    assert_writable("held_review", "held_review")

    everything = held_queue.pending()
    waiting = everything[:limit] if limit else everything
    if not waiting:
        printer("nothing held — the pipeline was sure about everything it saw")
        return {"judged": 0, "remaining": 0, "counts": {}}

    oldest = waiting[0]["date"]
    withheld_n = sum(1 for r in waiting if r["kind"] == held_queue.WITHHELD)
    printer(
        f"\n{len(everything)} held item(s) waiting, oldest {oldest}. "
        f"Showing {len(waiting)}.\n"
        f"{withheld_n} of these were withheld from an issue; the rest were near "
        f"misses that cost nothing.\n"
        f"None of them was published. Judging them trains the rule that held them.\n"
    )
    printer(LABEL_LEGEND_FOR_HELD)

    import time as _time

    started = _time.monotonic()
    counts: dict[str, int] = {}
    n = 0
    stopped = False

    for i, row in enumerate(waiting, start=1):
        item = store.load_item(row["work_key"])
        printer(
            f"\n[{i}/{len(waiting)}] {row['date']}  {row['kind']}  "
            f"({row['rule']})\n"
            f"  {row.get('title') or (item.bibliography.title if item else '')}\n"
            f"  why held: {row['detail']}"
        )
        key = _ask_label(prompt, printer)
        if key is None:
            stopped = True
            break
        label = LABEL_KEYS[key]
        counts[label] = counts.get(label, 0) + 1
        if label == "skip" or item is None:
            continue
        # Written before the next question is asked. This session batched its
        # rows and wrote them once at the end — the same fault that lost thirty
        # code-probe judgements (F1), still here because the hotfix only
        # reached the sessions that had already gone wrong.
        n += append_one(
            "held_review",
            held_review_row(item, row, label, row.get("source") or "journal"),
        )

    elapsed = _time.monotonic() - started
    remaining = len(everything) - n
    write_progress(last_pending_judged=n, last_pending_remaining=remaining)
    printer(f"\njudged {n} in {elapsed / 60:.1f} min ({counts})")
    if remaining > 0:
        printer(
            f"{remaining} still waiting — re-run `uc review --pending` "
            "to continue where you left off"
        )
    return {
        "judged": n,
        "remaining": remaining,
        "counts": counts,
        "stopped_early": stopped,
        "minutes": round(elapsed / 60, 2),
    }


LABEL_LEGEND_FOR_HELD = (
    "  k  keep — the rule was wrong, this should have been published\n"
    "  n  not urban research at all\n"
    "  q  urban research, not the kind we cover\n"
    "  m  our kind, but the METHOD is weak\n"
    "  r  our kind, but the RESULTS are weak\n"
    "  s  skip — offer it again next time"
)


def sample_published(
    since: date, per_day: int = 2, seed: int = 42
) -> list[tuple[date, str]]:
    """A stratified sample of what actually went out, for reading after the fact.

    Deterministic on `seed` so the same window gives the same sample: two people
    comparing notes, or the same person resuming, should be looking at the same
    cards. Stratified by day rather than drawn from the pool, because the
    question is "has any day gone wrong", and a uniform draw over a month
    concentrates in the days that published most.
    """
    import random

    rng = random.Random(seed)
    out: list[tuple[date, str]] = []
    for issue in _issues_since(since):
        keys = list(issue.items)
        if not keys:
            continue
        rng.shuffle(keys)
        for key in keys[:per_day]:
            out.append((issue.date, key))
    return out


def _issues_since(since: date):
    from .models import Issue

    from . import paths

    issues = []
    directory = paths.CONTENT / "issues"
    if not directory.exists():
        return issues
    for path in sorted(directory.glob("*.json")):
        if path.stem < str(since):
            continue
        try:
            issues.append(Issue.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return issues


def run_sample_session(
    since: date, per_day: int = 2, prompt: Prompt = _default_prompt, printer=print
) -> dict:
    """Read a sample of what was published. Sanity, not gatekeeping.

    Deliberately read-only. Publication already happened; the point is to notice
    a drift, not to correct one card. Anything worth correcting becomes a label.
    """
    picks = sample_published(since, per_day=per_day)
    if not picks:
        printer(f"no issues published since {since}")
        return {"read": 0}

    printer(
        f"\n{len(picks)} card(s) from {len({d for d, _ in picks})} issue(s) "
        f"since {since}.\n"
        f"This is a sanity check on what went out, not a gate — nothing here "
        f"changes a published issue.\n"
    )
    read = 0
    for i, (d, key) in enumerate(picks, start=1):
        item = store.load_item(key)
        if item is None:
            continue
        summary = item.summary.en
        printer(
            f"\n[{i}/{len(picks)}] {d}\n"
            f"  {item.bibliography.title}\n"
            f"  what: {summary.what if summary else '(none)'}\n"
            f"  why:  {summary.why if summary else '(none)'}"
        )
        answer = (prompt("   [enter] fine / [w]rong / [q]uit: ") or "").strip().lower()
        read += 1
        if answer[:1] == "q":
            break
        if answer[:1] == "w":
            printer(
                "   noted. Label it with `uc review --label relevance "
                f"--date {d}` to make it count."
            )
    write_progress(last_sample_since=str(since))
    printer(f"\nread {read} card(s)")
    return {"read": read, "sampled": len(picks)}
