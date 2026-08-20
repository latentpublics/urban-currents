"""Correct the run-log rows where `quiet` meant two different things (0U, U5).

`decide()` grants `quiet` for **published nothing**. `daily.py` then overwrote
that with `issue.quiet_day`, which means **nothing cleared the headline
threshold** — a different fact about a day that did publish. U5 removed the
overwrite, so every row from now on is right. This is about the 23 rows already
in the archive.

They are not a curiosity. Of 58 rows, 23 say `quiet` while carrying 9 to 20
published items, and **no row anywhere says `quiet` with nothing published** —
so in the committed archive the word currently carries no information about
silence at all. `uc weekly` counts those days as silent, `uc status` reads the
same rows, and anything later that asks "how often did this thing publish
nothing" gets 23 for an answer when the truth is 0.

What it does, per row where `status == "quiet"` and `published > 0`:

  * `status` → `"published"` — the verdict `decide()` would reach today.
  * `headline_present` → `False`, taken from the issue's own `quiet_day` when
    the issue file exists, so the fact the old value was carrying is kept
    rather than thrown away.
  * `migrated_by` → `"0U-U5"`, which is what makes `--revert` exact. A marker
    in the row beats a manifest under `runs/` (gitignored, so a re-clone would
    lose the ability to undo) and beats inferring it from the values, which
    would sweep up genuine future rows that also have no headline.

`content/` is pipeline output and is never hand-edited; this is the sanctioned
path — a script, run once, that says what it touched.

    uv run python scripts/migrate_runlog_quiet.py            # --check (default)
    uv run python scripts/migrate_runlog_quiet.py --apply
    uv run python scripts/migrate_runlog_quiet.py --revert
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import paths  # noqa: E402

MARKER = "0U-U5"


def _rows() -> list[tuple[Path, dict[str, Any]]]:
    log_dir = paths.CONTENT / "runs_log"
    out = []
    for p in sorted(log_dir.glob("*.json")):
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            print(f"  ! {p.name} is not readable JSON; left alone")
    return out


def _write(path: Path, row: dict[str, Any]) -> None:
    # Byte-identical to what `record()` writes: sorted keys, indent 2, LF.
    path.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _headline_present(date_str: str) -> bool | None:
    """What the old `quiet` was really saying, recovered from the issue."""
    issue = paths.CONTENT / "issues" / f"{date_str}.json"
    if not issue.exists():
        return None
    try:
        data = json.loads(issue.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if "quiet_day" in data:
        return not bool(data["quiet_day"])
    headline = data.get("headline") or {}
    return bool(headline.get("present"))


def check() -> int:
    hits = [(p, r) for p, r in _rows()
            if r.get("status") == "quiet" and (r.get("published") or 0) > 0]
    done = [(p, r) for p, r in _rows() if r.get("migrated_by") == MARKER]
    print(f"rows needing correction: {len(hits)}")
    for _, r in hits:
        print(f"  {r['date']}  status=quiet  published={r.get('published')}")
    print(f"rows already migrated:   {len(done)}")
    return len(hits)


def apply() -> int:
    n = 0
    for path, row in _rows():
        if row.get("status") != "quiet" or (row.get("published") or 0) <= 0:
            continue
        row["status"] = "published"
        present = _headline_present(str(row.get("date")))
        # The old value said there was no headline. If the issue file confirms
        # it, use the file; if the issue is gone, keep the claim rather than
        # inventing a `None` — this row is the only witness left.
        row["headline_present"] = False if present is None else present
        row["migrated_by"] = MARKER
        _write(path, row)
        print(f"  {row['date']}: quiet -> published, headline_present="
              f"{row['headline_present']}")
        n += 1
    print(f"{n} row(s) corrected")
    return n


def revert() -> int:
    n = 0
    for path, row in _rows():
        if row.get("migrated_by") != MARKER:
            continue
        row["status"] = "quiet"
        row.pop("headline_present", None)
        row.pop("migrated_by", None)
        _write(path, row)
        print(f"  {row['date']}: published -> quiet (reverted)")
        n += 1
    print(f"{n} row(s) reverted")
    return n


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode == "--apply":
        apply()
    elif mode == "--revert":
        revert()
    elif mode == "--check":
        check()
    else:
        print(__doc__)
        sys.exit(2)
