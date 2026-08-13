"""Longest run of words a summary shares with its abstract (phase 0c, P6-2).

Copyright protects expression, not fact. Crossref is explicit that abstracts are
copyrighted by their publishers and that its own right to redistribute them does
not transfer to us; I4OA says the same. So reading an abstract and writing new
sentences from the facts in it is a different act from copying it, and the line
between those two acts is measurable.

Eight consecutive words is the threshold the summarize prompt states. It is a
convention, not a legal boundary — the point is that the number exists, is
checked, and is reported rather than assumed.

Reports only. Nothing is regenerated here: a summary is expensive and the first
question is how many are affected at all.

Usage:
    uv run python scripts/verbatim_check.py [--n 8] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import store  # noqa: E402

_WORD = re.compile(r"[\w']+", re.UNICODE)


def words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def longest_common_run(a: list[str], b: list[str]) -> tuple[int, int]:
    """Length of the longest shared consecutive run, and where it starts in `a`.

    Classic O(len(a) x len(b)) table, kept to two rows. Abstracts run to a few
    hundred words, so this is microseconds and needs no cleverness.
    """
    if not a or not b:
        return 0, 0
    prev = [0] * (len(b) + 1)
    best = best_at = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, best_at = cur[j], i - cur[j]
        prev = cur
    return best, best_at


def summary_text(item) -> str:
    en = item.summary.en
    if not en:
        return ""
    return " ".join(p for p in (en.what, en.why, en.caveats) if p)


def measure(threshold: int = 8) -> dict:
    rows = []
    checked = 0
    for item in store.iter_items():
        abstract = item.bibliography.abstract or ""
        summary = summary_text(item)
        if not abstract.strip() or not summary.strip():
            continue
        checked += 1
        a_words, s_words = words(abstract), words(summary)
        run, at = longest_common_run(s_words, a_words)
        if run >= threshold:
            matched = s_words[at : at + run]
            # The prompt *requires* facts to be carried over unchanged, and a
            # decimal like 0.768 tokenises to two words, so a run of numbers and
            # units inflates the raw count without being reused prose. Counting
            # the alphabetic tokens separates "a copied sentence" from "a copied
            # measurement", which is the distinction that matters: copyright
            # protects the expression, not the number.
            prose = sum(1 for w in matched if not w.isdigit())
            rows.append({
                "work_key": item.work_key,
                "title": item.bibliography.title,
                "run_length": run,
                "prose_words": prose,
                "matched_text": " ".join(matched),
                "prompt_version": (
                    item.provenance.llm.prompt_version if item.provenance.llm else None
                ),
            })

    rows.sort(key=lambda r: (-r["prose_words"], -r["run_length"]))
    prose_over = [r for r in rows if r["prose_words"] >= threshold]
    return {
        "threshold_words": threshold,
        "summaries_checked": checked,
        "over_threshold": len(rows),
        "share_over_threshold": round(len(rows) / checked, 4) if checked else None,
        # The number to act on: runs that are still over the line once digits
        # are discounted.
        "over_threshold_prose_only": len(prose_over),
        "share_prose_only": round(len(prose_over) / checked, 4) if checked else None,
        "longest_run": rows[0]["run_length"] if rows else 0,
        "longest_prose_run": prose_over[0]["prose_words"] if prose_over else 0,
        "violations": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="consecutive-word threshold")
    ap.add_argument("--json", help="also write the full result here")
    a = ap.parse_args()

    result = measure(a.n)
    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps({k: v for k, v in result.items() if k != "violations"}, indent=2))
    for row in result["violations"][:20]:
        print(
            f"\n  {row['work_key']}  ({row['run_length']} tokens, "
            f"{row['prose_words']} non-numeric)"
        )
        print(f"    {row['matched_text']}")


if __name__ == "__main__":
    main()
