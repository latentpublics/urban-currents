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
# A capitalised run of two or more words: city names, model names, dataset
# names, institutions, journal titles. Matched on the raw text, before casing is
# lost, and removed from both sides before anything is compared.
_PROPER = re.compile(r"\b[A-Z][\w'-]*(?:\s+(?:of|the|de|van|and)\s+)?(?:\s+[A-Z][\w'-]*)+")
_ACRONYM = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b")
_NUMBERISH = re.compile(r"\b[\d][\d,.\-−–/%]*\b")


def words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def prose_words(text: str, vocabulary: set[str]) -> list[str]:
    """Words left after removing everything the prompt requires to be copied.

    Copyright protects creative expression, not a list of facts, and PRD §5.5
    *requires* the numbers, model names, dataset names and place names to be
    carried over unchanged. Measuring those as verbatim reuse counts obedience
    to one rule as a breach of another.

    So they come out before the comparison: numbers and units, capitalised
    multi-word names, acronyms, and every surface form in the controlled
    vocabulary. What is left is the sentence-building — the part that is ours to
    write and therefore the only part where copying means anything.
    """
    stripped = _PROPER.sub(" ", text or "")
    stripped = _ACRONYM.sub(" ", stripped)
    stripped = _NUMBERISH.sub(" ", stripped)
    return [w for w in words(stripped) if w not in vocabulary]


def vocabulary_surface_words() -> set[str]:
    """Every word appearing in a controlled-vocabulary surface form.

    Removed word-wise rather than phrase-wise on purpose: "random forest" and
    "forest" should both drop out, since the shared token is the method's name
    either way and no rewriting would remove it.
    """
    from pipeline.linking.vocab_match import Vocabulary

    out: set[str] = set()
    for facet in ("methods", "data", "tools"):
        try:
            vocab = Vocabulary.load(facet)
        except Exception:  # noqa: BLE001 — a missing vocab file is not fatal here
            continue
        for surface in getattr(vocab, "_by_surface", {}):
            out.update(words(surface))
    return out


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


def measure(threshold: int = 8, prose_threshold: int = 12) -> dict:
    rows = []
    prose_rows: list[dict] = []
    checked = 0
    by_version: dict[str, int] = {}
    vocab = vocabulary_surface_words()
    for item in store.iter_items():
        abstract = item.bibliography.abstract or ""
        summary = summary_text(item)
        if not abstract.strip() or not summary.strip():
            continue
        checked += 1
        version = item.provenance.llm.prompt_version if item.provenance.llm else None
        by_version[version] = by_version.get(version, 0) + 1

        # The measurement that matters: facts stripped from both sides first.
        pa, ps = prose_words(abstract, vocab), prose_words(summary, vocab)
        prose_run, prose_at = longest_common_run(ps, pa)
        if prose_run >= prose_threshold:
            prose_rows.append({
                "work_key": item.work_key,
                "title": item.bibliography.title,
                "run_length": prose_run,
                "matched_text": " ".join(ps[prose_at : prose_at + prose_run]),
                "prompt_version": version,
            })

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

    # Split by prompt version, because `content/` holds items from more than one.
    # A pooled rate cannot tell whether the no-verbatim rule worked: items an
    # earlier selection published and the current one does not keep the
    # summaries they were written with, and they are the majority here.
    per_version: dict[str, dict] = {}
    for version, n in by_version.items():
        over = [r for r in rows if r["prompt_version"] == version]
        prose = [r for r in over if r["prose_words"] >= threshold]
        per_version[version or "unknown"] = {
            "summaries": n,
            "over_threshold": len(over),
            "over_threshold_prose_only": len(prose),
            "share_prose_only": round(len(prose) / n, 4) if n else None,
            "longest_prose_run": max((r["prose_words"] for r in prose), default=0),
        }

    prose_rows.sort(key=lambda r: -r["run_length"])
    prose_by_version: dict[str, int] = {}
    for r in prose_rows:
        key = r["prompt_version"] or "unknown"
        prose_by_version[key] = prose_by_version.get(key, 0) + 1

    return {
        "threshold_words": threshold,
        "summaries_checked": checked,
        # The measurement to act on: proper nouns, acronyms, numbers and
        # controlled-vocabulary terms removed from both sides first, because
        # PRD 5.5 requires those to be carried over unchanged.
        "prose_only": {
            "threshold_words": prose_threshold,
            "over_threshold": len(prose_rows),
            "share": round(len(prose_rows) / checked, 4) if checked else None,
            "longest_run": prose_rows[0]["run_length"] if prose_rows else 0,
            "by_prompt_version": prose_by_version,
            "violations": prose_rows[:25],
        },
        "by_prompt_version": per_version,
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
    ap.add_argument("--n", type=int, default=8, help="raw consecutive-word threshold")
    ap.add_argument("--prose-n", type=int, default=12, help="threshold after facts are stripped")
    ap.add_argument("--json", help="also write the full result here")
    a = ap.parse_args()

    result = measure(a.n, a.prose_n)
    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    summary = {k: v for k, v in result.items() if k != "violations"}
    summary["prose_only"] = {
        k: v for k, v in summary["prose_only"].items() if k != "violations"
    }
    print(json.dumps(summary, indent=2))
    for row in result["prose_only"]["violations"][:10]:
        print(f"\n  PROSE {row['work_key']}  ({row['run_length']} words)")
        print(f"    {row['matched_text']}")
    print("\n--- raw measurement, facts included (for comparison) ---")
    for row in result["violations"][:10]:
        print(
            f"\n  {row['work_key']}  ({row['run_length']} tokens, "
            f"{row['prose_words']} non-numeric)"
        )
        print(f"    {row['matched_text']}")


if __name__ == "__main__":
    main()
