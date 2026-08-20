"""Does every LLM cache key cover everything that changes the answer? (0T, V4-2)

Three consecutive batches found the same defect in three different places:

  0Q  the **model** was not in the key, so two models writing under one
      `prompt_version` overwrote each other and nothing noticed
  0R  the **retry hint** was not in the key, so a sharpened retry prompt read
      back the answer the vaguer one had produced and the fix looked inert
  0S  the **facts block** was not in the key, so changing what the model was
      shown would have returned the previous day's reasoning

Each was fixed where it was found, which left the next one waiting. This walks
every call site and reports what its key covers, so the answer is a table rather
than another anecdote.

Usage:
    uv run python scripts/cache_key_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths  # noqa: E402

# What each task passes as its caller-supplied label, and what the user prompt
# is actually built from. The digest added in 0T covers model + system + user +
# schema for all of them; this table is what the *label* alone would have
# covered, which is what the three incidents were about.
SITES = [
    {
        "task": "summarize",
        "site": "pipeline/summarize/run.py:152",
        "label": "work_key",
        "user_from": "title + abstract",
        "label_covered_model": False,
        "label_covered_user": False,
    },
    {
        "task": "summarize (retry)",
        "site": "pipeline/summarize/run.py:179",
        "label": "work_key + '.retry'",
        "user_from": "title + abstract",
        "label_covered_model": False,
        "label_covered_user": False,
    },
    {
        "task": "extract",
        "site": "pipeline/linking/extract.py:109",
        "label": "work_key",
        "user_from": "title + abstract",
        "label_covered_model": False,
        "label_covered_user": False,
    },
    {
        "task": "headline",
        "site": "pipeline/summarize/headline.py:181",
        "label": "work_key, or work_key#retry-<hint hash>",
        "user_from": "title + summary.what + summary.why",
        "label_covered_model": False,
        # 0R put the hint in the key; the summary text it reads was still not.
        "label_covered_user": False,
    },
    {
        "task": "synthesis",
        "site": "pipeline/synthesis.py:745",
        "label": "synthesis-<date>-<facts hash>",
        "user_from": "tag groups + highlights",
        "label_covered_model": False,
        "label_covered_user": True,
    },
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("What the caller's own label covered, before 0T:\n")
    print(f"{'task':<20}{'label':<38}{'model?':<9}{'input?':<8}")
    gaps = 0
    for s in SITES:
        model = "yes" if s["label_covered_model"] else "NO"
        user = "yes" if s["label_covered_user"] else "NO"
        gaps += (not s["label_covered_model"]) + (not s["label_covered_user"])
        print(f"{s['task']:<20}{s['label']:<38}{model:<9}{user:<8}")
    print(f"\n{gaps} gap(s) across {len(SITES)} call site(s).")

    print("\nAfter 0T every key also carries a digest of "
          "(model, system, user, schema),")
    print("so the columns above are covered for every site without any of them")
    print("having to remember. Verified live:\n")

    from pipeline.llm import request_digest

    base = request_digest("m", "sys", "user", None)
    checks = [
        ("model", request_digest("m2", "sys", "user", None)),
        ("system prompt", request_digest("m", "sys2", "user", None)),
        ("user prompt", request_digest("m", "sys", "user2", None)),
        ("schema", request_digest("m", "sys", "user", {"a": 1})),
    ]
    for name, other in checks:
        print(f"  changing the {name:<14} changes the key: {other != base}")
    print(f"  identical request is stable:       "
          f"{request_digest('m', 'sys', 'user', None) == base}")

    cache = paths.LLM_CACHE
    if cache.exists():
        # A request-keyed file ends `.<10 hex>.json`. Counting dots does not
        # work: `doi_10.1016_j.trc.2026.105768.json` has five of them and is
        # legacy-keyed.
        import re as _re

        keyed_re = _re.compile(r"\.[0-9a-f]{10}\.json$")
        files = list(cache.rglob("*.json"))
        keyed = sum(1 for f in files if keyed_re.search(f.name))
        legacy = len(files) - keyed
        print(f"\ncache entries: {legacy} legacy-keyed, {keyed} request-keyed")
        print("Legacy entries are re-keyed as they are read — a one-time")
        print("grandfathering, because they were written from inputs that are")
        print("still on disk. Invalidating them instead would re-summarise")
        print("2,224 items to learn nothing.")


if __name__ == "__main__":
    main()
