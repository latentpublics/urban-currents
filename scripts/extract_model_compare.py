"""Can extraction move to a cheaper model? (phase 0L, N0)

**The rule is written here, above the code, and it was written before anything
ran.** The 60-day backfill costs an estimated $8.64 against a $9.00 ceiling, and
extraction is 46% of cumulative LLM spend (529 calls, $1.085 of $2.372) — so if
extraction can move to `flash-lite` the backfill fits with room, and if it
cannot the backfill has to be planned around the full price instead.

## The pre-registered rule

Take items already extracted with `gemini-3.5-flash`, extract them again with
`gemini-3.5-flash-lite`, and compare:

| measurement | adopt if |
|---|---|
| matched controlled-vocabulary tags | **does not fall** |
| `unmatched` candidates | reported, not a gate |
| Jaccard between the two versions' tag sets | **>= 0.80** |

Adopt only if the matched-tag count holds **and** Jaccard clears 0.80. A cheaper
model that finds the same *number* of tags but different ones has not preserved
the signal, it has replaced it — and the overlay vocabulary is what every entity
page and the canon affinity are built from.

`summarize` is not touched. Those are sentences a reader sees.

## Not polluting the cache

The cache key is `{prompt_version}/{work_key}` — **the model is not part of it.**
Worth stating plainly: as things stand, two models writing under the same prompt
version overwrite each other's entries and nothing notices. So this comparison
writes into a scratch cache directory rather than the real one, and the finding
about the key is reported rather than quietly worked around.

Usage:
    uv run python scripts/extract_model_compare.py --n 40
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.config import cfg  # noqa: E402
from pipeline.linking.extract import extract_overlay  # noqa: E402
from pipeline.llm import LLMClient  # noqa: E402
from pipeline.metrics import Run  # noqa: E402


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def tag_set(payload: dict) -> set:
    """Every candidate string the model produced, facet-qualified."""
    out = set()
    for facet, values in (payload or {}).items():
        for v in values or []:
            out.add(f"{facet}:{v}")
    return out


def matched_tags(payload: dict) -> tuple[set, set]:
    """(matched against the controlled vocabulary, unmatched).

    The matched set is what actually reaches an item; unmatched candidates go to
    `unmatched.jsonl` for a human to promote later. A model that emits more
    strings but fewer *matches* has made the overlay worse, not better.
    """
    from pipeline.linking.vocab_match import match_facet

    matched, unmatched = set(), set()
    for facet, values in (payload or {}).items():
        if not values:
            continue
        try:
            result = match_facet(values, facet)
        except Exception:
            unmatched |= {f"{facet}:{v}" for v in values}
            continue
        matched |= {f"{facet}:{r.id}" for r in result.refs}
        unmatched |= {f"{facet}:{v}" for v in result.unmatched}
    return matched, unmatched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    baseline_model = cfg("llm.extract.model", cfg("llm.model", "gemini-3.5-flash"))
    version = cfg("llm.extract.prompt_version", "extract/overlay@0.1.0")

    # Only items that already have a cached extraction under this prompt
    # version. Comparing against a fresh run of the same model would measure
    # sampling noise on top of the model change.
    import re as _re

    safe_version = _re.sub(r"[^A-Za-z0-9._@-]", "_", version)
    cache_dir = paths.LLM_CACHE / safe_version
    cached_keys = set()
    if cache_dir.exists():
        for p in cache_dir.glob("*.json"):
            cached_keys.add(p.stem)

    items = []
    for item in store.iter_items():
        if not item.bibliography.abstract:
            continue
        key = _re.sub(r"[^A-Za-z0-9._-]", "_", item.work_key)
        if key in cached_keys:
            items.append(item)
        if len(items) >= args.n:
            break

    print(f"{len(items)} items with a cached {baseline_model} extraction")
    if len(items) < 10:
        print("BLOCKED: too few cached extractions to compare against")
        return

    from datetime import date as _date

    run = Run.for_date(_date.today())

    # Baseline reads the existing cache: no calls, no cost.
    base_client = LLMClient(task="extract")
    base, _ = extract_overlay(items, run, client=base_client)

    # Candidate writes to a scratch cache, so the real one keeps only `flash`.
    scratch = Path(tempfile.mkdtemp(prefix="uc-extract-cmp-"))
    original_cache = paths.LLM_CACHE
    paths.LLM_CACHE = scratch  # type: ignore[misc]
    try:
        lite_client = LLMClient(task="extract", model=args.model)
        lite, lite_stats = extract_overlay(items, run, client=lite_client)
        throttled = lite_client.rate_limited
        slept = lite_client.backoff_s
    finally:
        paths.LLM_CACHE = original_cache  # type: ignore[misc]

    rows = []
    for item in items:
        k = item.work_key
        if k not in base or k not in lite:
            continue
        b_matched, b_unmatched = matched_tags(base[k])
        l_matched, l_unmatched = matched_tags(lite[k])
        rows.append({
            "work_key": k,
            "base_matched": len(b_matched),
            "lite_matched": len(l_matched),
            "base_unmatched": len(b_unmatched),
            "lite_unmatched": len(l_unmatched),
            "jaccard_raw": round(jaccard(tag_set(base[k]), tag_set(lite[k])), 4),
            "jaccard_matched": round(jaccard(b_matched, l_matched), 4),
        })

    n = len(rows)
    if not n:
        print(
            f"BLOCKED: no comparable pairs — baseline {len(base)}, "
            f"candidate {len(lite)}, candidate status {lite_stats.get('status')}, "
            f"stop {lite_stats.get('stop_reason')}"
        )
        return

    def mean(key: str) -> float:
        return round(sum(r[key] for r in rows) / n, 4)

    base_matched = sum(r["base_matched"] for r in rows)
    lite_matched = sum(r["lite_matched"] for r in rows)
    j_matched = mean("jaccard_matched")
    j_raw = mean("jaccard_raw")

    matched_holds = lite_matched >= base_matched
    jaccard_holds = j_matched >= 0.80
    adopt = bool(matched_holds and jaccard_holds)

    result = {
        "n": n,
        "baseline_model": baseline_model,
        "candidate_model": args.model,
        "prompt_version": version,
        # Reported because it decides whether adopting is even safe.
        "model_in_cache_key": False,
        "matched_tags": {"baseline": base_matched, "candidate": lite_matched},
        "unmatched": {
            "baseline": sum(r["base_unmatched"] for r in rows),
            "candidate": sum(r["lite_unmatched"] for r in rows),
        },
        "jaccard_matched_mean": j_matched,
        "jaccard_raw_mean": j_raw,
        "matched_holds": matched_holds,
        "jaccard_holds": jaccard_holds,
        "adopt": adopt,
        "throttled": throttled,
        "backoff_s": slept,
        "candidate_status": lite_stats.get("status"),
        "rule": "adopt only if matched tags do not fall AND jaccard(matched) >= 0.80",
        "rows": rows,
    }
    out = paths.RUNS / "extract_model_compare.json"
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"\nn = {n} items")
    print(
        f"matched tags     {base_matched} ({baseline_model}) -> "
        f"{lite_matched} ({args.model})   {'PASS' if matched_holds else 'FAIL'}"
    )
    print(
        f"unmatched        {result['unmatched']['baseline']} -> "
        f"{result['unmatched']['candidate']}"
    )
    print(f"jaccard matched  {j_matched}   {'PASS' if jaccard_holds else 'FAIL'} (>= 0.80)")
    print(f"jaccard raw      {j_raw}")
    print(f"throttled        {throttled} time(s), {slept:.0f}s asleep")
    print(f"\nADOPT: {adopt}")
    print(f"-> {out}")
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
