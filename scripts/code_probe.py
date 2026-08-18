"""Why does a paper that releases code never get published? (phase 0L, N2)

0k measured it: over five prepared days, **61 arXiv candidates carried a code
signal and 0 survived selection.** Their relevance runs p50 0.054, max 0.463,
against a 0.80 publication floor. A digest of urban data science has never once
published a paper that releases its code, and the home page counts a
`with code` badge.

Two explanations fit, and they need opposite responses:

- the classifier is wrong, and a stylistic artefact is pushing these papers
  down — the training set is 72% journal articles and journals mention code at
  1.3% against arXiv's 14.6%, so "mentions code" is weakly *negative* evidence;
- the classifier is right, and papers that release code mostly are not the kind
  of urban research we cover — in which case **the home page is what is wrong.**

## The rule, written before looking

1. Count the labelled arXiv items that mention code.
2. **Fewer than 8 and the labels cannot answer this** — they are a top-15 sample
   and these papers sit at p50 0.054, so they were never eligible to appear in
   it. Build a probe instead.
3. 8 or more: compare keep rates with and without a code signal, with Fisher's
   exact test.

| result | conclusion |
|---|---|
| keep rate with code **equal or higher** | the classifier is wrong; a style artefact |
| keep rate **clearly lower** | the classifier is right, and the home page copy is wrong |
| not significant | neither claim; build the probe |

## The probe, if one is needed

`runs/labels/code_probe.jsonl` — **a third label file, and it may never be
pooled with the other two.** `relevance.jsonl` is a ranked top-N and
`affinity_probe.jsonl` is band-stratified over canon affinity; this one is
stratified over *relevance* among code-bearing candidates. Three samples, three
sampling frames, three different questions. Pooling any two of them produces a
number that looks fine and means nothing, which is why the write guard refuses
by name.

Stratified high/mid/low rather than drawn from the bottom: sampling only
low-scoring papers would confirm that low scores are low, which is not the
question.

**This prepares the pool. It does not label anything.**
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.labeling import load_labels, superseded  # noqa: E402
from pipeline.models import Item  # noqa: E402
from pipeline.signals import code_signal  # noqa: E402

BANDS = (("high", 0.50, 1.01), ("mid", 0.20, 0.50), ("low", 0.0, 0.20))
SAMPLING = "code_stratified"


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p for [[a, b], [c, d]]. No scipy."""
    n = a + b + c + d
    if n == 0:
        return 1.0

    def hyper(x: int) -> float:
        return math.comb(a + b, x) * math.comb(c + d, a + c - x) / math.comb(n, a + c)

    observed = hyper(a)
    lo, hi = max(0, a + c - (c + d)), min(a + b, a + c)
    total = sum(p for x in range(lo, hi + 1) if (p := hyper(x)) <= observed * (1 + 1e-9))
    return round(min(1.0, total), 6)


def has_code(item: Item) -> bool:
    sig = code_signal(item)
    return bool(sig and sig.value is True)


def _candidate_items() -> dict[str, Item]:
    """Every arXiv candidate on disk, published or not."""
    index: dict[str, Item] = {}
    for run_dir in sorted(paths.RUNS.glob("run_*")):
        for name in ("classify.jsonl", "labeling_pool.jsonl"):
            path = run_dir / "stages" / name
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = Item.model_validate_json(line)
                except Exception:  # noqa: BLE001
                    continue
                if item.work_key.startswith("arxiv:"):
                    index.setdefault(item.work_key, item)
    for item in store.iter_items():
        if item.work_key.startswith("arxiv:"):
            index.setdefault(item.work_key, item)
    return index


def step_one_can_the_labels_answer() -> dict:
    """Count labelled arXiv items carrying a code signal."""
    index = _candidate_items()
    rows = []
    for label_row in superseded(load_labels("relevance")):
        if label_row.get("source") != "arxiv":
            continue
        item = store.load_item(label_row["work_key"]) or index.get(label_row["work_key"])
        if item is None:
            continue
        rows.append({
            "work_key": label_row["work_key"],
            "label": label_row.get("label"),
            "keep": label_row.get("label") == "keep",
            "code": has_code(item),
            "score": label_row.get("score"),
        })

    with_code = [r for r in rows if r["code"]]
    without = [r for r in rows if not r["code"]]
    k_with = sum(1 for r in with_code if r["keep"])
    k_without = sum(1 for r in without if r["keep"])

    answerable = len(with_code) >= 8
    verdict = "labels cannot answer (n < 8) — build the probe"
    p = None
    if answerable:
        p = fisher_exact(
            k_with, len(with_code) - k_with, k_without, len(without) - k_without
        )
        rate_with = k_with / len(with_code)
        rate_without = k_without / len(without) if without else 0.0
        if p < 0.05 and rate_with < rate_without:
            verdict = "classifier is right — the home page copy is what is wrong"
        elif rate_with >= rate_without:
            verdict = "classifier is wrong — a stylistic artefact"
        else:
            verdict = "not significant — neither claim; build the probe"

    return {
        "n_arxiv_labels": len(rows),
        "with_code": len(with_code),
        "without_code": len(without),
        "keeps_with_code": k_with,
        "keeps_without_code": k_without,
        "keep_rate_with_code": round(k_with / len(with_code), 4) if with_code else None,
        "keep_rate_without_code": round(k_without / len(without), 4) if without else None,
        "fisher_p": p,
        "answerable": answerable,
        "verdict": verdict,
    }


def build_pool(per_band: int = 10) -> dict:
    """Stratify code-bearing arXiv candidates over relevance and write the pool."""
    index = _candidate_items()
    already = {r["work_key"] for r in superseded(load_labels("relevance"))}
    already |= {r["work_key"] for r in load_labels("affinity_probe")}

    scored = []
    for key, item in index.items():
        if key in already or not has_code(item):
            continue
        score = float(getattr(item.scores, "relevance", 0.0) or 0.0)
        scored.append((score, item))
    scored.sort(key=lambda t: -t[0])

    rows = []
    per_band_counts = {}
    for name, lo, hi in BANDS:
        in_band = [(s, it) for s, it in scored if lo <= s < hi]
        per_band_counts[name] = len(in_band)
        step = max(1, len(in_band) // per_band) if in_band else 1
        picked = in_band[::step][:per_band]
        for rank, (score, item) in enumerate(picked, start=1):
            sig = code_signal(item)
            rows.append({
                "work_key": item.work_key,
                "band": name,
                "rank_in_band": rank,
                "score": round(score, 4),
                "title": item.bibliography.title,
                "date": str(item.first_published) if item.first_published else None,
                "code_basis": getattr(sig, "basis", None),
                "source": "arxiv",
                # The sampling frame, stated on every row. Three label files,
                # three frames; this is what makes pooling them detectable.
                "sampling": SAMPLING,
                "not_for_precision_at_k": True,
            })

    pool_path = paths.LABELS / "code_probe_pool.jsonl"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "pool": str(pool_path),
        "n": len(rows),
        "per_band_available": per_band_counts,
        "per_band_picked": {b: sum(1 for r in rows if r["band"] == b) for b, _, _ in BANDS},
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    step_one = step_one_can_the_labels_answer()
    print("Step 1 — can the labels answer?")
    for k in (
        "n_arxiv_labels", "with_code", "without_code",
        "keep_rate_with_code", "keep_rate_without_code", "fisher_p",
    ):
        print(f"  {k:24} {step_one[k]}")
    print(f"  verdict: {step_one['verdict']}")

    result = {"step_one": step_one}
    if not step_one["answerable"]:
        pool = build_pool()
        result["pool"] = pool
        print(f"\nStep 2 — probe pool prepared: {pool['n']} items")
        print(f"  available per band: {pool['per_band_available']}")
        print(f"  picked per band:    {pool['per_band_picked']}")
        print(f"  -> {pool['pool']}")

    out = paths.RUNS / "code_probe.json"
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
