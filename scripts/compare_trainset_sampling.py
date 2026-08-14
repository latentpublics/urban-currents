"""U4: what a wider whitelist needs from the sampler (phase 0h).

T5 found that widening the whitelist from 95 to 161 journals *reduced* journal
positives from 2,796 to 2,318. The cap was not the only cause — the sampler asks
40 journals at a time sorted by citation count, so a chunk's quota is filled by
whichever of its journals publishes most and is cited hardest, and adding
journals spreads the same quota over more chunks.

Three sampling designs, one shared arXiv evaluation set, no config changed:

| name | journal cap | sampling |
|---|---|---|
| u4-cap2800    | 2,800 | chunked, 40 journals per query — today's behaviour |
| u4-cap4000    | 4,000 | chunked |
| u4-perjournal | 4,000 | one query per journal, ceiling 40, floor 5 |

The headline task is arXiv-urban vs arXiv-other (N3), which is the only job the
classifier still has. Journal composition is reported next to it because a
training set can be 161 journals or five journals wearing 161 names, and the
evaluation metric cannot tell the difference.

Usage:
    uv run python scripts/compare_trainset_sampling.py --json runs/trainset_sampling.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.paths import RUNS  # noqa: E402

VARIANTS = ("u4-cap2800", "u4-cap4000", "u4-perjournal")


def load(variant: str) -> list[dict]:
    path = RUNS / "trainset" / variant / "trainset.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def composition(rows: list[dict]) -> dict:
    """Who the journal positives actually came from.

    `venue_id` is absent from training sets built before U4; those report the
    counts they can and say so, rather than reporting zero distinct journals as
    though the concentration were total.
    """
    journal = [r for r in rows if r["source"] == "journal"]
    venues = Counter(r.get("venue_id") for r in journal if r.get("venue_id"))
    if not venues:
        return {
            "journal_rows": len(journal),
            "note": "no venue_id on these rows — rebuilt before U4 added it",
        }
    counts = sorted(venues.values(), reverse=True)
    return {
        "journal_rows": len(journal),
        "distinct_journals": len(venues),
        "median_rows_per_journal": statistics.median(counts),
        "max_rows_from_one_journal": counts[0],
        "top10_share": round(sum(counts[:10]) / len(journal), 4),
        "journals_contributing_under_5": sum(1 for c in counts if c < 5),
        # Whitelist journals that contributed nothing are the real cost of the
        # chunked sampler: they are on the list and absent from the training set.
        "whitelisted_but_absent": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--no-train", action="store_true", help="Composition only")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import os

    os.environ.setdefault("UC_JOURNALS_FILE", "vocab/sources/journals.rebuilt.v2.yaml")
    from train_classifier import train  # type: ignore

    whitelist = set()
    try:
        import yaml

        wl = yaml.safe_load(
            (ROOT / "vocab/sources/journals.rebuilt.v2.yaml").read_text(encoding="utf-8")
        )
        whitelist = {s["id"] for s in wl["sources"] if s.get("include", True)}
    except Exception:  # noqa: BLE001
        pass

    results: dict[str, dict] = {}
    for variant in VARIANTS:
        rows = load(variant)
        if not rows:
            results[variant] = {"error": "not built"}
            continue
        comp = composition(rows)
        if whitelist and "distinct_journals" in comp:
            present = {r.get("venue_id") for r in rows if r["source"] == "journal"}
            comp["whitelisted_but_absent"] = len(whitelist - present)
        entry = {"composition": comp, "total_rows": len(rows)}
        if not a.no_train:
            meta = train(variant, seed=42, threshold=a.threshold, save=False)
            m = meta["metrics"]
            # `at_threshold` is keyed by the threshold itself, so the figures
            # cannot be read without knowing which one they came from.
            at = (m.get("at_threshold") or {}).get(str(a.threshold), {})
            entry["arxiv_task"] = {
                "auc": m.get("auc"),
                "average_precision": m.get("average_precision"),
                "threshold": a.threshold,
                **at,
            }
            entry["journal_sanity_check"] = meta.get("journal_sanity_check")
        results[variant] = entry

    out = {
        "threshold": a.threshold,
        "whitelist": "vocab/sources/journals.rebuilt.v2.yaml (161 included)",
        "eval_set": "runs/trainset/eval_arxiv.jsonl — shared, excluded from every trainset",
        "adopted": "nothing; classifier.model_version stays clf-v2-2026-08-13",
        "results": results,
    }
    if a.json:
        Path(a.json).write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

    print(f"\n{'variant':<16} {'rows':>6} {'journal':>8} {'venues':>7} {'absent':>7} "
          f"{'top10':>7} {'AUC':>7} {'AP':>7} {'prec':>7} {'rec':>7}")
    for variant, r in results.items():
        if "error" in r:
            print(f"{variant:<16} {r['error']}")
            continue
        c, t = r["composition"], r.get("arxiv_task", {})
        print(
            f"{variant:<16} {r['total_rows']:>6} {c['journal_rows']:>8} "
            f"{c.get('distinct_journals', '—'):>7} {str(c.get('whitelisted_but_absent')):>7} "
            f"{c.get('top10_share', '—'):>7} {str(t.get('auc')):>7} "
            f"{str(t.get('average_precision')):>7} {str(t.get('precision')):>7} "
            f"{str(t.get('recall')):>7}"
        )


if __name__ == "__main__":
    main()
