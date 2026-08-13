"""T5: what adopting the rebuilt whitelist would cost the classifier (phase 0g).

`clf-v2`'s journal positives came from the 159-entry whitelist. The rebuild
takes that to 370 entries and 164 included, which changes the training set — so
the cost should be known before the list is adopted, not after.

**The obvious comparison is not fair, and the reason matters.** The evaluation
set's positives are "an arXiv paper a whitelist journal accepted", and
"whitelist" means whichever list was in force when the set was built. Judging
both models on the old set asks whether the rebuild helps at predicting the old
list's taste; judging both on the new set asks the mirror image. Either alone
flatters one model by construction.

So both models are scored on **both** evaluation sets, and the diagonal is
reported next to the off-diagonal. A model that wins only on its own set has
learnt the set, not the field.

Nothing is adopted. `classifier.model_version` stays `clf-v2-2026-08-13`.

Usage:
    uv run python scripts/compare_classifiers.py --json runs/classifier_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from pipeline.filters.embed import embed  # noqa: E402

SCRATCH = Path(
    r"C:/Users/jour/AppData/Local/Temp/claude/C--Users-jour-Documents-GitHub-urban-currents"
    r"/429ac4c4-2d54-461b-9a76-a12bd04f9a5a/scratchpad"
)


def load_eval(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("text") or " ".join(
            filter(None, [row.get("title"), row.get("abstract")])
        )
        if not text:
            continue
        texts.append(text)
        labels.append(int(row.get("label", 0)))
    return texts, labels


def score(model_path: Path, texts: list[str], labels: list[int], threshold: float) -> dict:
    model = joblib.load(model_path)
    probs = model.predict_proba(embed(texts))[:, 1]
    y = np.array(labels)
    flagged = probs >= threshold
    tp = int(((flagged) & (y == 1)).sum())
    fp = int(((flagged) & (y == 0)).sum())
    fn = int(((~flagged) & (y == 1)).sum())
    return {
        "auc": round(float(roc_auc_score(y, probs)), 4),
        "average_precision": round(float(average_precision_score(y, probs)), 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "flagged_rate": round(float(flagged.mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    models = {
        "clf-v2 (old whitelist)": ROOT / "models/clf-v2-2026-08-13.joblib",
        "clf-v4-rebuilt": ROOT / "models/clf-v4-rebuilt-2026-08-14.joblib",
    }
    evals = {
        "eval built from rebuild v2": ROOT / "runs/trainset/eval_arxiv.jsonl",
        "eval built from old whitelist": SCRATCH / "trainset_v2_backup/eval_arxiv.jsonl",
    }

    results: dict[str, dict] = {}
    for ename, epath in evals.items():
        if not epath.exists():
            results[ename] = {"error": f"missing: {epath}"}
            continue
        texts, labels = load_eval(epath)
        results[ename] = {"n": len(texts), "positives": sum(labels), "models": {}}
        for mname, mpath in models.items():
            if not mpath.exists():
                results[ename]["models"][mname] = {"error": "missing model"}
                continue
            results[ename]["models"][mname] = score(mpath, texts, labels, a.threshold)

    out = {
        "threshold": a.threshold,
        "adopted": "clf-v2-2026-08-13 (unchanged)",
        "fairness_note": (
            "Each evaluation set's positives are defined by whichever whitelist "
            "built it, so each set flatters its own model. The off-diagonal is "
            "the honest comparison."
        ),
        "results": results,
    }
    if a.json:
        Path(a.json).write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

    for ename, r in results.items():
        if "error" in r:
            print(f"{ename}: {r['error']}")
            continue
        print(f"\n{ename}  (n={r['n']}, positives={r['positives']})")
        print(f"   {'model':<26} {'AUC':>7} {'AP':>7} {'prec':>7} {'rec':>7}")
        for mname, m in r["models"].items():
            if "error" in m:
                print(f"   {mname:<26} {m['error']}")
                continue
            print(f"   {mname:<26} {m['auc']:>7} {m['average_precision']:>7} "
                  f"{str(m['precision']):>7} {str(m['recall']):>7}")


if __name__ == "__main__":
    main()
