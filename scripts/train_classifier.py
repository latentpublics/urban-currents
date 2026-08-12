"""Train the relevance classifier and judge it on the task it actually has.

Local embeddings (``BAAI/bge-base-en-v1.5``) → logistic regression. The output is
a **calibrated probability**, which is the practical reason this beats a seed
centroid: a threshold on it can be interpreted and defended.

**The headline metric is arXiv-only.** Phase 0 reported AUC 0.976 on a holdout
that was mostly whitelist-journal articles against random ML papers — a nearly
self-evident task, and one the classifier no longer performs: after the entry
paths split (N4) a journal article enters on membership and never reaches the
model. What remains is arXiv-urban vs arXiv-other, which is materially harder,
and that is what ``metrics`` now reports. Journal numbers are kept as a sanity
check, clearly labelled as such.

Evaluation uses the shared ``runs/trainset/eval_arxiv.jsonl`` — built once,
excluded from every training set — so variants are comparable.

Usage:
    uv run python scripts/train_classifier.py --variant v2
    uv run python scripts/train_classifier.py --variant v1 --no-save
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import cfg  # noqa: E402
from pipeline.filters.embed import embed  # noqa: E402
from pipeline.paths import MODELS, RUNS  # noqa: E402

TRAINSET_DIR = RUNS / "trainset"
EVAL_PATH = TRAINSET_DIR / "eval_arxiv.jsonl"

SWEEP_THRESHOLDS = [round(0.05 * i, 2) for i in range(2, 19)]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def trainset_path(variant: str) -> Path:
    p = TRAINSET_DIR / variant / "trainset.jsonl"
    # Phase 0's single trainset lives one level up; fall back so the old file
    # stays usable without being copied around.
    return p if p.exists() else TRAINSET_DIR / "trainset.jsonl"


def _metrics_at(y, proba, threshold: float) -> dict[str, float]:
    from sklearn.metrics import precision_score, recall_score

    pred = (proba >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "flagged_rate": round(float(pred.mean()), 4),
    }


def evaluate(clf, rows: list[dict], threshold: float) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not rows:
        return {}
    X = embed([r["text"] for r in rows])
    y = np.array([r["label"] for r in rows])
    proba = clf.predict_proba(X)[:, 1]

    out: dict[str, Any] = {
        "n": len(rows),
        "n_positive": int(y.sum()),
        "auc": round(float(roc_auc_score(y, proba)), 4) if len(set(y)) > 1 else None,
        "average_precision": (
            round(float(average_precision_score(y, proba)), 4) if len(set(y)) > 1 else None
        ),
        "at_threshold": {str(threshold): _metrics_at(y, proba, threshold)},
        "mean_proba_positive": round(float(proba[y == 1].mean()), 4) if (y == 1).any() else None,
        "mean_proba_negative": round(float(proba[y == 0].mean()), 4) if (y == 0).any() else None,
    }
    out["sweep"] = [
        {"threshold": t, **_metrics_at(y, proba, t)} for t in SWEEP_THRESHOLDS
    ]
    return out


def train(variant: str, seed: int, threshold: float, save: bool) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    rows = load_rows(trainset_path(variant))
    eval_rows = load_rows(EVAL_PATH)
    eval_ids = {r["id"] for r in eval_rows}

    leaked = [r for r in rows if r["id"] in eval_ids]
    rows = [r for r in rows if r["id"] not in eval_ids]
    if leaked:
        print(f"dropped {len(leaked)} rows that also appear in the eval set")

    print(f"variant {variant}: {len(rows)} training rows, {len(eval_rows)} eval rows")

    X = embed([r["text"] for r in rows], show_progress=True)
    y = np.array([r["label"] for r in rows])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X, y)

    # The headline: arXiv-urban vs arXiv-other on the shared holdout.
    arxiv_eval = evaluate(clf, eval_rows, threshold)

    # Sanity check only — this is the task the classifier no longer performs.
    journal_rows = [r for r in rows if r["source"] == "journal"]
    journal_check: dict[str, Any] = {}
    if journal_rows:
        sample = journal_rows[:400]
        Xj = embed([r["text"] for r in sample])
        pj = clf.predict_proba(Xj)[:, 1]
        journal_check = {
            "n": len(sample),
            "mean_proba": round(float(pj.mean()), 4),
            "share_above_threshold": round(float((pj >= threshold).mean()), 4),
            "note": "in-sample sanity check, not a performance claim",
        }

    version = f"clf-{variant}-{date.today().isoformat()}"
    meta = {
        "version": version,
        "variant": variant,
        "trained_at": date.today().isoformat(),
        "embedding_model": cfg("embedding.model"),
        "embedding_dim": int(X.shape[1]),
        "n_train": len(rows),
        "threshold": threshold,
        "random_state": seed,
        "headline_task": "arxiv_urban_vs_arxiv_other",
        "metrics": arxiv_eval,
        "journal_sanity_check": journal_check,
        "trainset_meta": json.loads(
            (trainset_path(variant).parent / "trainset.meta.json").read_text(encoding="utf-8")
        )
        if (trainset_path(variant).parent / "trainset.meta.json").exists()
        else {},
        "eval_meta": json.loads((TRAINSET_DIR / "eval_arxiv.meta.json").read_text(encoding="utf-8"))
        if (TRAINSET_DIR / "eval_arxiv.meta.json").exists()
        else {},
    }

    if save:
        import joblib

        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, MODELS / f"{version}.joblib")
        (MODELS / f"{version}.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"model: {MODELS / (version + '.joblib')}")

    print(
        json.dumps(
            {
                "version": version,
                "arxiv_auc": arxiv_eval.get("auc"),
                "arxiv_ap": arxiv_eval.get("average_precision"),
                "at_threshold": arxiv_eval.get("at_threshold"),
                "journal_sanity_check": journal_check,
            },
            indent=2,
        )
    )
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="v2")
    p.add_argument("--seed", type=int, default=int(cfg("classifier.random_state", 42)))
    p.add_argument("--threshold", type=float, default=float(cfg("classifier.threshold", 0.35)))
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()
    train(a.variant, a.seed, a.threshold, save=not a.no_save)


if __name__ == "__main__":
    main()
