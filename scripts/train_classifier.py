"""Train the relevance classifier (PRD §5.4).

Local embeddings (``BAAI/bge-base-en-v1.5``) → logistic regression. The output is
a **calibrated probability**, which is the practical reason this beats a seed
centroid: a threshold on it can be interpreted and defended, whereas a threshold
on cosine similarity is a number someone picked.

Reports 20% holdout AUC, precision/recall, and — because that is the failure mode
the design predicts (PRD §5.4, §10) — **per-source recall**, so journal bias
against arXiv papers shows up as a number rather than a suspicion.

Writes ``models/clf-{date}.joblib`` and a sibling ``.json`` with the metrics and
the training metadata. ``provenance.classifier_version`` records which one ran.

Usage:
    uv run python scripts/train_classifier.py [--holdout 0.2]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import cfg  # noqa: E402
from pipeline.filters.embed import embed  # noqa: E402
from pipeline.paths import MODELS, RUNS  # noqa: E402

TRAINSET = RUNS / "trainset" / "trainset.jsonl"


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def train(holdout: float, seed: int, out_dir: Path) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    rows = load_rows(TRAINSET)
    if not rows:
        raise SystemExit(f"no training data at {TRAINSET}")
    print(f"loaded {len(rows)} examples")

    texts = [r["text"] for r in rows]
    y = np.array([r["label"] for r in rows])
    sources = np.array([r["source"] for r in rows])

    print("embedding (cached on disk; first run downloads the model)...")
    X = embed(texts, show_progress=True)
    print(f"embeddings: {X.shape}")

    idx = np.arange(len(rows))
    tr, te = train_test_split(idx, test_size=holdout, random_state=seed, stratify=y)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X[tr], y[tr])

    proba = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y[te], proba))
    ap = float(average_precision_score(y[te], proba))

    threshold = float(cfg("classifier.threshold", 0.5))
    pred = (proba >= threshold).astype(int)
    prec = float(precision_score(y[te], pred, zero_division=0))
    rec = float(recall_score(y[te], pred, zero_division=0))

    # Per-source recall on the holdout: the predicted failure mode is that
    # journal-heavy training scores arXiv urban papers too low.
    per_source = {}
    for src in sorted(set(sources[te])):
        mask = sources[te] == src
        if not mask.any():
            continue
        entry = {"n": int(mask.sum()), "mean_proba": round(float(proba[mask].mean()), 4)}
        if y[te][mask].max() == 1:
            entry["recall_at_threshold"] = round(
                float(recall_score(y[te][mask], pred[mask], zero_division=0)), 4
            )
        else:
            entry["false_positive_rate"] = round(float(pred[mask].mean()), 4)
        per_source[src] = entry

    p_curve, r_curve, t_curve = precision_recall_curve(y[te], proba)
    curve = [
        {"threshold": round(float(t), 3), "precision": round(float(p), 4), "recall": round(float(r), 4)}
        for p, r, t in zip(p_curve[:-1:20], r_curve[:-1:20], t_curve[::20])
    ]

    version = f"clf-{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{version}.joblib"

    import joblib

    joblib.dump(clf, model_path)

    meta = {
        "version": version,
        "trained_at": date.today().isoformat(),
        "embedding_model": cfg("embedding.model"),
        "embedding_dim": int(X.shape[1]),
        "n_examples": len(rows),
        "n_train": int(len(tr)),
        "n_holdout": int(len(te)),
        "holdout_fraction": holdout,
        "random_state": seed,
        "threshold": threshold,
        "metrics": {
            "auc": round(auc, 4),
            "average_precision": round(ap, 4),
            "precision_at_threshold": round(prec, 4),
            "recall_at_threshold": round(rec, 4),
        },
        "per_source": per_source,
        "pr_curve": curve,
        "trainset_meta": json.loads(
            (RUNS / "trainset" / "trainset.meta.json").read_text(encoding="utf-8")
        )
        if (RUNS / "trainset" / "trainset.meta.json").exists()
        else {},
    }
    (out_dir / f"{version}.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    print(json.dumps({k: meta[k] for k in ("version", "metrics", "per_source")}, indent=2))
    print(f"model: {model_path}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", type=float, default=float(cfg("classifier.holdout_fraction", 0.2)))
    p.add_argument("--seed", type=int, default=int(cfg("classifier.random_state", 42)))
    p.add_argument("--out", default=str(MODELS))
    a = p.parse_args()
    train(a.holdout, a.seed, Path(a.out))


if __name__ == "__main__":
    main()
