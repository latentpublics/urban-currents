"""Relevance classifier (PRD §5.4).

Logistic regression over local embeddings, trained on what the field already
agrees is urban studies (whitelist journals) rather than on a hand-picked seed
set. Output is a **calibrated probability**, which is why a threshold here is
interpretable in a way a cosine similarity never is.

If no trained model exists — or embeddings are unavailable — we fall back to a
transparent keyword-density heuristic and record ``classifier_version:
"heuristic-v0"`` on every item it touches. The fallback keeps the pipeline
runnable; the version string keeps the report honest about which one ran.

**And the fallback is now loud** (phase 0X, X1). It used to happen in silence:
`except Exception` swallowed the reason, the stage reported OK, and the day
went out. That is worse than it sounds, because the heuristic is not on the
same scale as the trained model — it saturates at 0.95 but scores a typical
abstract between 0.05 and 0.4, while `selection.arxiv.floor` is 0.80 and was
calibrated against the trained model. So a silent fallback does not degrade the
arXiv path, it **empties** it, and the issue quietly becomes journal-only while
still claiming to be a scan of the field. That is the same failure
`REQUIRED_SOURCES` exists for, one layer down, so it is reported the same way:
the reason travels with the prediction, `stage_classify` writes it into the run
and marks the stage DEGRADED, and `looked()` refuses the day.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..config import arxiv_vocab, cfg
from ..models import Item
from ..paths import MODELS
from .embed import embed, embed_text
from .gate import _compile


@dataclass
class Prediction:
    probabilities: list[float]
    version: str
    # Why the trained model was not used, when it was not (0X, X1). `None`
    # means it was. This is the sentence a person needs and the one the bare
    # `except Exception` below used to eat.
    fallback_reason: Optional[str] = None


class HeuristicClassifier:
    """Keyword-density stand-in. Deliberately crude and clearly labelled."""

    version = "heuristic-v0"

    def __init__(self, reason: Optional[str] = None) -> None:
        self._patterns = _compile(arxiv_vocab().get("keywords", []) or [])
        # Carried so the run can say *why* it is holding a keyword counter
        # instead of the classifier the floor was calibrated against.
        self.reason = reason or "no trained classifier was available"

    def predict(self, items: Sequence[Item]) -> Prediction:
        probs = []
        for it in items:
            text = f"{it.bibliography.title}\n{it.bibliography.abstract or ''}"
            hits = sum(1 for p in self._patterns if p.search(text))
            # 0 hits -> 0.05, saturating near 0.95 at ~8 distinct keywords.
            probs.append(round(min(0.95, 0.05 + 0.115 * hits), 4))
        return Prediction(
            probabilities=probs, version=self.version, fallback_reason=self.reason
        )


class TrainedClassifier:
    def __init__(self, model, version: str, meta: dict):
        self._model = model
        self.version = version
        self.meta = meta

    @classmethod
    def load(cls, path: Path) -> "TrainedClassifier":
        import joblib

        model = joblib.load(path)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(model, path.stem, meta)

    def predict(self, items: Sequence[Item]) -> Prediction:
        texts = [embed_text(it.bibliography.title, it.bibliography.abstract) for it in items]
        if not texts:
            return Prediction(probabilities=[], version=self.version)
        X = embed(texts)
        probs = self._model.predict_proba(X)[:, 1]
        return Prediction(probabilities=[round(float(p), 4) for p in probs], version=self.version)


def latest_model_path() -> Optional[Path]:
    """The model to use, named explicitly where possible.

    Picking the lexicographically last file was fine when there was one model a
    day; with variants on disk (``clf-v1-…``, ``clf-v2-…``, ``clf-v3-…``) it
    silently selects ``v3`` — which the comparison showed is the *worst* of the
    three. Which model is in production is a decision, so it is written down in
    ``classifier.model_version`` rather than inferred from a filename sort.
    """
    pinned = cfg("classifier.model_version")
    if pinned:
        p = MODELS / f"{pinned}.joblib"
        if p.exists():
            return p
        # A pin that does not resolve is a configuration error worth seeing.
        raise FileNotFoundError(
            f"classifier.model_version={pinned!r} but {p} does not exist; "
            f"train it or update config/pipeline.yaml"
        )
    candidates = sorted(MODELS.glob("clf-*.joblib"))
    return candidates[-1] if candidates else None


def load_classifier(allow_fallback: bool = True):
    """Newest ``models/clf-*.joblib``, else the heuristic — and say which.

    The fallback still happens, because a stale joblib or a missing embedding
    package must not stop a day's run. What changed in 0X is that the reason
    survives: it is attached to the `HeuristicClassifier`, travels out on the
    `Prediction`, and is written into the run by `stage_classify`.
    """
    p = latest_model_path()
    reason: Optional[str] = None
    if p is None:
        reason = f"no trained model in {MODELS}"
    else:
        try:
            clf = TrainedClassifier.load(p)
            # Fail fast if embeddings are unavailable, rather than mid-run.
            embed([embed_text("probe", "probe")])
            return clf
        except Exception as e:
            # Missing embeddings or a stale joblib must not stop the run.
            if not allow_fallback:
                raise
            reason = f"{p.name} could not be used: {type(e).__name__}: {e}"
    if not allow_fallback:
        raise RuntimeError(f"no trained classifier available ({reason})")
    return HeuristicClassifier(reason)


def score_items(items: Sequence[Item], clf=None) -> Prediction:
    clf = clf or load_classifier()
    pred = clf.predict(items)
    for it, p in zip(items, pred.probabilities):
        it.scores.relevance = p
        it.scores.components.relevance = p
        it.provenance.classifier_version = pred.version
    return pred
