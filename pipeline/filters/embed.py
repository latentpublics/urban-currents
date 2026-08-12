"""Local text embeddings (PRD §5.4).

``BAAI/bge-base-en-v1.5`` via sentence-transformers, on CPU. Local rather than
hosted for three reasons: one less API key, zero marginal cost so backfills and
retraining are free, and — the one that actually matters — the same input gives
the same vector, so classifier experiments reproduce.

``config/pipeline.yaml: embedding.provider`` is the swap point for a hosted API
later. Phase 0 default is local.
"""

from __future__ import annotations

import hashlib
import functools
from typing import Optional, Sequence

import numpy as np

from ..config import cfg
from ..paths import ROOT

CACHE_DIR = ROOT / ".cache" / "embeddings"


class EmbeddingUnavailable(RuntimeError):
    """sentence-transformers (or its model) is not installed/downloadable."""


def embed_text(item_title: str, abstract: Optional[str]) -> str:
    """The exact string the classifier sees. Kept in one place so training and
    inference cannot drift apart."""
    return f"{item_title.strip()}\n\n{(abstract or '').strip()}".strip()


@functools.lru_cache(maxsize=1)
def _model():
    name = cfg("embedding.model", "BAAI/bge-base-en-v1.5")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - environment dependent
        raise EmbeddingUnavailable(
            "sentence-transformers is not installed; `uv sync --extra embed`"
        ) from e
    try:
        return SentenceTransformer(name, device="cpu")
    except Exception as e:  # pragma: no cover - network/model dependent
        raise EmbeddingUnavailable(f"could not load embedding model {name}: {e}") from e


def _cache_key(text: str) -> str:
    name = cfg("embedding.model", "BAAI/bge-base-en-v1.5")
    return hashlib.sha256(f"{name}\x00{text}".encode("utf-8")).hexdigest()


def embed(texts: Sequence[str], use_cache: bool = True, show_progress: bool = False) -> np.ndarray:
    """Embed texts, caching each vector on disk by (model, text) hash."""
    if cfg("embedding.provider", "local") != "local":
        raise EmbeddingUnavailable(
            f"embedding.provider={cfg('embedding.provider')} is not implemented"
        )

    vectors: list[Optional[np.ndarray]] = [None] * len(texts)
    todo: list[int] = []

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(texts):
            p = CACHE_DIR / f"{_cache_key(t)}.npy"
            if p.exists():
                try:
                    vectors[i] = np.load(p)
                    continue
                except Exception:
                    pass
            todo.append(i)
    else:
        todo = list(range(len(texts)))

    if todo:
        model = _model()
        batch_size = int(cfg("embedding.batch_size", 32))
        # Encode in chunks and flush the cache after each one. A single
        # model.encode() over 20k abstracts takes tens of minutes on CPU, and if
        # it is interrupted every vector is lost — which would make the 90-day
        # backfill effectively un-resumable.
        chunk = max(batch_size * 8, 256)
        for start in range(0, len(todo), chunk):
            window = todo[start : start + chunk]
            computed = model.encode(
                [texts[i] for i in window],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for i, vec in zip(window, computed):
                vectors[i] = vec.astype(np.float32)
                if use_cache:
                    np.save(CACHE_DIR / f"{_cache_key(texts[i])}.npy", vectors[i])
            if show_progress:
                done = min(start + chunk, len(todo))
                print(f"  embedded {done}/{len(todo)}", flush=True)

    return np.vstack([v for v in vectors if v is not None]) if vectors else np.zeros((0, 0))

