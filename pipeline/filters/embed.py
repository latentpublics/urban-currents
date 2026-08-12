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
from pathlib import Path
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
        batch = [texts[i] for i in todo]
        computed = model.encode(
            batch,
            batch_size=int(cfg("embedding.batch_size", 32)),
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        for i, vec in zip(todo, computed):
            vectors[i] = vec.astype(np.float32)
            if use_cache:
                np.save(CACHE_DIR / f"{_cache_key(texts[i])}.npy", vectors[i])

    return np.vstack([v for v in vectors if v is not None]) if vectors else np.zeros((0, 0))


def embedding_available() -> bool:
    try:
        _model()
        return True
    except Exception:
        return False


def cache_path() -> Path:
    return CACHE_DIR
