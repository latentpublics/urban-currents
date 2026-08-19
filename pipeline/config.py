"""Configuration and secrets.

Secrets come from ``.env`` only. Note the OpenAlex variable name is
``OPENALEX_KEY`` — not ``OPENALEX_API_KEY``.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from . import paths

_ENV_LOADED = False


def load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(paths.ROOT / ".env")
    except ImportError:  # dotenv missing: fall back to a minimal parser
        env = paths.ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    _ENV_LOADED = True


def secret(name: str) -> Optional[str]:
    """Read a secret. Never log or print the return value."""
    load_env()
    v = os.environ.get(name)
    return v.strip() if v else None


def openalex_key() -> Optional[str]:
    return secret("OPENALEX_KEY")


def anthropic_key() -> Optional[str]:
    return secret("ANTHROPIC_API_KEY")


def github_token() -> Optional[str]:
    """Optional. Raises GitHub's search limit from 60/hour to 5,000.

    Absent is a supported state, not a degraded one: the backlink probe runs
    unauthenticated and simply does less per hour. Nothing in this repo creates
    a token.
    """
    return secret("GITHUB_TOKEN")


def google_key() -> Optional[str]:
    return secret("GOOGLE_API_KEY")


def contact_email() -> str:
    return secret("CONTACT_EMAIL") or "urban-currents@example.com"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@functools.lru_cache(maxsize=None)
def pipeline_config() -> dict[str, Any]:
    return _load_yaml(paths.CONFIG / "pipeline.yaml")


@functools.lru_cache(maxsize=None)
def scoring_config() -> dict[str, Any]:
    return _load_yaml(paths.CONFIG / "scoring.yaml")


@functools.lru_cache(maxsize=None)
def arxiv_vocab() -> dict[str, Any]:
    return _load_yaml(paths.VOCAB / "sources" / "arxiv.yaml")


_JOURNALS_CACHE: dict[str, Any] = {}


def journals_vocab() -> dict[str, Any]:
    """Whitelist journals, cached against the file's mtime.

    Deliberately not an lru_cache: `build_journal_whitelist.py` rewrites this
    file mid-session and the new list has to take effect. But it also cannot be
    re-parsed per call — `_is_whitelist_journal` asks for it once per item, and
    a 90-day backfill asks ~25,000 times, which turned a 160-entry YAML parse
    into the dominant cost of the stage.
    """
    # `UC_JOURNALS_FILE` points this at a different list without touching
    # `journals.yaml`. Added in phase 0g so a rebuilt whitelist could be trained
    # and evaluated against — comparing a candidate list means running the whole
    # trainset build on it, and the only alternative was overwriting the file
    # the comparison is against.
    override = os.environ.get("UC_JOURNALS_FILE")
    path = Path(override) if override else paths.VOCAB / "sources" / "journals.yaml"
    stamp = path.stat().st_mtime_ns if path.exists() else 0
    if _JOURNALS_CACHE.get("stamp") != stamp or _JOURNALS_CACHE.get("path") != str(path):
        _JOURNALS_CACHE.clear()
        _JOURNALS_CACHE.update(
            {"stamp": stamp, "path": str(path), "doc": _load_yaml(path)}
        )
    return _JOURNALS_CACHE["doc"]


def vocab_file(name: str) -> dict[str, Any]:
    return _load_yaml(paths.VOCAB / name)


def reset_caches() -> None:
    _JOURNALS_CACHE.clear()
    pipeline_config.cache_clear()
    scoring_config.cache_clear()
    arxiv_vocab.cache_clear()


def cfg(path: str, default: Any = None) -> Any:
    """Dotted lookup into pipeline.yaml, e.g. ``cfg("arxiv.page_size", 200)``."""
    node: Any = pipeline_config()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
