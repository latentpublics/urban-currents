"""Shared collector helpers: identifier normalisation and text cleanup.

``work_key`` priority is arXiv ID → DOI → OpenAlex ID (PRD §5.2), and once
assigned it never changes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

ARXIV_SOURCE_ID = "S4306400194"

# Matches 2608.01234, 2608.01234v3, arXiv:2608.01234, and the pre-2007 form
# cs.CY/0701001.
_ARXIV_NEW = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(v\d+)?", re.I)
_ARXIV_OLD = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$", re.I)


def normalize_arxiv_id(value: Optional[str]) -> Optional[str]:
    """Extract a bare arXiv ID (version stripped) from an ID, URL, or DOI."""
    if not value:
        return None
    v = value.strip()
    m = _ARXIV_DOI.search(v)
    if m:
        v = m.group(1)
    m = _ARXIV_NEW.search(v)
    if m:
        return m.group(1)
    m = _ARXIV_OLD.search(v)
    if m:
        return m.group(1).lower()
    return None


def normalize_doi(value: Optional[str]) -> Optional[str]:
    """Lower-cased bare DOI, with any URL prefix stripped."""
    if not value:
        return None
    v = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if v.startswith(prefix):
            v = v[len(prefix) :]
    return v or None


def normalize_openalex_id(value: Optional[str]) -> Optional[str]:
    """``https://openalex.org/W123`` → ``W123``."""
    if not value:
        return None
    return value.strip().rsplit("/", 1)[-1] or None


def arxiv_doi(arxiv_id: str) -> str:
    return f"10.48550/arxiv.{arxiv_id}"


def work_key_from(
    arxiv_id: Optional[str] = None,
    doi: Optional[str] = None,
    openalex_id: Optional[str] = None,
) -> Optional[str]:
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if doi:
        return f"doi:{normalize_doi(doi)}"
    if openalex_id:
        return f"openalex:{normalize_openalex_id(openalex_id)}"
    return None


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return _WS.sub(" ", value.replace("\n", " ")).strip() or None


def normalize_title(title: str) -> str:
    """Lower-case, punctuation-stripped, whitespace-collapsed (PRD §5.2 rule 3)."""
    t = unicodedata.normalize("NFKD", title).lower()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def last_name(author_name: str) -> str:
    """Best-effort surname. Handles 'Jane Q. Doe' and 'Doe, Jane'."""
    name = author_name.strip()
    if "," in name:
        return normalize_title(name.split(",", 1)[0])
    parts = name.split()
    return normalize_title(parts[-1]) if parts else ""


def invert_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """Rebuild plain text from OpenAlex's ``abstract_inverted_index``."""
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    if not positions:
        return None
    positions.sort()
    return clean_text(" ".join(w for _, w in positions))
