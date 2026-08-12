"""Structured judgements (PRD §3.2 ``signals``, formerly ``caveat_flags``).

These are cheap and get reused as badges, filters and graph properties, so they
are computed by rule wherever a rule is honest. The LLM only fills the two that
need reading comprehension (``geographic_scope``, and ``data_available`` when the
rule is unsure), and its answers are marked ``basis: "llm"`` so they can be
audited separately.
"""

from __future__ import annotations

import re
from typing import Optional

from .models import Item, Signal

_CODE_URL = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org|codeberg\.org)/"
    r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+",
    re.I,
)
_CODE_PHRASE = re.compile(
    r"\b(code|implementation|source code)\s+(is\s+)?(publicly\s+)?(available|released)\b", re.I
)
_NO_RELEASE = re.compile(
    r"\bneither\s+the\s+data\s+nor\s+the\s+code\b|\bnot\s+publicly\s+(available|released)\b", re.I
)
_DATA_PHRASE = re.compile(
    r"\b(data|dataset|datasets)\s+(is|are)?\s*(publicly\s+)?(available|released|open)\b", re.I
)
_DATA_RELEASE = re.compile(r"\bwe\s+(release|publish|make\s+available)\b[^.]{0,60}\bdata", re.I)

# "3.4M images", "43 metropolitan areas", "12 cities", "n = 1,204"
_SAMPLE = re.compile(
    r"\b(\d[\d,.]*\s*(?:k|m|b|million|billion|thousand)?\s*"
    r"(?:images|records|trips|observations|respondents|users|cities|"
    r"areas|households|points|samples|papers|tweets|posts|segments|"
    r"buildings|sensors|stations|trajectories))\b",
    re.I,
)
_N_EQUALS = re.compile(r"\bn\s*=\s*\d[\d,]*", re.I)

# A full range is the informative detail, so it is matched first. A single
# alternation would report "from 2019" for "...from 2019-2023", because regex
# scanning is left-to-right by position rather than by specificity.
_TEMPORAL_RANGE = re.compile(
    r"\b(?:19|20)\d{2}\s*(?:-|–|—|to|through|and)\s*(?:19|20)\d{2}\b", re.I
)
_TEMPORAL_LOOSE = re.compile(
    r"\b(?:from|between|since)\s+(?:19|20)\d{2}\b"
    r"|\b\d+\s+(?:years|months|weeks)\s+of\b",
    re.I,
)


def _text(item: Item) -> str:
    return f"{item.bibliography.title}\n{item.bibliography.abstract or ''}"


def code_signal(item: Item) -> Signal:
    text = _text(item)
    if _NO_RELEASE.search(text):
        return Signal(value=False, confidence="high", basis="rule")
    m = _CODE_URL.search(text)
    if m:
        return Signal(value=True, url=m.group(0).rstrip(".,);"), confidence="high", basis="rule")
    if _CODE_PHRASE.search(text):
        return Signal(value=True, confidence="medium", basis="rule")
    return Signal(value=False, confidence="medium", basis="rule")


def data_signal(item: Item) -> Signal:
    text = _text(item)
    if _NO_RELEASE.search(text):
        return Signal(value=False, confidence="high", basis="rule")
    if _DATA_RELEASE.search(text) or _DATA_PHRASE.search(text):
        return Signal(value=True, confidence="medium", basis="rule")
    return Signal(value=False, confidence="low", basis="rule")


def sample_size_signal(item: Item) -> Signal:
    text = _text(item)
    m = _SAMPLE.search(text) or _N_EQUALS.search(text)
    if m:
        return Signal(
            value=True, detail=m.group(0).strip(), confidence="high", basis="rule"
        )
    return Signal(value=False, confidence="medium", basis="rule")


def temporal_signal(item: Item) -> Signal:
    text = _text(item)
    m = _TEMPORAL_RANGE.search(text) or _TEMPORAL_LOOSE.search(text)
    if m:
        return Signal(value=True, detail=m.group(0).strip(), confidence="high", basis="rule")
    return Signal(value=False, confidence="medium", basis="rule")


def apply_rule_signals(item: Item) -> Item:
    """Fill every rule-based signal. Safe to re-run; LLM-filled fields are kept."""
    item.signals.code_available = code_signal(item)
    if item.signals.data_available is None or item.signals.data_available.basis == "rule":
        item.signals.data_available = data_signal(item)
    item.signals.sample_size_reported = sample_size_signal(item)
    item.signals.temporal_coverage_reported = temporal_signal(item)
    if item.signals.is_retracted is None:
        item.signals.is_retracted = Signal(value=False, confidence="high", basis="rule")
    return item


def apply_badges(item: Item) -> Item:
    badges: list[str] = []
    if item.signals.code_available and item.signals.code_available.value is True:
        badges.append("code")
    if item.signals.data_available and item.signals.data_available.value is True:
        badges.append("data")
    badges.append(
        "published" if item.publication_status.state == "published" else "preprint"
    )
    item.badges = badges  # type: ignore[assignment]
    return item


def geographic_scope_from_llm(value: Optional[str]) -> Optional[Signal]:
    allowed = {"single_city", "multi_city", "national", "global", "not_applicable"}
    if value in allowed:
        return Signal(value=value, confidence="medium", basis="llm")
    return None
