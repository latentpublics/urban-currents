"""Volume gate (PRD §5.3).

Small, high-yield arXiv categories go straight to the classifier. The three
large, low-yield ones (cs.LG / cs.CV / cs.AI) must clear a deliberately generous
OR keyword match first. Anything not from arXiv (a journal item from OpenAlex)
is never gated — it is already inside the whitelist.

The gate's recall is measured once per Phase 0; see ``uc gate-recall``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..config import arxiv_vocab
from ..models import Item


def _compile(keywords: Iterable[str]) -> list[re.Pattern]:
    pats = []
    for kw in keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        if kw.endswith("*"):
            body = re.escape(kw[:-1])
            pats.append(re.compile(rf"\b{body}\w*", re.I))
        else:
            body = re.escape(kw).replace(r"\ ", r"[\s\-]")
            pats.append(re.compile(rf"\b{body}\b", re.I))
    return pats


@dataclass
class Gate:
    ungated: set[str] = field(default_factory=set)
    gated: set[str] = field(default_factory=set)
    patterns: list[re.Pattern] = field(default_factory=list)

    @classmethod
    def from_vocab(cls) -> "Gate":
        v = arxiv_vocab()
        cats = v.get("categories", {}) or {}
        return cls(
            ungated=set(cats.get("ungated", []) or []),
            gated=set(cats.get("gated", []) or []),
            patterns=_compile(v.get("keywords", []) or []),
        )

    def matched_keywords(self, item: Item) -> list[str]:
        text = f"{item.bibliography.title}\n{item.bibliography.abstract or ''}"
        hits = []
        for p in self.patterns:
            m = p.search(text)
            if m:
                hits.append(m.group(0).lower())
        return sorted(set(hits))

    def decide(self, item: Item) -> tuple[bool, str]:
        """Returns (passes, reason)."""
        cats = set(item.bibliography.categories or [])
        if not cats:
            # Journal items and anything without category metadata bypass the gate.
            return True, "no_categories"
        if cats & self.ungated:
            return True, "ungated_category"
        if cats & self.gated:
            hits = self.matched_keywords(item)
            if hits:
                return True, "keyword:" + ",".join(hits[:5])
            return False, "no_keyword_match"
        return True, "unknown_category"


def apply_gate(items: Iterable[Item], gate: Optional[Gate] = None) -> tuple[list[Item], list[tuple[Item, str]]]:
    g = gate or Gate.from_vocab()
    kept: list[Item] = []
    dropped: list[tuple[Item, str]] = []
    for it in items:
        ok, reason = g.decide(it)
        if ok:
            kept.append(it)
        else:
            dropped.append((it, reason))
    return kept, dropped
