"""Match free-text overlay candidates to controlled vocabulary.

This module is the gate that keeps ``entities`` free of free strings (PRD §9).
An LLM candidate becomes an ``EntityRef`` only if it matches a vocabulary entry
by exact label, alias, or a high fuzzy ratio. Everything else is written to
``unmatched.jsonl`` — that file is the queue for growing the vocabulary, and it
is more informative than a tag nobody can filter on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from rapidfuzz import process
from rapidfuzz.fuzz import ratio

from ..config import vocab_file
from ..models import EntityRef

FUZZY_THRESHOLD = 88


@dataclass(frozen=True)
class VocabEntry:
    id: str
    label: str
    parent: Optional[str] = None
    facet: str = "methods"


class Vocabulary:
    """Flattened lookup over one facet's YAML file."""

    def __init__(self, facet: str, entries: dict[str, VocabEntry]):
        self.facet = facet
        self._by_surface = entries  # normalised surface form → entry
        self._surfaces = list(entries.keys())

    @classmethod
    def load(cls, facet: str) -> "Vocabulary":
        entries: dict[str, VocabEntry] = {}

        def add(entry: VocabEntry, surfaces: Iterable[str]) -> None:
            for s in surfaces:
                key = _norm(s)
                if key and key not in entries:
                    entries[key] = entry

        if facet == "methods":
            doc = vocab_file("methods.yaml")
            for family in doc.get("families", []) or []:
                fam = VocabEntry(family["id"], family["label"], None, facet)
                add(fam, [family["label"], *(family.get("aliases") or [])])
                for m in family.get("methods", []) or []:
                    entry = VocabEntry(m["id"], m["label"], family["id"], facet)
                    add(entry, [m["label"], *(m.get("aliases") or [])])
        else:
            fname = {"data": "data.yaml", "tools": "tools.yaml"}[facet]
            doc = vocab_file(fname)
            for it in doc.get("items", []) or []:
                entry = VocabEntry(it["id"], it["label"], it.get("parent"), facet)
                add(entry, [it["label"], *(it.get("aliases") or [])])

        return cls(facet, entries)

    def match(self, candidate: str) -> tuple[Optional[VocabEntry], float]:
        key = _norm(candidate)
        if not key:
            return None, 0.0
        if key in self._by_surface:
            return self._by_surface[key], 1.0
        if not self._surfaces:
            return None, 0.0
        hit = process.extractOne(key, self._surfaces, scorer=ratio, score_cutoff=FUZZY_THRESHOLD)
        if hit is None:
            return None, 0.0
        surface, score, _ = hit
        return self._by_surface[surface], round(score / 100.0, 4)


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("-", " ").replace("_", " ").split())


@dataclass
class MatchResult:
    refs: list[EntityRef]
    unmatched: list[str]


def match_facet(candidates: Iterable[str], facet: str, vocab: Optional[Vocabulary] = None) -> MatchResult:
    v = vocab or Vocabulary.load(facet)
    refs: list[EntityRef] = []
    unmatched: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        entry, score = v.match(cand)
        if entry is None:
            unmatched.append(cand)
            continue
        if entry.id in seen:
            continue
        seen.add(entry.id)
        refs.append(EntityRef(id=entry.id, label=entry.label, confidence=round(score, 4)))
    return MatchResult(refs=refs, unmatched=unmatched)
