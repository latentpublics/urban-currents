"""R5-2: turn the unmatched extraction queue into vocabulary candidates.

The diagnosis first (R5-1): across the five prepared days the LLM proposed 670
distinct (item, facet, term) overlay tags that the controlled vocabulary had no
entry for — about 5.4 rejected per item against 1.2 accepted. The extraction is
not the bottleneck. The vocabulary is: `method:` has 6 families, and a
method-method projection can never have more nodes than the vocabulary has
methods, however many papers accumulate.

What this does:

- **Alias merges**, applied. Only where a term is the same thing as an existing
  entry under a different surface — case, plural, hyphenation, or a version
  suffix on a named tool. `yolov5` is YOLO. Every merge is printed.
- **Candidates**, listed and never adopted. Everything else goes into the
  `candidates:` block with its occurrence count, for curation. Promotion is
  YJUN's; a vocabulary that grows itself stops being controlled.

One-paper coinages are filtered out by an occurrence floor. "probe-driven
dynamic data catalog" is a phrase from a single abstract, not a term this field
shares, and a vocabulary full of those is noise wearing a schema.

Usage:
    uv run python scripts/vocab_candidates.py --check
    uv run python scripts/vocab_candidates.py --min-items 2
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

VOCAB = ROOT / "vocab"
FACET_FILES = {"methods": "methods.yaml", "data": "data.yaml", "tools": "tools.yaml"}
# Version suffixes on a named tool: yolov5, yolo11, gpt-4. The base name is the
# entry; the version is not a separate thing to track.
_VERSION_SUFFIX = re.compile(r"^(?P<base>[a-z][a-z\- ]*?)[ \-]?v?\d[\w.\-]*$")


def normalise(term: str) -> str:
    t = term.strip().lower().replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    return t[:-1] if t.endswith("s") and not t.endswith("ss") else t


def load_unmatched() -> dict[str, Counter]:
    """Deduped by (work_key, facet, term): a re-run must not inflate a term."""
    seen: set[tuple] = set()
    per_facet: dict[str, Counter] = defaultdict(Counter)
    for path in glob.glob(str(ROOT / "runs" / "run_*" / "unmatched.jsonl")):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            facet = row.get("facet")
            if facet not in FACET_FILES:
                continue
            term = (row.get("candidate") or "").strip().lower()
            if not term:
                continue
            key = (row.get("work_key"), facet, term)
            if key in seen:
                continue
            seen.add(key)
            per_facet[facet][term] += 1
    return per_facet


def existing_surfaces(doc: dict) -> dict[str, str]:
    """Every label and alias in a vocabulary file, normalised, to its entry id."""
    out: dict[str, str] = {}

    def walk(entries):
        for entry in entries or []:
            eid = entry.get("id")
            if eid:
                for surface in [entry.get("label")] + list(entry.get("aliases") or []):
                    if surface:
                        out[normalise(str(surface))] = eid
            for key in ("methods", "datasets", "tools", "children"):
                walk(entry.get(key))

    for key in ("families", "items", "entries", "tools", "datasets"):
        walk(doc.get(key))
    return out


def find_alias(term: str, surfaces: dict[str, str]) -> str | None:
    """An existing entry this term is only a surface variant of."""
    n = normalise(term)
    if n in surfaces:
        return surfaces[n]
    m = _VERSION_SUFFIX.match(n)
    if m:
        base = m.group("base").strip()
        if base in surfaces:
            return surfaces[base]
    return None


def add_alias(doc: dict, entry_id: str, alias: str) -> bool:
    def walk(entries) -> bool:
        for entry in entries or []:
            if entry.get("id") == entry_id:
                aliases = entry.setdefault("aliases", [])
                if alias not in aliases:
                    aliases.append(alias)
                    return True
                return False
            for key in ("methods", "datasets", "tools", "children"):
                if walk(entry.get(key)):
                    return True
        return False

    return any(walk(doc.get(k)) for k in ("families", "items", "entries", "tools", "datasets"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-items", type=int, default=2,
                    help="Distinct items that must propose a term for it to be a candidate")
    ap.add_argument("--check", action="store_true", help="report without writing")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    per_facet = load_unmatched()
    merged: list[tuple[str, str, str]] = []
    proposed: dict[str, list[tuple[str, int]]] = {}

    for facet, counter in per_facet.items():
        path = VOCAB / FACET_FILES[facet]
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        surfaces = existing_surfaces(doc)

        candidates: list[tuple[str, int]] = []
        for term, n in counter.most_common():
            entry_id = find_alias(term, surfaces)
            if entry_id:
                if add_alias(doc, entry_id, term):
                    merged.append((facet, term, entry_id))
                continue
            if n >= a.min_items:
                candidates.append((term, n))
        proposed[facet] = candidates

        if a.check:
            continue

        doc["candidates"] = [
            {
                "label": term,
                "suggested_id": f"{facet.rstrip('s') if facet != 'data' else 'data'}:"
                                + re.sub(r"[^a-z0-9]+", "-", normalise(term)).strip("-"),
                "occurrences": n,
                "source": "extraction queue (phase 0e R5)",
            }
            for term, n in candidates
        ]
        text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100)
        header = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("#")
        )
        path.write_text(
            header
            + "\n# REVIEW: `candidates:` below came from the extraction queue in phase 0e.\n"
            + "# REVIEW: nothing here is adopted. To adopt one, move it into a family,\n"
            + "# REVIEW: give it a stable id, fold near-duplicates into `aliases`, and\n"
            + "# REVIEW: delete it from this block.\n\n"
            + text,
            encoding="utf-8",
            newline="\n",
        )

    print(f"alias merges ({len(merged)}):")
    for facet, term, entry_id in merged:
        print(f"  {facet:<8} {term:<40} -> {entry_id}")
    print()
    for facet, cands in sorted(proposed.items()):
        print(f"{facet}: {len(cands)} candidates at >= {a.min_items} items")
        for term, n in cands[:15]:
            print(f"    {n:>3}  {term}")
    if a.check:
        print("\n(--check: nothing written)")


if __name__ == "__main__":
    main()
