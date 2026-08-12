"""Bootstrap methods.yaml / data.yaml candidates from training-set abstracts (PRD §3.6).

Mines n-grams from the positive training abstracts, drops anything the vocabulary
already covers, and appends the survivors as **candidate entries marked
`review: true`**. Nothing here is curated — the point is to turn "invent a
controlled vocabulary from nothing" into "say yes or no to a ranked list".

Candidates are written to a separate `candidates:` block rather than merged into
the curated entries, so a bad suggestion can never silently become a tag: the
matcher only reads the curated blocks.

Usage:
    uv run python scripts/bootstrap_vocab.py [--top 60]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.linking.vocab_match import Vocabulary, _norm  # noqa: E402
from pipeline.paths import RUNS, VOCAB  # noqa: E402

TRAINSET = RUNS / "trainset" / "trainset.jsonl"

# Cue phrases that reliably precede a method or a data source in an abstract.
METHOD_CUES = re.compile(
    r"\b(?:using|based on|via|apply|applied|employ(?:ing)?|propose[sd]?|"
    r"we use|trained? (?:a|an|the)|framework|approach|algorithm|model)\b",
    re.I,
)
DATA_CUES = re.compile(
    r"\b(?:data(?:set)?s? (?:from|of)|collected from|drawn from|records? (?:of|from)|"
    r"imagery|survey|sensors?|traces?|we use|based on)\b",
    re.I,
)

METHOD_TAIL = re.compile(
    r"\b([a-z][a-z\-]+(?:\s+[a-z][a-z\-]+){0,3}\s+"
    r"(?:model|models|network|networks|regression|algorithm|method|methods|"
    r"framework|analysis|learning|clustering|simulation|estimation|inference))\b",
    re.I,
)
DATA_TAIL = re.compile(
    r"\b([a-z][a-z\-]+(?:\s+[a-z][a-z\-]+){0,3}\s+"
    r"(?:data|dataset|datasets|imagery|images|records|surveys|traces|"
    r"trajectories|counts|measurements|statistics))\b",
    re.I,
)

STOPWORDS = {
    "the", "this", "our", "we", "their", "these", "those", "such", "other",
    "same", "new", "first", "second", "both", "each", "more", "most", "large",
    "small", "different", "various", "several", "proposed", "existing",
}

# The mined phrases are dominated by "<praise adjective> + <generic head noun>":
# novel framework, comprehensive analysis, unified approach. Those are rhetoric,
# not method names, and a vocabulary built from them would tag every paper
# identically. A candidate has to carry at least one token that is neither.
GENERIC_MODIFIERS = {
    "novel", "comprehensive", "conceptual", "theoretical", "unified", "general",
    "traditional", "standard", "simple", "efficient", "effective", "robust",
    "hybrid", "integrated", "improved", "advanced", "systematic", "detailed",
    "empirical", "quantitative", "qualitative", "comparative", "extensive",
    "to", "using", "and", "of", "for", "with", "in", "on", "a", "an",
}
GENERIC_HEADS = {
    "model", "models", "method", "methods", "framework", "analysis", "approach",
    "algorithm", "data", "dataset", "datasets", "learning", "estimation",
    "network", "networks", "inference", "records", "images", "statistics",
    "measurements", "surveys", "counts", "imagery", "traces", "trajectories",
}

MIN_COUNT = 8


def load_positive_texts() -> list[str]:
    if not TRAINSET.exists():
        raise SystemExit(f"no training set at {TRAINSET}; run build_trainset.py first")
    out = []
    for line in TRAINSET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("label") == 1 and row.get("abstract"):
            out.append(row["abstract"])
    return out


def mine(texts: list[str], tail: re.Pattern, cues: re.Pattern) -> Counter:
    counter: Counter[str] = Counter()
    for text in texts:
        if not cues.search(text):
            continue
        for m in tail.finditer(text):
            phrase = _norm(m.group(1))
            words = phrase.split()
            if len(words) < 2 or words[0] in STOPWORDS:
                continue
            if any(ch.isdigit() for ch in phrase):
                continue
            if not any(
                w not in GENERIC_MODIFIERS and w not in GENERIC_HEADS for w in words
            ):
                continue
            counter[phrase] += 1
    return counter


def already_covered(phrase: str, vocab: Vocabulary) -> bool:
    entry, score = vocab.match(phrase)
    return entry is not None and score >= 0.9


def append_candidates(path: Path, facet: str, candidates: list[tuple[str, int]]) -> int:
    """Append a `candidates:` block. Idempotent: an existing block is replaced."""
    text = path.read_text(encoding="utf-8")
    marker = "\n# ---- BOOTSTRAPPED CANDIDATES"
    if marker in text:
        text = text[: text.index(marker)]

    slug_seen: set[str] = set()
    lines = [
        "",
        "# ---- BOOTSTRAPPED CANDIDATES ----------------------------------------",
        "#",
        "# REVIEW: generated by scripts/bootstrap_vocab.py from training-set abstracts.",
        "# REVIEW: NOT curated and NOT used for matching — the matcher reads only the",
        "# REVIEW: curated blocks above. To adopt one: move it up, give it a stable id,",
        f"# REVIEW: fold near-duplicates into `aliases`, and delete it from here.",
        f"# REVIEW: {len(candidates)} candidates, ranked by frequency in positives.",
        "#",
        "candidates:",
    ]
    for phrase, n in candidates:
        slug = re.sub(r"[^a-z0-9]+", "-", phrase).strip("-")[:40]
        if slug in slug_seen:
            continue
        slug_seen.add(slug)
        lines.append(f"  - label: \"{phrase}\"")
        lines.append(f"    suggested_id: \"{facet}:{slug}\"")
        lines.append(f"    occurrences: {n}")
    path.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(slug_seen)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=60)
    args = p.parse_args()

    texts = load_positive_texts()
    print(f"mining {len(texts)} positive abstracts")

    for facet, tail, cues, fname in (
        ("method", METHOD_TAIL, METHOD_CUES, "methods.yaml"),
        ("data", DATA_TAIL, DATA_CUES, "data.yaml"),
    ):
        vocab = Vocabulary.load("methods" if facet == "method" else "data")
        counts = mine(texts, tail, cues)
        candidates = [
            (phrase, n)
            for phrase, n in counts.most_common()
            if n >= MIN_COUNT and not already_covered(phrase, vocab)
        ][: args.top]
        n = append_candidates(VOCAB / fname, facet, candidates)
        print(f"  {fname}: {n} candidates appended (from {len(counts)} mined phrases)")


if __name__ == "__main__":
    main()
