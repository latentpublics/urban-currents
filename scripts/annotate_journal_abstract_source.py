"""Add `abstract_source` to every entry in vocab/sources/journals.yaml.

An in-place annotation rather than a regeneration. `build_journal_whitelist.py`
would rewrite the file from OpenAlex and reset every `include:` flag to its
heuristic default — and those flags carry YJUN's hand review, which a rebuild
would silently throw away. So this edits the lines it owns and touches nothing
else, and the generator emits the same field for future builds.

The field is a **route**, not a verdict: a journal we cannot get abstracts for
stays on the whitelist, and its articles publish in `Also published today`.
Reviewing the whitelist now asks two independent questions — is this urban
research, and can we read it.

Usage:
    uv run python scripts/annotate_journal_abstract_source.py [--check]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.collectors.abstracts import abstract_source_for_publisher  # noqa: E402

TARGET = ROOT / "vocab" / "sources" / "journals.yaml"

_PUBLISHER = re.compile(r'^\s*publisher:\s*(.*)$')
_EXISTING = re.compile(r'^\s*abstract_source:\s*')


def annotate(path: Path, check: bool = False) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    tally: Counter[str] = Counter()

    for line in lines:
        if _EXISTING.match(line):
            continue  # rewritten below from the current mapping
        out.append(line)
        m = _PUBLISHER.match(line)
        if not m:
            continue
        raw = m.group(1).strip().strip('"').strip("'")
        publisher = None if raw in ("", "null", "~") else raw
        source = abstract_source_for_publisher(publisher)
        tally[source] += 1
        out.append(f'    abstract_source: "{source}"')

    text = "\n".join(out) + "\n"
    if not check:
        path.write_text(text, encoding="utf-8", newline="\n")
    return dict(tally)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report without writing")
    a = ap.parse_args()
    tally = annotate(TARGET, check=a.check)
    total = sum(tally.values())
    print(f"{'would annotate' if a.check else 'annotated'} {total} journals in {TARGET.name}")
    for source, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {source:>12}: {n:>3}  ({n / total:.0%})" if total else f"  {source}: {n}")


if __name__ == "__main__":
    main()
