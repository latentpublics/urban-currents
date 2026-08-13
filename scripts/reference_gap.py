"""R6: how much of the reference list is a book we will never see (phase 0e).

OpenAlex keeps the references it can resolve to a Work, which is a median 0.75
of what publishers deposit to Crossref (measured in phase 0d). What falls out is
whatever has no DOI — and in planning literature that is disproportionately
books and reports. Jacobs, Lefebvre, Harvey, government reports: the field's
canon is substantially monographs, so **our canon is a canon of papers, not a
canon of the field.**

That cannot be fixed from here. It can be sized, and it can be named: the
unresolved reference strings come back as text, and the classics are in them.

Rule-based classification only, no LLM. The signals are the ones a bibliography
already carries — a publisher-and-place pattern, page ranges written `pp.`,
edition markers, the absence of a journal-style volume/issue.

Usage:
    uv run python scripts/reference_gap.py --sample 100 --json runs/reference_gap.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from pipeline import store  # noqa: E402
from pipeline.collectors.abstracts import CROSSREF_API  # noqa: E402
from pipeline.config import contact_email  # noqa: E402

# A book leaves fingerprints a journal article does not.
_BOOK_SIGNALS = (
    (re.compile(r"\bpp?\.\s*\d", re.I), "page range written pp."),
    (re.compile(r"\b(\d+(st|nd|rd|th)\s+ed(ition)?\.?)\b", re.I), "edition marker"),
    (re.compile(r"\b(press|publisher|publishing|verlag|routledge|springer-verlag|"
                r"wiley|blackwell|sage publications|university press|books)\b", re.I),
     "publisher name"),
    (re.compile(r"\b(eds?\.|editors?)\b", re.I), "editor marker"),
    (re.compile(r"\bin:\s", re.I), "chapter 'In:' marker"),
    (re.compile(r"\b(report|working paper|technical report|discussion paper|thesis|"
                r"dissertation)\b", re.I), "report or thesis"),
)
# A journal reference usually carries a volume and issue or a journal name.
_JOURNAL_SIGNAL = re.compile(r"\b\d+\s*\(\s*\d+\s*\)|\bvol\.?\s*\d+|\bno\.?\s*\d+", re.I)


def classify(text: str) -> tuple[str, list[str]]:
    hits = [label for pattern, label in _BOOK_SIGNALS if pattern.search(text)]
    if hits and not _JOURNAL_SIGNAL.search(text):
        return "book_or_report", hits
    if hits:
        return "ambiguous", hits
    return "unclassified", []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    candidates = [
        it for it in store.iter_items()
        if it.ids.doi and not it.ids.arxiv and it.graph.referenced_works
    ]
    picked = random.Random(42).sample(candidates, min(a.sample, len(candidates)))

    client = httpx.Client(
        headers={"User-Agent": (
            f"UrbanCurrents/0.1 (+https://github.com/youngjour/urban-currents; "
            f"mailto:{contact_email()})"
        )},
        timeout=30.0,
        follow_redirects=True,
    )

    total_refs = with_doi = without_doi = 0
    kinds: Counter = Counter()
    signals: Counter = Counter()
    unresolved: Counter = Counter()
    compared = 0

    for item in picked:
        try:
            r = client.get(f"{CROSSREF_API}{item.ids.doi}", params={"mailto": contact_email()})
        except Exception:  # noqa: BLE001
            continue
        if r.status_code != 200:
            continue
        refs = (r.json().get("message") or {}).get("reference") or []
        if not refs:
            continue
        compared += 1
        for ref in refs:
            total_refs += 1
            if ref.get("DOI"):
                with_doi += 1
                continue
            without_doi += 1
            text = " ".join(
                str(ref.get(k)) for k in
                ("unstructured", "article-title", "volume-title", "author", "year",
                 "journal-title", "series-title", "edition")
                if ref.get(k)
            ).strip()
            if not text:
                kinds["no_text_at_all"] += 1
                continue
            kind, hits = classify(text)
            kinds[kind] += 1
            for h in hits:
                signals[h] += 1
            unresolved[text[:160]] += 1

    result = {
        "population": "content/items with a DOI and a non-empty referenced_works",
        "items_sampled": len(picked),
        "items_with_crossref_references": compared,
        "references_seen": total_refs,
        "with_doi": with_doi,
        "without_doi": without_doi,
        "share_without_doi": round(without_doi / total_refs, 4) if total_refs else None,
        "kinds_among_without_doi": dict(kinds.most_common()),
        "share_book_or_report_of_all": (
            round(kinds["book_or_report"] / total_refs, 4) if total_refs else None
        ),
        "signals": dict(signals.most_common()),
        "top_unresolved": [
            {"text": t, "seen": n} for t, n in unresolved.most_common(30)
        ],
        "method": (
            "Rule-based, no LLM. A reference with a book signal and no "
            "journal-style volume/issue is counted as a book or report; one with "
            "both is ambiguous and counted separately."
        ),
    }

    if a.json:
        Path(a.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
    print(json.dumps({k: v for k, v in result.items() if k != "top_unresolved"}, indent=2))
    print("\ntop unresolved reference strings:")
    for row in result["top_unresolved"][:30]:
        print(f"  {row['seen']:>3}x  {row['text'][:110]}")


if __name__ == "__main__":
    main()
