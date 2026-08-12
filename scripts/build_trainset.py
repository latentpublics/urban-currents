"""Build the relevance-classifier training set (PRD §5.4).

Positives (~4,000): works from the whitelist journals, **but only 70% of them.**
The other 30% are arXiv papers carrying an urban OpenAlex topic. That split is
not a rounding detail — journal abstracts are written in planning and social
science prose, arXiv abstracts in ML prose. Train on journals alone and the
model scores down exactly the arXiv urban-computing papers this product exists
to surface.

Negatives (~4,000): arXiv cs.LG / cs.CV / cs.AI from the same period, with the
positives removed and anything carrying an urban subfield topic excluded.

Output: ``runs/trainset/trainset.jsonl`` — one row per example with the text the
classifier will see, its label, and its source, so per-source recall can be
measured later (PRD §5.4, "measured failure mode").

Usage:
    uv run python scripts/build_trainset.py [--positives 4000] [--negatives 4000]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.collectors.base import ARXIV_SOURCE_ID, invert_abstract  # noqa: E402
from pipeline.collectors.openalex import configure_pyalex  # noqa: E402
from pipeline.config import cfg, journals_vocab  # noqa: E402
from pipeline.filters.embed import embed_text  # noqa: E402
from pipeline.metrics import OpenAlexBudget  # noqa: E402
from pipeline.paths import RUNS  # noqa: E402

OUT_DIR = RUNS / "trainset"
MIN_ABSTRACT_CHARS = 200
ARXIV_POSITIVE_SHARE = 0.30


def _budget() -> OpenAlexBudget:
    return OpenAlexBudget(
        daily_usd=float(cfg("openalex.daily_budget_usd", 1.0)),
        stop_fraction=float(cfg("openalex.budget_stop_fraction", 0.8)),
    )


def _row(work: dict, label: int, source: str) -> dict | None:
    title = work.get("display_name") or ""
    abstract = invert_abstract(work.get("abstract_inverted_index"))
    if not title or not abstract or len(abstract) < MIN_ABSTRACT_CHARS:
        return None
    return {
        "id": (work.get("id") or "").rsplit("/", 1)[-1],
        "label": label,
        "source": source,
        "title": title,
        "abstract": abstract,
        "text": embed_text(title, abstract),
        "primary_topic": ((work.get("primary_topic") or {}).get("display_name")),
        "subfield": (
            ((work.get("primary_topic") or {}).get("subfield") or {}).get("id") or ""
        ).rsplit("/", 1)[-1],
    }


def collect(pyalex, query, budget, want: int, label: int, source: str, per_page: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for page in query.paginate(per_page=per_page, n_max=None):
        budget.charge(float((getattr(page, "meta", {}) or {}).get("cost_usd") or 0.0))
        for w in page:
            r = _row(w, label, source)
            if r and r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
        if len(rows) >= want:
            break
    return rows[:want]


def build(n_pos: int, n_neg: int, since: str, until: str, seed: int) -> Path:
    pyalex = configure_pyalex()
    budget = _budget()
    rng = random.Random(seed)

    doc = journals_vocab()
    source_ids = [s["id"] for s in (doc.get("sources") or []) if s.get("include", True)]
    if not source_ids:
        raise SystemExit("vocab/sources/journals.yaml has no included sources")

    per_page = int(cfg("openalex.per_page", 100))
    n_journal = int(n_pos * (1 - ARXIV_POSITIVE_SHARE))
    n_arxiv_pos = n_pos - n_journal

    print(f"positives: {n_journal} from {len(source_ids)} journals, {n_arxiv_pos} from arXiv")

    # -- positives, journals --------------------------------------------
    journal_rows: list[dict] = []
    chunks = [source_ids[i : i + 40] for i in range(0, len(source_ids), 40)]
    want_each = max(1, n_journal // len(chunks) + 1)
    for chunk in chunks:
        q = (
            pyalex.Works()
            .filter(
                **{
                    "primary_location.source.id": "|".join(chunk),
                    "from_publication_date": since,
                    "to_publication_date": until,
                    "type": "article",
                    "language": "en",
                    "has_abstract": True,
                }
            )
            .sort(cited_by_count="desc")
        )
        journal_rows.extend(collect(pyalex, q, budget, want_each, 1, "journal", per_page))
        if len(journal_rows) >= n_journal:
            break
    rng.shuffle(journal_rows)
    journal_rows = journal_rows[:n_journal]

    # -- positives, arXiv urban -----------------------------------------
    subfields = [str(s) for s in (cfg("openalex.whitelist_subfields", ["3322"]) or [])]
    q = (
        pyalex.Works()
        .filter(
            **{
                "primary_location.source.id": ARXIV_SOURCE_ID,
                "primary_topic.subfield.id": "|".join(subfields),
                "from_publication_date": since,
                "to_publication_date": until,
                "language": "en",
                "has_abstract": True,
            }
        )
        .sort(publication_date="desc")
    )
    arxiv_pos = collect(pyalex, q, budget, n_arxiv_pos, 1, "arxiv_urban", per_page)

    # -- negatives -------------------------------------------------------
    # arXiv works outside the urban subfields. `filter_not` on the subfield keeps
    # the obvious positives out; the id overlap check below catches the rest.
    q = (
        pyalex.Works()
        .filter(
            **{
                "primary_location.source.id": ARXIV_SOURCE_ID,
                "from_publication_date": since,
                "to_publication_date": until,
                "language": "en",
                "has_abstract": True,
            }
        )
        .filter_not(**{"primary_topic.subfield.id": "|".join(subfields)})
        .sort(publication_date="desc")
    )
    negatives = collect(pyalex, q, budget, int(n_neg * 1.3), 0, "arxiv_other", per_page)

    positive_ids = {r["id"] for r in journal_rows} | {r["id"] for r in arxiv_pos}
    negatives = [r for r in negatives if r["id"] not in positive_ids]
    rng.shuffle(negatives)
    negatives = negatives[:n_neg]

    rows = journal_rows + arxiv_pos + negatives
    rng.shuffle(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "trainset.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    meta = {
        "since": since,
        "until": until,
        "seed": seed,
        "counts": {
            "journal_positive": len(journal_rows),
            "arxiv_positive": len(arxiv_pos),
            "negative": len(negatives),
            "total": len(rows),
        },
        "arxiv_positive_share": ARXIV_POSITIVE_SHARE,
        "journals_used": len(source_ids),
        "openalex_cost_usd": round(budget.spent, 6),
        "openalex_calls": budget.calls,
    }
    (OUT_DIR / "trainset.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(meta, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--positives", type=int, default=4000)
    p.add_argument("--negatives", type=int, default=4000)
    p.add_argument("--since", default="2024-01-01")
    p.add_argument("--until", default="2026-06-30")
    p.add_argument("--seed", type=int, default=int(cfg("classifier.random_state", 42)))
    a = p.parse_args()
    build(a.positives, a.negatives, a.since, a.until, a.seed)


if __name__ == "__main__":
    main()
