"""Build relevance-classifier training sets and a shared evaluation set.

The core idea is unchanged (PRD §5.4): do not hand-define what counts as urban
research, take the field's own settled answer. What changed in Phase 0b is
*which* answer we take for the arXiv side, and how the result is judged.

**The task the classifier actually has.** After the entry paths split (N4), a
whitelist-journal article enters on membership and never touches the classifier.
So the only job left is **arXiv-urban vs arXiv-other**, and that is the only
thing worth measuring. A shared arXiv-only evaluation set is built once here and
excluded from every training set, so the variants are compared on identical
ground.

**Positive definitions.**

- ``journal`` — articles from whitelist journals. Same as Phase 0.
- ``arxiv_strict`` — **published in a whitelist journal AND carrying an arXiv
  location.** The cleanest definition available: it does not depend on OpenAlex
  subfield classification, and "a journal in this field accepted it" is not
  circular. The catch is volume — only ~224 exist since 2020, so it cannot fill
  a training set alone.
- ``arxiv_subfield`` — arXiv-primary works whose ``primary_topic.subfield.id``
  is in a given set. Phase 0 used 3322|3313|**3305**, and 3305 (Geography,
  Planning and Development) contributed 409 of 1,102 — the suspected source of
  the soft-social-science false positives. v2 narrows it to 3322|3313.

**Variants.**

| variant | journal positives | arXiv positives | negatives |
|---|---|---|---|
| v1 | yes | subfield 3322/3313/3305 | arXiv non-urban |
| v2 | yes | strict + subfield 3322/3313 | arXiv non-urban |
| v3 | **no** | strict + subfield 3322/3313 | arXiv non-urban |

v3 exists to answer a question v2 alone cannot: do journal positives help the
arXiv task, or does their prose style hurt it? Embeddings are local, so asking
costs nothing but time.

Usage:
    uv run python scripts/build_trainset.py --variant v2
    uv run python scripts/build_trainset.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.collectors.base import ARXIV_SOURCE_ID, invert_abstract  # noqa: E402
from pipeline.collectors.openalex import configure_pyalex  # noqa: E402
from pipeline.config import cfg, journals_vocab  # noqa: E402
from pipeline.filters.embed import embed_text  # noqa: E402
from pipeline.metrics import OpenAlexBudget  # noqa: E402
from pipeline.paths import RUNS  # noqa: E402

OUT_DIR = RUNS / "trainset"
EVAL_PATH = OUT_DIR / "eval_arxiv.jsonl"
MIN_ABSTRACT_CHARS = 200

NARROW_SUBFIELDS = "3322|3313"
WIDE_SUBFIELDS = "3322|3313|3305"

VARIANTS = {
    "v1": {
        "journal_positives": 2800,
        "arxiv_subfields": WIDE_SUBFIELDS,
        "use_strict": False,
        "note": "Phase 0 baseline: subfield 3322/3313/3305, journal positives included",
    },
    "v2": {
        "journal_positives": 2800,
        "arxiv_subfields": NARROW_SUBFIELDS,
        "use_strict": True,
        "note": "strict (journal-accepted arXiv) + narrowed subfields, journal positives included",
    },
    "v3": {
        "journal_positives": 0,
        "arxiv_subfields": NARROW_SUBFIELDS,
        "use_strict": True,
        "note": "arXiv-only task: no journal positives at all",
    },
}


def _budget() -> OpenAlexBudget:
    return OpenAlexBudget(
        daily_usd=float(cfg("openalex.daily_budget_usd", 1.0)),
        stop_fraction=float(cfg("openalex.budget_stop_fraction", 0.8)),
    )


def _row(work: dict, label: int, source: str) -> Optional[dict]:
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
        # Which journal a positive came from. Not a feature — the classifier
        # never sees it — but without it there is no way to ask whether a
        # training set is 160 journals or five journals wearing 160 names, which
        # is the whole question when the whitelist widens (U4).
        "venue_id": (
            ((work.get("primary_location") or {}).get("source") or {}).get("id") or ""
        ).rsplit("/", 1)[-1],
        "primary_topic": ((work.get("primary_topic") or {}).get("display_name")),
        "subfield": (
            ((work.get("primary_topic") or {}).get("subfield") or {}).get("id") or ""
        ).rsplit("/", 1)[-1],
    }


def collect(query, budget, want: int, label: int, source: str, per_page: int) -> list[dict]:
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


def whitelist_ids() -> list[str]:
    ids = [s["id"] for s in (journals_vocab().get("sources") or []) if s.get("include", True)]
    if not ids:
        raise SystemExit("vocab/sources/journals.yaml has no included sources")
    return ids


def _chunks(seq: list[str], n: int = 40) -> list[list[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


# --------------------------------------------------------------------------
# Positive / negative pools
# --------------------------------------------------------------------------


def fetch_arxiv_strict(pyalex, budget, since: str, until: str, per_page: int) -> list[dict]:
    """Works published in a whitelist journal that also have an arXiv location.

    The date floor is deliberately earlier than the other pools: this definition
    is scarce, and it does not decay with age the way a topical query does.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for chunk in _chunks(whitelist_ids()):
        q = pyalex.Works().filter(
            **{
                "primary_location.source.id": "|".join(chunk),
                "locations.source.id": ARXIV_SOURCE_ID,
                "from_publication_date": since,
                "to_publication_date": until,
                "type": "article",
                "language": "en",
                "has_abstract": True,
            }
        )
        for r in collect(q, budget, 10_000, 1, "arxiv_strict", per_page):
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    return rows


def fetch_arxiv_subfield(
    pyalex, budget, subfields: str, want: int, since: str, until: str, per_page: int
) -> list[dict]:
    q = (
        pyalex.Works()
        .filter(
            **{
                "primary_location.source.id": ARXIV_SOURCE_ID,
                "primary_topic.subfield.id": subfields,
                "from_publication_date": since,
                "to_publication_date": until,
                "language": "en",
                "has_abstract": True,
            }
        )
        .sort(publication_date="desc")
    )
    return collect(q, budget, want, 1, "arxiv_subfield", per_page)


def fetch_arxiv_negatives(
    pyalex, budget, want: int, since: str, until: str, per_page: int
) -> list[dict]:
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
        .filter_not(**{"primary_topic.subfield.id": WIDE_SUBFIELDS})
        .sort(publication_date="desc")
    )
    return collect(q, budget, want, 0, "arxiv_other", per_page)


def fetch_journal_positives_per_journal(
    pyalex,
    budget,
    since: str,
    until: str,
    per_page: int,
    rng: random.Random,
    floor: int = 5,
    ceiling: int = 40,
) -> list[dict]:
    """One query per journal, with a floor and a ceiling on what each contributes.

    The chunked sampler asks 40 journals at once sorted by citation count, so a
    chunk's quota is filled by whichever of its journals publishes most and is
    cited hardest — and a small journal in a chunk with *Cities* in it can
    contribute nothing at all. Widening the whitelist makes that worse, not
    better: the same quota spreads over more chunks while the same few journals
    keep answering it.

    The cost is one request per journal instead of one per chunk, which at 161
    journals is real but small. Whether it is worth paying is what this measures.
    """
    rows: list[dict] = []
    per_journal: dict[str, int] = {}
    for sid in whitelist_ids():
        q = (
            pyalex.Works()
            .filter(
                **{
                    "primary_location.source.id": sid,
                    "from_publication_date": since,
                    "to_publication_date": until,
                    "type": "article",
                    "language": "en",
                    "has_abstract": True,
                }
            )
            .sort(cited_by_count="desc")
        )
        got = collect(q, budget, ceiling, 1, "journal", per_page)
        per_journal[sid] = len(got)
        # A journal that cannot meet the floor contributes what it has. Dropping
        # it would reintroduce exactly the bias this is meant to remove.
        rows.extend(got)
    rng.shuffle(rows)
    below_floor = sum(1 for n in per_journal.values() if n < floor)
    print(
        json.dumps({
            "per_journal_sampling": {
                "journals": len(per_journal),
                "floor": floor,
                "ceiling": ceiling,
                "below_floor": below_floor,
                "at_ceiling": sum(1 for n in per_journal.values() if n >= ceiling),
                "rows": len(rows),
            }
        })
    )
    return rows


def fetch_journal_positives(
    pyalex, budget, want: int, since: str, until: str, per_page: int, rng: random.Random
) -> list[dict]:
    rows: list[dict] = []
    chunks = _chunks(whitelist_ids())
    want_each = max(1, want // len(chunks) + 1)
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
        rows.extend(collect(q, budget, want_each, 1, "journal", per_page))
        if len(rows) >= want:
            break
    rng.shuffle(rows)
    return rows[:want]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build_eval_set(
    pyalex, budget, n_pos: int, n_neg: int, since: str, until: str, per_page: int,
    seed: int,
) -> dict[str, Any]:
    """The shared arXiv-only holdout every variant is judged on.

    Positives use the strict definition because it is the most defensible ground
    truth available: a journal in the field accepted the paper. Built once and
    excluded from every training set, so variant scores are comparable.
    """
    rng = random.Random(seed)
    strict = fetch_arxiv_strict(pyalex, budget, since, until, per_page)
    rng.shuffle(strict)
    positives = strict[:n_pos]

    negatives = fetch_arxiv_negatives(pyalex, budget, n_neg * 2, "2024-01-01", until, per_page)
    rng.shuffle(negatives)
    negatives = negatives[:n_neg]

    rows = positives + negatives
    rng.shuffle(rows)
    write_jsonl(EVAL_PATH, rows)

    meta = {
        "positives": len(positives),
        "negatives": len(negatives),
        "strict_pool_available": len(strict),
        "positive_definition": "published in a whitelist journal AND has an arXiv location",
        "since": since,
        "until": until,
        "seed": seed,
    }
    (OUT_DIR / "eval_arxiv.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(meta, indent=2))
    return meta


def build_variant(
    variant: str,
    n_arxiv_pos: int,
    n_neg: int,
    since: str,
    until: str,
    seed: int,
    journal_cap: Optional[int] = None,
    per_journal: bool = False,
    per_journal_floor: int = 5,
    per_journal_ceiling: int = 40,
    out_name: Optional[str] = None,
) -> dict[str, Any]:
    spec = dict(VARIANTS[variant])
    if journal_cap is not None:
        spec["journal_positives"] = journal_cap
    pyalex = configure_pyalex()
    budget = _budget()
    rng = random.Random(seed)
    per_page = int(cfg("openalex.per_page", 100))

    eval_ids = {r["id"] for r in read_jsonl(EVAL_PATH)}
    if not eval_ids:
        raise SystemExit("build the shared eval set first: --eval-only")

    arxiv_rows: list[dict] = []
    if spec["use_strict"]:
        arxiv_rows += fetch_arxiv_strict(pyalex, budget, "2020-01-01", until, per_page)
    strict_kept = len([r for r in arxiv_rows if r["id"] not in eval_ids])

    need = max(0, n_arxiv_pos - strict_kept)
    arxiv_rows += fetch_arxiv_subfield(
        pyalex, budget, spec["arxiv_subfields"], need, since, until, per_page
    )

    journal_rows: list[dict] = []
    if spec["journal_positives"]:
        if per_journal:
            journal_rows = fetch_journal_positives_per_journal(
                pyalex, budget, since, until, per_page, rng,
                floor=per_journal_floor, ceiling=per_journal_ceiling,
            )[: int(spec["journal_positives"])]
        else:
            journal_rows = fetch_journal_positives(
                pyalex, budget, int(spec["journal_positives"]), since, until, per_page, rng
            )

    negatives = fetch_arxiv_negatives(pyalex, budget, int(n_neg * 1.4), since, until, per_page)

    # Nothing in the shared eval set may appear in training, or the comparison
    # measures memorisation rather than generalisation.
    seen: set[str] = set()
    rows: list[dict] = []
    for r in arxiv_rows + journal_rows + negatives:
        if r["id"] in eval_ids or r["id"] in seen:
            continue
        seen.add(r["id"])
        rows.append(r)

    positives = [r for r in rows if r["label"] == 1]
    negs = [r for r in rows if r["label"] == 0][:n_neg]
    rows = positives + negs
    rng.shuffle(rows)

    out_dir = OUT_DIR / (out_name or variant)
    write_jsonl(out_dir / "trainset.jsonl", rows)

    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    arxiv_pos = sum(v for k, v in by_source.items() if k.startswith("arxiv_") and k != "arxiv_other")
    meta = {
        "variant": out_name or variant,
        "base_variant": variant,
        "journal_positive_cap": spec["journal_positives"],
        "journal_sampling": (
            f"per_journal floor={per_journal_floor} ceiling={per_journal_ceiling}"
            if per_journal
            else "chunked (40 journals per query, cited_by_count desc)"
        ),
        "note": spec["note"],
        "arxiv_subfields": spec["arxiv_subfields"],
        "uses_strict_definition": spec["use_strict"],
        "counts": {"total": len(rows), **by_source},
        "arxiv_positive_total": arxiv_pos,
        "strict_share_of_arxiv_positives": (
            round(by_source.get("arxiv_strict", 0) / arxiv_pos, 3) if arxiv_pos else 0.0
        ),
        "eval_ids_excluded": len(eval_ids),
        "since": since,
        "until": until,
        "seed": seed,
        "openalex_cost_usd": round(budget.spent, 6),
    }
    (out_dir / "trainset.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(VARIANTS), default="v2")
    p.add_argument("--eval-only", action="store_true", help="Build the shared eval set and stop")
    p.add_argument("--eval-positives", type=int, default=200)
    p.add_argument("--eval-negatives", type=int, default=600)
    p.add_argument("--arxiv-positives", type=int, default=1100)
    p.add_argument("--negatives", type=int, default=4000)
    p.add_argument("--since", default="2024-01-01")
    p.add_argument("--until", default="2026-06-30")
    p.add_argument("--seed", type=int, default=int(cfg("classifier.random_state", 42)))
    # U4 (phase 0h): measure what a wider whitelist needs from the sampler.
    # Nothing here changes a default — the variants are named and compared.
    p.add_argument("--journal-cap", type=int, help="Override the journal positive cap")
    p.add_argument("--per-journal", action="store_true", help="Sample per journal, not per chunk")
    p.add_argument("--per-journal-floor", type=int, default=5)
    p.add_argument("--per-journal-ceiling", type=int, default=40)
    p.add_argument("--out-name", help="Write under runs/trainset/<name> instead of the variant")
    a = p.parse_args()

    if a.eval_only:
        build_eval_set(
            configure_pyalex(), _budget(), a.eval_positives, a.eval_negatives,
            "2020-01-01", a.until, int(cfg("openalex.per_page", 100)), a.seed,
        )
        return
    build_variant(
        a.variant, a.arxiv_positives, a.negatives, a.since, a.until, a.seed,
        journal_cap=a.journal_cap,
        per_journal=a.per_journal,
        per_journal_floor=a.per_journal_floor,
        per_journal_ceiling=a.per_journal_ceiling,
        out_name=a.out_name,
    )


if __name__ == "__main__":
    main()
