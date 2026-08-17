"""U1: the "our kind of paper" classifier — the pipeline, not the model (phase 0h).

The relevance classifier answers *is this urban research*. It cannot answer *is
this the kind of urban research we cover*, and the `q` labels are the evidence
that the second question is separate: 7 of 30 labels are `drop_not_our_kind`,
which the relevance classifier scored highly and was right to.

**This does not train.** With 30 labels, of which 7 are `q`, any model fitted
here would be a description of YJUN's first afternoon of labelling. What it does
is make the fit a one-command step once the labels exist, and measure — now,
cheaply — whether the features being collected separate the two classes at all.

Three things it is careful about:

1. **The two label files are read separately and stay separately counted.**
   Pooling them for *training* is legitimate; pooling them for a *rate* is not.
   A band-stratified probe over-samples low affinity by construction, so the
   class balance of a pooled training set is an artefact of the sampling design,
   and any probability this model emits is uncalibrated against the real
   population until it is corrected by the band weights. Reported, never assumed.

2. **`n` (not urban) is excluded**, per the directive. That is the relevance
   classifier's error to answer for. Training on it would teach this model to
   re-do the first stage's job on the first stage's mistakes.

3. **The abstract embedding does not enter as 768 free parameters.** With tens of
   labels that is not a classifier, it is a lookup table. It enters as one
   scalar — cosine to the keep centroid, computed leave-one-out so a row never
   contributes to the centroid it is scored against.

Usage:
    uv run python scripts/train_kind_classifier.py --json runs/kind_classifier.json
    uv run python scripts/train_kind_classifier.py --train    # refuses if short
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline import store  # noqa: E402
from pipeline.graph.citation import load_reference_base  # noqa: E402
from pipeline.labeling import (  # noqa: E402
    PROBE_FACETS,
    RANKED_FACETS,
    load_labels,
    venue_prior_map,
)

# Positive and negative classes. Everything else is either not this model's
# question (`n`) or not a judgement (`skip`).
POSITIVE = "keep"
NEGATIVE = "drop_not_our_kind"
EXCLUDED = {"drop_not_urban": "relevance classifier's question, not this one"}

# Scalar features only. Each one has to be justifiable on its own with a
# two-figure label count; nothing here is a learned representation.
SCALAR_FEATURES = (
    "cites_canon",
    "canon_affinity_foundation",
    "canon_affinity_with_instruments",
    "canon_affinity_linear",
    "canon_affinity_raw",
    "canon_hits_sqrt",
    "refs_total",
    "venue_prior",
    "overlay_tags",
    "abstract_centroid_cos",
)

# Events per variable. The conventional floor for logistic regression is 10 per
# parameter on the *minority* class; below it the coefficients are fitted to
# noise and the sign of a feature can flip between resamples.
EVENTS_PER_VARIABLE = 10


def rows_by_facet() -> dict[str, list[dict]]:
    """Every label file, read on its own terms and kept apart.

    Deliberately not a single list. The moment these are concatenated the
    sampling design is lost, and the sampling design is the only thing that says
    whether a class balance means anything.
    """
    out: dict[str, list[dict]] = {}
    for facet in sorted(RANKED_FACETS | PROBE_FACETS):
        rows = load_labels(facet)
        if rows:
            out[facet] = rows
    return out


_STAGE_CACHE: dict[str, dict] = {}


def load_labelled_item(row: dict):
    """Find a labelled item, whether or not it was published.

    `content/items/` holds the 24 that made the issue; the labelling sample is 30
    drawn from a wider pool, so a labelled item is often not there. Reading only
    the store silently dropped 5 of 26 usable labels — including 2 of 7 `q`,
    which is a third of the minority class this classifier exists to learn. The
    day's stage output is where the rest live.
    """
    from datetime import date as _date

    from pipeline.labeling import _load_probe_pool
    from pipeline.metrics import Run
    from pipeline.stages import read_stage

    item = store.load_item(row["work_key"])
    if item is not None:
        return item

    day = row.get("date") or ""
    if day not in _STAGE_CACHE:
        found: dict = {}
        if day:
            run = Run.for_date(_date.fromisoformat(day))
            # Later stages first: each one carries more of the item than the
            # last, and the summary is a feature here.
            for stage in ("summarize", "labeling_pool", "classify"):
                for it in read_stage(run, stage) or []:
                    found.setdefault(it.work_key, it)
        for it in _load_probe_pool():
            found.setdefault(it.work_key, it)
        _STAGE_CACHE[day] = found
    return _STAGE_CACHE[day].get(row["work_key"])


def affinity_features() -> dict[str, dict[str, float]]:
    """Canon affinity under every normalisation, for each work in the base."""
    from journal_metrics import canon_affinity, canon_sets, cites_canon  # type: ignore

    foundation, instrument = canon_sets()
    both = {**instrument, **foundation}
    out: dict[str, dict[str, float]] = {}
    for record in load_reference_base():
        refs = record.get("referenced_works") or []
        hits = sum(1 for r in refs if r in foundation)
        out[record["work_key"]] = {
            # The binary first: the probe found the zero line carries the
            # `not_our_kind` signal (40% against 10%) and the grades above zero
            # do not. A float here so it can enter a linear model unchanged.
            "cites_canon": 1.0 if cites_canon(refs, foundation) else 0.0,
            "canon_affinity_foundation": canon_affinity(refs, foundation),
            "canon_affinity_with_instruments": canon_affinity(refs, both),
            "canon_affinity_linear": canon_affinity(refs, foundation, "linear"),
            "canon_affinity_raw": canon_affinity(refs, foundation, "none"),
            "canon_hits_sqrt": round(hits / (len(refs) ** 0.5), 6) if refs else 0.0,
            "refs_total": float(len(refs)),
        }
    return out


def build_design(embed_texts: bool = True) -> tuple[list[dict], dict]:
    """One row per usable label, with its provenance and every scalar feature.

    Rows whose item is no longer in the store are dropped and counted; a design
    matrix quietly shorter than the label file is how a training set stops
    matching the labels it claims to be built from.
    """
    affinity = affinity_features()
    priors = venue_prior_map()
    by_facet = rows_by_facet()

    design: list[dict] = []
    dropped = {"item_missing": 0, "excluded_class": 0, "unlabelled_class": 0}
    for facet, rows in by_facet.items():
        for row in rows:
            label = row.get("label")
            if label in EXCLUDED:
                dropped["excluded_class"] += 1
                continue
            if label not in (POSITIVE, NEGATIVE):
                dropped["unlabelled_class"] += 1
                continue
            item = load_labelled_item(row)
            if item is None:
                dropped["item_missing"] += 1
                continue
            aff = affinity.get(row["work_key"], {})
            source_id = item.bibliography.primary_location.source_id
            design.append({
                "work_key": row["work_key"],
                "facet": facet,
                "sampling": row.get("sampling", "ranked_top_n"),
                "band": row.get("band"),
                "source": row.get("source", "unknown"),
                "y": 1 if label == POSITIVE else 0,
                "text": " ".join(
                    filter(None, [item.bibliography.title, item.bibliography.abstract])
                ),
                **{k: aff.get(k) for k in (
                    "cites_canon",
                    "canon_affinity_foundation",
                    "canon_affinity_with_instruments",
                    "canon_affinity_linear",
                    "canon_affinity_raw",
                    "canon_hits_sqrt",
                    "refs_total",
                )},
                "venue_prior": priors.get(source_id),
                "overlay_tags": float(
                    len(item.entities.methods)
                    + len(item.entities.data)
                    + len(item.entities.tools)
                ),
            })

    if embed_texts and design:
        _add_centroid_cosine(design)

    counts = {
        "by_facet": {
            f: {
                "n": len(rows),
                "positive": sum(1 for r in rows if r.get("label") == POSITIVE),
                "negative": sum(1 for r in rows if r.get("label") == NEGATIVE),
                "sampling": rows[0].get("sampling", "ranked_top_n"),
            }
            for f, rows in by_facet.items()
        },
        "usable_rows": len(design),
        "positive": sum(r["y"] for r in design),
        "negative": sum(1 for r in design if r["y"] == 0),
        "dropped": dropped,
        "dropped_note": EXCLUDED,
    }
    return design, counts


def _add_centroid_cosine(design: list[dict]) -> None:
    """Leave-one-out cosine to the keep centroid.

    In-sample it would be near-perfect and mean nothing: every positive helps
    build the centroid it is then scored against. Leaving the row out is the
    cheapest honest version, and at this label count the difference between the
    two is the whole result.
    """
    try:
        import numpy as np

        from pipeline.filters.embed import embed
    except Exception as exc:  # pragma: no cover - optional at report time
        for row in design:
            row["abstract_centroid_cos"] = None
            row["embed_error"] = str(exc)
        return

    vectors = embed([r["text"] for r in design])
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    positives = np.array([r["y"] == 1 for r in design])

    for i, row in enumerate(design):
        mask = positives.copy()
        mask[i] = False
        if not mask.any():
            row["abstract_centroid_cos"] = None
            continue
        centroid = unit[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        row["abstract_centroid_cos"] = round(float(unit[i] @ centroid), 6)


def separation(design: list[dict]) -> dict:
    """Per-feature separation, which is what a feature contributes before a fit.

    Reported as the two class medians and the rank-biserial correlation, not as
    a coefficient. A coefficient here would be read as importance, and with this
    many labels it is mostly a statement about the regulariser.
    """
    out: dict[str, dict] = {}
    for feature in SCALAR_FEATURES:
        pos = [r[feature] for r in design if r["y"] == 1 and r.get(feature) is not None]
        neg = [r[feature] for r in design if r["y"] == 0 and r.get(feature) is not None]
        if not pos or not neg:
            out[feature] = {
                "note": "one class empty — no separation can be computed",
                "n_positive": len(pos),
                "n_negative": len(neg),
            }
            continue
        # AUC as the proportion of (positive, negative) pairs ordered correctly.
        wins = sum(
            1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg
        )
        auc = wins / (len(pos) * len(neg))
        out[feature] = {
            "n_positive": len(pos),
            "n_negative": len(neg),
            "median_keep": round(statistics.median(pos), 4),
            "median_not_our_kind": round(statistics.median(neg), 4),
            "auc": round(auc, 4),
            # 0.5 is a coin. Direction is what a refutation needs, and it is the
            # one thing 30 labels can supply.
            "direction": "as expected" if auc > 0.5 else ("inverted" if auc < 0.5 else "flat"),
        }
    return out


def minimum_labels(n_features: int = len(SCALAR_FEATURES)) -> dict:
    """How many `q` labels a fit would need before its coefficients mean anything.

    Two floors, because they answer different questions. The events-per-variable
    floor says when the coefficients stop being noise. The interval floor says
    when the *reported* number stops being useless: a keep rate estimated from
    15 negatives carries a 95% interval about 25 points wide, which spans every
    decision it could inform.
    """
    epv_full = EVENTS_PER_VARIABLE * n_features
    epv_core = EVENTS_PER_VARIABLE * 3
    return {
        "rule": f"{EVENTS_PER_VARIABLE} events per variable on the minority class",
        "all_scalar_features": {
            "n_features": n_features,
            "minimum_q": epv_full,
            "comment": (
                "every scalar feature in, including the four affinity "
                "normalisations that are near-collinear with each other"
            ),
        },
        "reduced_feature_set": {
            "n_features": 3,
            "minimum_q": epv_core,
            "comment": (
                "one affinity normalisation, venue_prior, centroid cosine — the "
                "realistic version, since the affinity variants are one signal "
                "measured four ways"
            ),
        },
        "half_width_note": (
            "a band keep rate from 10 labels has a 95% interval roughly ±30 "
            "points; from 30, roughly ±18; from 100, roughly ±10"
        ),
        "recommendation": (
            f"{epv_core} `q` labels before fitting anything, and treat the fit "
            f"as provisional until {epv_full}. At 7 today, the four prepared "
            f"days plus the probe are the path there — but only if the drop "
            f"rate holds, and it may not: the probe deliberately samples where "
            f"drops are likelier."
        ),
    }


def judgement(design: list[dict], sep: dict) -> dict:
    """Does the journal path need this classifier at all?

    The directive asks the question and it is the right one. If citing the canon
    is what "our kind" means, the journal path already has its answer as a
    number and a model would be a re-description of it.
    """
    journal = [r for r in design if r["source"] == "journal"]
    j_pos = sum(r["y"] for r in journal)
    j_neg = len(journal) - j_pos
    aff = sep.get("canon_affinity_foundation", {})
    scored = [r for r in design if r.get("canon_affinity_foundation") is not None]
    return {
        "journal_rows": len(journal),
        "journal_positive": j_pos,
        "journal_negative": j_neg,
        # The affinity figure's real population, which is not the label count:
        # arXiv rows have no reference list, so they are absent from it entirely.
        "affinity_scored_rows": len(scored),
        "affinity_scored_positive": aff.get("n_positive"),
        "affinity_scored_negative": aff.get("n_negative"),
        "affinity_auc": aff.get("auc"),
        "verdict": (
            "undecided — the probe has not been labelled yet"
            if (aff.get("n_negative") or 0) < 3
            else "measurable"
        ),
        "why_the_auc_is_not_evidence_yet": (
            "The affinity AUC is computed over the rows that have an affinity at "
            "all, and only two of the seven `q` labels do — the other five are "
            "arXiv, which has no reference list. So the number rests on two "
            "negatives. It is consistent with the signal working and would not "
            "look different if it did not."
        ),
        "all_affinity_normalisations_tie": (
            "foundation, with_instruments, linear, raw and hits_sqrt all report "
            "the same AUC, because both scored negatives sit at exactly zero and "
            "every normalisation maps zero to zero. This comparison cannot "
            "choose between them and will not until a negative has a non-zero "
            "affinity."
        ),
        "reasoning": (
            "If `canon_affinity` separates keep from not-our-kind on the journal "
            "path, the journal path does not need a kind classifier: the "
            "threshold is the model, and the labels move from training data to "
            "verification. That halves what YJUN has to label, permanently. The "
            "probe is what decides it — the ranked sample cannot, because it "
            "only ever showed the top of the ranking, where affinity is high by "
            "construction and every item in a band looks alike."
        ),
        "what_would_refute_it": (
            "keep rate in the zero band close to the high band. Then affinity is "
            "measuring how well-cited a subfield is rather than whether we cover "
            "it, and the classifier is needed after all."
        ),
    }


def fit(design: list[dict], features: list[str]) -> dict:  # pragma: no cover - gated
    """Fit, only when the label count allows it. Not reached in phase 0h."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    usable = [r for r in design if all(r.get(f) is not None for f in features)]
    X = np.array([[float(r[f]) for f in features] for r in usable])
    y = np.array([r["y"] for r in usable])
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit((X - mu) / sigma, y)
    return {
        "n": len(usable),
        "features": features,
        "standardised_coefficients": {
            f: round(float(c), 4) for f, c in zip(features, model.coef_[0])
        },
        "calibration_warning": (
            "the training rows come from two sampling designs; predicted "
            "probabilities are not calibrated to the daily population"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--train", action="store_true", help="Fit if the labels allow it")
    ap.add_argument("--no-embed", action="store_true", help="Skip the embedding feature")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    design, counts = build_design(embed_texts=not a.no_embed)
    sep = separation(design)
    minimum = minimum_labels()
    out = {
        "trained": False,
        "counts": counts,
        "separation": sep,
        "minimum_labels": minimum,
        "judgement": judgement(design, sep),
        "population": (
            "labelled items with a `keep` or `drop_not_our_kind` judgement, "
            "from both label files, counted separately by file above"
        ),
    }

    negatives = counts["negative"]
    floor = minimum["reduced_feature_set"]["minimum_q"]
    if a.train:
        if negatives < floor:
            out["refused"] = (
                f"{negatives} negative labels, floor is {floor}; a fit here would "
                f"describe the labeller's afternoon, not the field"
            )
        else:
            out["trained"] = True
            out["fit"] = fit(
                design,
                ["canon_affinity_foundation", "venue_prior", "abstract_centroid_cos"],
            )

    if a.json:
        Path(a.json).write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

    print(f"usable rows {counts['usable_rows']}  "
          f"(positive {counts['positive']}, negative {counts['negative']})")
    for facet, c in counts["by_facet"].items():
        print(f"   {facet:<16} n={c['n']:<4} keep={c['positive']:<4} q={c['negative']:<4} "
              f"[{c['sampling']}]")
    # Every feature has its own population: a row missing that feature is not in
    # its comparison. `canon_affinity` is defined only for items in the
    # reference base, so its AUC is computed over a fraction of the labels and
    # the counts belong next to it, not in a footnote.
    print(f"\n   {'feature':<34} {'n(k/q)':>9} {'keep':>9} {'q':>9} {'auc':>7}")
    for feature, s in sep.items():
        if "auc" not in s:
            print(f"   {feature:<34} {s['n_positive']:>4}/{s['n_negative']:<4} {s['note']}")
            continue
        n = f"{s['n_positive']}/{s['n_negative']}"
        print(f"   {feature:<34} {n:>9} {s['median_keep']:>9} "
              f"{s['median_not_our_kind']:>9} {s['auc']:>7}  {s['direction']}")
    print(f"\nminimum q: {minimum['reduced_feature_set']['minimum_q']} to fit, "
          f"{minimum['all_scalar_features']['minimum_q']} to trust — have {negatives}")
    if out.get("refused"):
        print(f"refused: {out['refused']}")


if __name__ == "__main__":
    main()
