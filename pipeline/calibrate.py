"""90-day backfill and quiet-day threshold calibration (PRD §5.6, Q2, Q3).

Two weeks of live operation gives 70-100 items. Picking a quantile from that is
statistically meaningless, so the threshold comes from a 90-day backfill instead
— thousands of scored items, from which we take the quantile that puts the
headline rate in the 30-50% band.

**The backfill does not summarise.** That is the single largest cost risk in
Phase 0 (PRD §10), and it is unnecessary: the headline score needs relevance,
badges, cluster size and overlay tags, all of which are available without an LLM
(the overlay comes from the rule-based vocabulary scan).

**The backfill must contain both entry paths.** A threshold calibrated on an
arXiv-only distribution and applied to a mixed population is calibrated against
the wrong thing: a whitelist-journal article carries relevance 1.0 by
membership, which alone contributes 0.40 to its headline score, while an arXiv
item carries a classifier probability that rarely reaches that. Measured with an
arXiv-only backfill: every one of five live days cleared the threshold (100%
against a 30-50% target). So journal collection is on by default here.

**The backfill must age like the archive.** Novelty is measured against the tags
already published, and in the live pipeline that set grows one day at a time. A
backfill that scores all 90 days against one frozen archive lets a tag that first
appeared in May still count as fresh in August, which pins novelty near 1.0 for a
few items every single day. Measured before this was fixed: 78 of 90 daily top
scores were *exactly* 0.6400, so the headline rate jumped from 92% to 6% across
one tie value and no threshold could land in the target band. The days are walked
in order and the archive grows behind them.

**And it must age on what would publish, not on what was scored.** Only 24 items
a day reach `content/`, chosen by the two-path slot rule, so those are the only
tags a live day can have seen before. Calibrating on the candidate pool measures
a population the product never shows.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from . import paths, store
from .config import cfg, scoring_config
from .metrics import Run
from .models import Item
from .score.headline import score_item


def backfill_dir() -> Path:
    """Resolved per call, not captured at import — otherwise a test with a
    temporary UC_ROOT would write into the real repository."""
    return paths.RUNS / "backfill"


def _scores_path() -> Path:
    return backfill_dir() / "scores.jsonl"


def _would_publish(day_items: list[Item], threshold: float) -> set[str]:
    """The work_keys this day would actually publish.

    The same `fill_slots` the live `select` stage uses, so the calibration
    population cannot drift away from what the pipeline actually publishes.
    """
    from .run_stages import fill_slots

    journal_taken, arxiv_taken = fill_slots(
        day_items,
        threshold,
        int(cfg("selection.slots.journal", 12)),
        int(cfg("selection.slots.arxiv", 12)),
    )
    return {it.work_key for it in journal_taken + arxiv_taken}


def _row(item: Item, threshold: float, published: bool) -> dict[str, Any]:
    from .run_stages import _is_whitelist_journal

    is_journal = _is_whitelist_journal(item)
    return {
        "work_key": item.work_key,
        "date": str(item.first_published or ""),
        "relevance": item.scores.relevance,
        "headline": item.scores.headline,
        "components": item.scores.components.model_dump(),
        "categories": item.bibliography.categories,
        "source": "journal" if is_journal else "arxiv",
        # `selected` is the candidate pool — everything that cleared its path's
        # entry test. `published` is the subset that would have filled the day's
        # 24 slots, and that is what the quiet-day threshold is calibrated on.
        "selected": True if is_journal else item.scores.relevance >= threshold,
        "published": published,
    }


def score_days(kept: list[Item], seen: set[str], threshold: float) -> list[dict[str, Any]]:
    """Score a whole backfill day by day, with the archive growing behind it.

    The same shape as running `uc score` on 90 consecutive days: each day sees
    the tags published on every earlier day and none of its own. Scoring the
    range against one frozen archive instead keeps novelty near 1.0 throughout
    and collapses the daily top scores onto a single value.

    ``seen`` is mutated, as the archive is.
    """
    by_day: dict[str, list[Item]] = {}
    for item in kept:
        by_day.setdefault(str(item.first_published or ""), []).append(item)
    # An undated item cannot belong to a day: score it against the final archive
    # and never add it, so it can neither age the archive nor carry a headline.
    undated = by_day.pop("", [])

    rows: list[dict[str, Any]] = []
    for day in sorted(by_day):
        day_items = by_day[day]
        for item in day_items:
            score_item(item, seen)
        published = _would_publish(day_items, threshold)
        rows.extend(_row(it, threshold, it.work_key in published) for it in day_items)
        for item in day_items:
            if item.work_key in published:
                for e in item.entities.methods + item.entities.data + item.entities.tools:
                    seen.add(e.id)

    for item in undated:
        score_item(item, seen)
    rows.extend(_row(it, threshold, False) for it in undated)
    return rows


def run_backfill(
    end: date,
    days: int = 90,
    sources: str = "all",
    max_pages: Optional[int] = None,
) -> dict[str, Any]:
    """Collect, gate, classify and score a date range without summarising."""
    from .filters.classifier import score_items
    from .filters.gate import Gate, apply_gate
    from .dedup.merge import merge_candidates
    from .linking.vocab_match import Vocabulary, scan_text
    from .signals import apply_badges, apply_rule_signals

    start = end - timedelta(days=days - 1)
    run = Run(f"backfill_{start}_{end}", end)
    from .run_stages import _is_whitelist_journal

    candidates: list[Item] = []
    if sources in ("all", "arxiv"):
        from .collectors.arxiv import ArxivCollector

        collector = ArxivCollector(run)
        candidates.extend(collector.collect(end, backfill_from=start, max_pages=max_pages))
    if sources in ("all", "openalex"):
        from .collectors.openalex import OpenAlexCollector

        try:
            oc = OpenAlexCollector(run)
            candidates.extend(oc.collect_journals(end, backfill_from=start, max_pages=200))
        except Exception as e:  # noqa: BLE001
            run.error(f"backfill openalex: {type(e).__name__}: {e}")

    run.count("arxiv_fetched", len(candidates))
    merged = merge_candidates(candidates, run_date=end).items
    kept, dropped = apply_gate(merged, Gate.from_vocab())
    run.count("after_dedup", len(merged))
    run.count("after_gate", len(kept))

    # Rejected items are kept on disk: the gate recall measurement (PRD §5.3)
    # samples from exactly this set.
    with (run.dir / "gate_rejected.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for it, reason in dropped:
            fh.write(
                json.dumps(
                    {
                        "work_key": it.work_key,
                        "reason": reason,
                        "title": it.bibliography.title,
                        "abstract": it.bibliography.abstract or "",
                        "categories": it.bibliography.categories,
                        "date": str(it.first_published or ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Same relevance rule as production (N4): whitelist membership is the entry
    # ticket and scores 1.0; only arXiv items go through the classifier.
    journal_items = [it for it in kept if _is_whitelist_journal(it)]
    arxiv_items = [it for it in kept if not _is_whitelist_journal(it)]
    for it in journal_items:
        it.scores.relevance = 1.0
        it.scores.components.relevance = 1.0
        it.provenance.classifier_version = "whitelist-membership"
    pred = score_items(arxiv_items)
    vocabs = {f: Vocabulary.load(f) for f in ("methods", "data", "tools")}

    # The archive as it stood *before* the range being replayed. The live days
    # already in content/ fall inside a 90-day window, and counting their tags
    # as previously seen would drive their own novelty to zero — the same
    # self-reference `stage_score` excludes for (PRD §9).
    kept_keys = {it.work_key for it in kept}
    seen: set[str] = {
        e.id
        for item in store.iter_items()
        if item.work_key not in kept_keys
        and not (item.first_published and start <= item.first_published <= end)
        for e in item.entities.methods + item.entities.data + item.entities.tools
    }

    threshold = float(cfg("classifier.threshold", 0.5))
    backfill_dir().mkdir(parents=True, exist_ok=True)

    for item in kept:
        apply_rule_signals(item)
        apply_badges(item)
        text = f"{item.bibliography.title} {item.bibliography.abstract or ''}"
        for facet in ("methods", "data", "tools"):
            refs = scan_text(text, facet, vocabs[facet])
            if refs:
                setattr(item.entities, facet, refs)

    rows = score_days(kept, seen, threshold)

    with _scores_path().open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    meta = {
        "start": str(start),
        "end": str(end),
        "days": days,
        "candidates": len(candidates),
        "after_dedup": len(merged),
        "after_gate": len(kept),
        "gate_rejected": len(dropped),
        "classifier_version": pred.version,
        "journal_items": len(journal_items),
        "arxiv_items": len(arxiv_items),
        "selection_threshold": threshold,
        "selected": sum(1 for r in rows if r["selected"]),
        "published": sum(1 for r in rows if r["published"]),
        "openalex_cost_usd": round(run.metrics.cost.openalex_usd, 6),
    }
    (backfill_dir() / "backfill.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    run.stage("backfill", "OK")
    run.save()
    return meta


def load_scores() -> list[dict]:
    p = _scores_path()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _spread(rows: list[dict], threshold: float, all_days: list[str]) -> dict[str, Any]:
    per_day = Counter(r["date"] for r in rows if r["relevance"] >= threshold)
    counts = sorted(per_day.get(d, 0) for d in all_days)
    return {
        "median_per_day": _quantile(counts, 0.5) if counts else 0,
        "p25_per_day": _quantile(counts, 0.25) if counts else 0,
        "p75_per_day": _quantile(counts, 0.75) if counts else 0,
        "min_per_day": counts[0] if counts else 0,
        "max_per_day": counts[-1] if counts else 0,
        "total_selected": sum(counts),
    }


def daily_distribution(rows: list[dict], threshold: float) -> dict[str, Any]:
    """Q2: per-day count of items that clear the selection threshold.

    Split by entry path as well as pooled. A whitelist journal article clears by
    membership, so the journal column is "how many whitelist articles appeared",
    which is a volume measurement but not a *signal* measurement — nothing was
    judged. Pooling the two moved this number from 28 to 72 when journals joined
    the backfill, and a reader comparing it with the earlier figure would read
    that as the field getting busier rather than the population changing.
    """
    all_days = sorted({r["date"] for r in rows if r["date"]})
    per_day = Counter(r["date"] for r in rows if r["relevance"] >= threshold)
    out: dict[str, Any] = {
        "days_observed": len(all_days),
        **_spread(rows, threshold, all_days),
        "per_day": {d: per_day.get(d, 0) for d in all_days},
        "by_source": {
            src: _spread([r for r in rows if r.get("source") == src], threshold, all_days)
            for src in sorted({r.get("source", "?") for r in rows})
        },
    }
    return out


CANDIDATE_FLOORS = (0.35, 0.50, 0.70, 0.90, 0.95)


def arxiv_candidates_by_floor(
    rows: Optional[list[dict]] = None, floors: tuple[float, ...] = CANDIDATE_FLOORS
) -> dict[str, Any]:
    """How many arXiv candidates a day clears each possible relevance floor (P5).

    The companion to the score-band precision table. Precision says how good the
    items above a floor are; this says whether there are enough of them. The
    arXiv path is asked for 12 slots a day, so a floor that yields a median of 3
    is not a precision decision, it is a decision to stop filling the path.

    Measurement only. The threshold is `classifier.threshold` and moving it is
    YJUN's call, on labels from more than one day.
    """
    rows = load_scores() if rows is None else rows
    if not rows:
        return {"status": "NO_DATA", "hint": "run `uc backfill --days 90` first"}

    by_day: dict[str, list[float]] = {}
    for r in rows:
        if r.get("source") == "arxiv" and r.get("date"):
            by_day.setdefault(r["date"], []).append(float(r["relevance"]))

    slots = int(cfg("selection.slots.arxiv", 12))
    out = []
    for floor in floors:
        counts = sorted(sum(1 for v in vs if v >= floor) for vs in by_day.values())
        out.append({
            "floor": floor,
            "median_per_day": _quantile(counts, 0.5),
            "p25_per_day": _quantile(counts, 0.25),
            "p75_per_day": _quantile(counts, 0.75),
            "min_per_day": counts[0] if counts else 0,
            "max_per_day": counts[-1] if counts else 0,
            "days_short_of_slots": sum(1 for c in counts if c < slots),
        })
    return {
        "status": "OK",
        "days": len(by_day),
        "arxiv_slots": slots,
        "configured_threshold": float(cfg("classifier.threshold", 0.35)),
        "floors": out,
    }


def _novelty_decay(published: list[dict]) -> dict[str, Any]:
    """Mean novelty per month across the replayed range.

    Recorded because it is the reason the headline score has so little to work
    with. The overlay vocabulary is a closed list, so once the archive has seen
    it the term goes to zero and stays there — measured over the 90 days, the
    monthly mean falls 0.12 -> 0.005 -> 0.002 -> 0.000. In steady state the
    headline score of a whitelist journal article is a constant 0.44, and the
    only thing that can lift a day above it is an arXiv item carrying code or
    data links. Whether a term that dies after two weeks belongs in the formula
    at all is PRD §5.6's question, not this function's.
    """
    by_month: dict[str, list[float]] = {}
    for r in published:
        month = str(r.get("date", ""))[:7]
        value = (r.get("components") or {}).get("novelty")
        if month and value is not None:
            by_month.setdefault(month, []).append(float(value))
    return {
        m: {"mean": round(sum(v) / len(v), 4), "n": len(v)}
        for m, v in sorted(by_month.items())
    }


WEIGHTS = ("relevance", "source_multiplicity", "artifact_completeness", "novelty")


def _component_audit(published: list[dict]) -> dict[str, Any]:
    """What each weighted component actually does on real data.

    A component that takes one value across nearly every item contributes
    nothing to ranking no matter what weight it carries. Measuring that per
    component is how anyone reading the report can see which parts of the
    headline formula met data and survived, rather than taking the weights in
    `config/scoring.yaml` at face value.
    """
    weights = scoring_config().get("weights") or {}
    out: dict[str, Any] = {}
    for name in WEIGHTS:
        values = [
            round(float((r.get("components") or {}).get(name, 0.0)), 4)
            for r in published
            if (r.get("components") or {}).get(name) is not None
        ]
        if not values:
            continue
        counts = Counter(values)
        top_value, top_n = counts.most_common(1)[0]

        # Split by entry path, because that is where the degeneracy lives: a
        # component can look varied across the whole population and still be a
        # single constant on one of the two paths.
        by_source: dict[str, Any] = {}
        for src in sorted({r.get("source", "?") for r in published}):
            vals = [
                round(float((r.get("components") or {}).get(name, 0.0)), 4)
                for r in published
                if r.get("source") == src
                and (r.get("components") or {}).get(name) is not None
            ]
            if not vals:
                continue
            c = Counter(vals)
            v, n = c.most_common(1)[0]
            by_source[src] = {
                "n": len(vals),
                "distinct_values": len(c),
                "modal_value": v,
                "modal_share": round(n / len(vals), 4),
            }

        out[name] = {
            "weight": weights.get(name),
            "n": len(values),
            "distinct_values": len(counts),
            "modal_value": top_value,
            "modal_share": round(top_n / len(values), 4),
            "by_source": by_source,
        }
    return out


def _provisional(day_tops: list[float], audit: dict[str, Any]) -> dict[str, Any]:
    """Whether the chosen threshold rests on a distribution that can carry one.

    Landing in the target band is not the same as the threshold meaning
    something. If most days share one top score, the threshold is splitting a
    tie and one more day above or below it moves the rate in steps of several
    points; if most components are constant, the score it splits is barely a
    score. Both are measured, so the flag clears itself when the formula is
    fixed rather than needing a human to remember to remove it.
    """
    reasons = []
    if day_tops:
        modal_share = Counter(day_tops).most_common(1)[0][1] / len(day_tops)
        if modal_share >= 0.5:
            reasons.append(
                f"{modal_share:.0%} of daily top scores share one value — the "
                f"threshold splits a tie, not a distribution"
            )
    # Per entry path, not pooled: `relevance` looks varied across the whole
    # population and is a single constant on the journal path, which is the
    # half of the product that degeneracy actually hits.
    paths_seen = {p for a in audit.values() for p in a.get("by_source", {})}
    for path in sorted(paths_seen):
        dead = sorted(
            name
            for name, a in audit.items()
            if (a.get("by_source", {}).get(path) or {}).get("modal_share", 0) >= 0.9
        )
        if len(dead) >= 3:
            reasons.append(
                f"on the {path} path {len(dead)} of {len(audit)} weighted "
                f"components are one value for 90%+ of published items "
                f"({', '.join(dead)}) — the weights do not describe what ranks"
            )
    return {"provisional": bool(reasons), "reasons": reasons}


def calibrate_threshold(
    target_low: float = 0.30, target_high: float = 0.50, apply: bool = False
) -> dict[str, Any]:
    """Q3: pick the headline threshold that lands the headline rate in-band.

    The rate is computed **per day** — the fraction of days that would carry a
    headline — not the fraction of items, because that is the thing the reader
    experiences.

    It is computed over the items that would have been *published*, not the whole
    candidate pool. A day's headline is the top card of its issue, and the issue
    holds 24 of the ~190 candidates; the pool's top score belongs to an item the
    reader never sees.
    """
    rows = load_scores()
    if not rows:
        return {"status": "NO_DATA", "hint": "run `uc backfill --days 90` first"}

    threshold_sel = float(cfg("classifier.threshold", 0.5))
    population = "published"
    if any("published" in r for r in rows):
        selected = [r for r in rows if r.get("published")]
    else:
        population = "candidate_pool"
        # A scores.jsonl written before the published flag existed. Fall back to
        # the candidate pool and say so, rather than silently reporting zero.
        selected = [r for r in rows if r["relevance"] >= threshold_sel]
    if not selected:
        return {"status": "NO_SELECTED_ITEMS", "n_scored": len(rows)}

    by_day: dict[str, float] = {}
    for r in selected:
        d = r["date"]
        if d:
            by_day[d] = max(by_day.get(d, 0.0), r["headline"])
    day_tops = sorted(by_day.values())
    if not day_tops:
        return {"status": "NO_DATED_ITEMS", "n_scored": len(rows)}

    target_mid = (target_low + target_high) / 2
    n_days = len(day_tops)

    # A headline appears on a day when that day's top score clears the
    # threshold, so a quantile of the daily tops looks like the right estimator.
    # It is not, because the tops are massively tied: every day publishes at
    # least one whitelist journal article, they all score exactly 0.44, and 57
    # of 90 daily tops are therefore that same number. The quantile lands inside
    # that tie mass and `>=` then admits every tied day at once — 100%, when a
    # threshold one increment higher gives 37%.
    #
    # So the rate is not estimated, it is enumerated. The rate only changes at a
    # distinct daily top, so those values are the complete set of achievable
    # thresholds; pick the one whose *measured* rate sits closest to the middle
    # of the band, preferring a threshold that is in band at all. Ties in that
    # comparison go to the lower threshold, which is the more inclusive choice.
    achievable = [
        (v, sum(1 for t in day_tops if t >= v) / n_days) for v in sorted(set(day_tops))
    ]
    in_band_options = [c for c in achievable if target_low <= c[1] <= target_high]
    chosen, rate = min(
        in_band_options or achievable, key=lambda c: (abs(c[1] - target_mid), c[0])
    )
    quantile_threshold = _quantile(day_tops, 1 - target_mid)
    component_audit = _component_audit(selected)

    scores = sorted(r["headline"] for r in selected)
    hist_edges = [round(i / 20, 2) for i in range(21)]
    hist = Counter()
    for s in scores:
        bucket = min(19, int(s * 20))
        hist[round(bucket / 20, 2)] += 1

    result = {
        "status": "OK",
        "population": population,
        "n_scored": len(rows),
        "n_selected": len(selected),
        "n_days": n_days,
        # Where the chosen threshold actually sits, not where it was aimed.
        "quantile": round(1 - rate, 4),
        "headline_threshold": round(chosen, 4),
        "headline_rate": round(rate, 4),
        "target_band": [target_low, target_high],
        "in_band": target_low <= rate <= target_high,
        # Kept so the tie problem stays visible in the artifact rather than
        # living only in this function's comments.
        "quantile_method": {
            "threshold": round(quantile_threshold, 4),
            "rate": round(
                sum(1 for v in day_tops if v >= quantile_threshold) / n_days, 4
            ),
            "distinct_daily_tops": len(achievable),
        },
        "achievable_rates": [
            {"threshold": round(v, 4), "rate": round(r, 4)} for v, r in achievable
        ],
        "novelty_decay": _novelty_decay(selected),
        "component_audit": component_audit,
        **_provisional(day_tops, component_audit),
        "score_quantiles": {
            str(q): round(_quantile(scores, q), 4)
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "histogram": {str(e): hist.get(e, 0) for e in hist_edges[:-1]},
        "daily_distribution": daily_distribution(rows, threshold_sel),
    }

    if apply:
        _write_threshold(result)
        result["applied_to"] = str(paths.CONFIG / "scoring.yaml")

    backfill_dir().mkdir(parents=True, exist_ok=True)
    (backfill_dir() / "calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def _write_threshold(result: dict[str, Any]) -> None:
    """Rewrite the two calibration blocks in config/scoring.yaml in place.

    Line editing rather than a YAML round-trip: the comments in that file explain
    the weights, and re-serialising would throw them away.
    """
    path = paths.CONFIG / "scoring.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_calibration = False
    for line in lines:
        if line.startswith("headline_threshold:"):
            out.append(f"headline_threshold: {result['headline_threshold']}")
            continue
        if line.startswith("calibration:"):
            in_calibration = True
            out.append("calibration:")
            out.append(f"  quantile: {result['quantile']}")
            out.append(f"  headline_rate: {result['headline_rate']}")
            # Named for the population it counts, not "n_scored" — that label
            # meant the candidate pool and now means the published set, and
            # numbers whose population is implicit are what caused this bug.
            out.append(f"  n_published: {result['n_selected']}")
            out.append(f"  n_days: {result['n_days']}")
            out.append(f"  calibrated_at: {date.today().isoformat()}")
            out.append('  source: "backfill"')
            out.append(f"  provisional: {str(result.get('provisional', False)).lower()}")
            for reason in result.get("reasons") or []:
                out.append(f"  # PROVISIONAL: {reason}")
            continue
        if in_calibration:
            if line.startswith("  ") or not line.strip():
                continue
            in_calibration = False
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    scoring_config.cache_clear()
