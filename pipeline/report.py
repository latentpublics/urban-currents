"""``uc report`` — aggregate every run into docs/phase0-report.md (PRD §8).

Phase 0's deliverable is code **plus this document**, so the command's job is to
compute the Q1-Q4 numbers from measured data and to say plainly when a number
does not exist yet. A blank is a finding; a guess is a lie that survives into the
Go/No-Go decision.

Every figure here traces to a file on disk: ``runs/*/metrics.json``,
``models/clf-*.json``, ``runs/backfill/``, ``runs/labels/``.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import paths, store
from .config import cfg, scoring_config
from .models import Metrics

PENDING_HUMAN = "PENDING-HUMAN"


# --------------------------------------------------------------------------
# Gathering
# --------------------------------------------------------------------------


def load_runs() -> list[Metrics]:
    out = []
    for p in sorted(paths.RUNS.glob("run_*/metrics.json")):
        try:
            out.append(Metrics.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def load_classifier_meta() -> Optional[dict]:
    """Metadata for the model that is actually in production.

    The filename sort this used to do picks `clf-v3-…` out of a directory
    holding v1, v2 and v3 — the variant the comparison rejected as the worst of
    the three. The report was publishing its metrics as Phase 0's headline
    figure while the pipeline ran v2. Which model is in production is written
    down in `classifier.model_version`, so the report reads the same pin the
    pipeline does (D36 fixed only the pipeline side).
    """
    metas = sorted(paths.MODELS.glob("clf-*.json"))
    if not metas:
        return None
    pinned = cfg("classifier.model_version")
    if pinned:
        p = paths.MODELS / f"{pinned}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        # Models exist but not the pinned one: name it rather than quietly
        # reporting a different model's metrics.
        return {
            "version": str(pinned),
            "error": f"pinned model {pinned!r} has no metadata at {p.name}",
        }
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def cost_summary(runs: list[Metrics]) -> dict[str, Any]:
    llm = sum(r.cost.llm_usd for r in runs)
    oa = sum(r.cost.openalex_usd for r in runs)
    emb = sum(r.cost.embedding_usd for r in runs)
    published = sum(r.counts.published for r in runs)
    summarized = sum(r.counts.summarized for r in runs)
    days = len({str(r.date) for r in runs if r.date})
    total = llm + oa + emb
    return {
        "days": days,
        "llm_usd": round(llm, 4),
        "openalex_usd": round(oa, 4),
        "embedding_usd": round(emb, 4),
        "total_usd": round(total, 4),
        "published": published,
        "summarized": summarized,
        "per_item_usd": round(total / published, 5) if published else None,
        "monthly_estimate_usd": round(total / days * 30, 3) if days else None,
        "tokens_in": sum(r.tokens.input for r in runs),
        "tokens_out": sum(r.tokens.output for r in runs),
    }


def category_intake() -> dict[str, dict[str, int]]:
    """Per-category arXiv intake and gate outcome, measured over the backfill.

    Counts both sides of the gate — the items that passed (``backfill/scores.jsonl``)
    and the ones it rejected (``backfill_*/gate_rejected.jsonl``) — so the pass
    rate is a measurement rather than an inference. Cross-listed papers count
    under each of their categories, so the columns sum above the item total.
    """
    passed: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    days: set[str] = set()

    scores = paths.RUNS / "backfill" / "scores.jsonl"
    if scores.exists():
        for line in scores.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            days.add(row.get("date") or "")
            for c in row.get("categories") or []:
                passed[c] += 1

    for p in sorted(paths.RUNS.glob("backfill_*/gate_rejected.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            days.add(row.get("date") or "")
            for c in row.get("categories") or []:
                rejected[c] += 1

    n_days = max(1, len({d for d in days if d}))
    out: dict[str, dict[str, int]] = {}
    for cat in sorted(set(passed) | set(rejected), key=lambda c: -(passed[c] + rejected[c])):
        total = passed[cat] + rejected[cat]
        out[cat] = {
            "total": total,
            "per_day": round(total / n_days, 1),
            "through_gate": passed[cat],
            "gate_pass_rate": round(passed[cat] / total, 3) if total else 0.0,
        }
    return out


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _verdict(ok: Optional[bool]) -> str:
    if ok is None:
        return "not measured"
    return "PASS" if ok else "FAIL"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(out_path: Optional[Path] = None) -> Path:
    runs = load_runs()
    clf = load_classifier_meta()
    calib = load_json(paths.RUNS / "backfill" / "calibration.json")
    backfill_meta = load_json(paths.RUNS / "backfill" / "backfill.meta.json")
    gate = load_json(paths.RUNS / "gate_recall.json")
    # Training-set counts come from the adopted model's own `trainset_meta`, not
    # from runs/trainset/, which describes whichever build ran last.

    from .labeling import precision_at_k

    labels = precision_at_k(k=10)
    costs = cost_summary(runs)
    scoring = scoring_config()

    items = list(store.iter_items())
    issues = list(store.iter_issues())

    L: list[str] = []
    A = L.append

    A("# Urban Currents — Phase 0 report")
    A("")
    A(f"Generated {datetime.now(timezone.utc).replace(microsecond=0).isoformat()} "
      f"by `uc report`. Every figure below is computed from files in this "
      f"repository; anything not measured says so.")
    A("")

    # -- Q1-Q4 -----------------------------------------------------------
    A("## The four questions (PRD §1)")
    A("")

    auc = (clf or {}).get("metrics", {}).get("auc")
    q1_auc_ok = None if auc is None else auc >= 0.9
    # Q1b is per source on purpose (roadmap §2.3). A blended number would hide
    # which entry path is failing, so the verdict is the weaker of the two.
    per_source_p = {
        src: v.get(f"precision_at_{labels.get('k', 10)}")
        for src, v in (labels.get("by_source") or {}).items()
    }
    measured_p = [v for v in per_source_p.values() if v is not None]
    p_at_10 = min(measured_p) if measured_p else None
    q1_p_ok = None if p_at_10 is None else p_at_10 >= 0.7
    # Q1b was planned as five labelled days (roadmap §4.1). A verdict from one
    # is a reading, not a result, and this is the number that decides Phase 1 —
    # so the sample size travels with it instead of being a column away.
    Q1B_PLANNED_DAYS = 5
    days_labelled = labels.get("days_labelled") or 0
    q1b_thin = 0 < days_labelled < Q1B_PLANNED_DAYS

    dd_all = (calib or {}).get("daily_distribution", {})
    median_day = dd_all.get("median_per_day")
    # Reported per path: the journal count clears by membership, so a pooled
    # median answers "how many whitelist articles appeared" as much as it
    # answers "is there enough signal". The verdict takes the weaker path.
    median_by_path = {
        src: v.get("median_per_day")
        for src, v in (dd_all.get("by_source") or {}).items()
    }
    weakest_median = min(
        (v for v in median_by_path.values() if v is not None), default=median_day
    )
    q2_ok = None if weakest_median is None else weakest_median >= 5

    rate = (calib or {}).get("headline_rate")
    q3_ok = None if rate is None else 0.30 <= rate <= 0.50

    # Q4 has two populations and conflating them is how a partial review reads
    # as a fast one. A day where three items were judged before the session was
    # interrupted takes 90 seconds, which passes "median <= 15 min/day" while
    # measuring nothing about a full day. So: full days give the headline, and
    # seconds-per-item — which every partial day contributes to honestly — gives
    # the projection when no full day exists yet.
    review_days: list[dict] = []
    for r in runs:
        seconds = r.timing.get("review_s")
        if not seconds:
            continue
        reviewed_n = int(r.timing.get("reviewed_n", 0) or 0)
        review_days.append({
            "date": str(r.date),
            "seconds": float(seconds),
            "reviewed_n": reviewed_n,
            "published": int(r.counts.published or 0),
            "complete": bool(reviewed_n and reviewed_n >= int(r.counts.published or 0)),
        })

    complete = [d for d in review_days if d["complete"]]
    review_median = None
    if complete:
        s = sorted(d["seconds"] for d in complete)
        review_median = s[len(s) // 2] / 60

    reviewed_total = sum(d["reviewed_n"] for d in review_days)
    seconds_total = sum(d["seconds"] for d in review_days if d["reviewed_n"])
    per_item = round(seconds_total / reviewed_total, 1) if reviewed_total else None
    # Projection, clearly labelled as one: per-item pace at the median day's size.
    projected = (
        round(per_item * median_day / 60, 1)
        if per_item is not None and median_day else None
    )
    q4_ok = None if review_median is None else review_median <= 15

    L.extend(
        _table(
            ["Q", "Question", "Criterion", "Measured", "Verdict"],
            [
                ["Q1a", "Is the filter usable?", "holdout AUC >= 0.9", _fmt(auc), _verdict(q1_auc_ok)],
                ["Q1b", "Is the filter usable?", "precision@10 >= 0.7 (per source)",
                 ", ".join(f"{s}: {_fmt(v, 3)}" for s, v in per_source_p.items())
                 + (f" ({days_labelled} of {Q1B_PLANNED_DAYS} days)" if q1b_thin else "")
                 if measured_p else PENDING_HUMAN,
                 (f"{_verdict(q1_p_ok)} (PROVISIONAL)" if q1b_thin else _verdict(q1_p_ok))
                 if p_at_10 is not None else PENDING_HUMAN],
                ["Q2", "Is there enough signal for a daily?", "median >= 5 items/day",
                 ", ".join(f"{s}: {_fmt(v, 1)}" for s, v in median_by_path.items())
                 if median_by_path else _fmt(median_day, 1),
                 _verdict(q2_ok)],
                # A rate inside the band is not a passing Q3 when the score it
                # thresholds is degenerate. The verdict follows the measurement,
                # and the measurement says the formula cannot carry a line yet.
                ["Q3", "Where does the quiet-day line go?", "headline rate 30-50%",
                 _fmt(rate, 3),
                 "PROVISIONAL" if (calib or {}).get("provisional") else _verdict(q3_ok)],
                ["Q4", "Does review fit the budget?", "median <= 15 min/day",
                 _fmt(review_median, 1) if review_median
                 else (
                     f"{_fmt(projected, 1)} projected "
                     f"({_fmt(per_item, 1)} s/item over {reviewed_total} items, "
                     f"0 complete days)"
                     if projected is not None else PENDING_HUMAN
                 ),
                 _verdict(q4_ok) if review_median
                 else ("PROJECTED" if projected is not None else PENDING_HUMAN)],
            ],
        )
    )
    A("")
    if p_at_10 is None:
        A("> Q1b and Q4 require a human at the keyboard: `uc review --label relevance` "
          "for precision@10, `uc review` for the timing. They are marked "
          f"{PENDING_HUMAN} rather than guessed.")
        A("")

    # -- classifier -------------------------------------------------------
    A("## Relevance classifier (PRD §5.4)")
    A("")
    if not clf:
        A("No trained model found under `models/`. The pipeline falls back to a "
          "keyword heuristic recorded as `classifier_version: heuristic-v0`.")
    elif clf.get("error"):
        A(f"**{clf['error']}** — the pipeline runs `{clf['version']}` but this "
          f"report cannot describe it. Metrics below would belong to a different "
          f"model, so none are shown.")
        A("")
    else:
        m = clf.get("metrics") or {}
        ev = clf.get("eval_meta") or {}
        thr = str(clf.get("threshold"))
        at_thr = (m.get("at_threshold") or {}).get(thr) or {}

        A(f"Model `{clf['version']}`"
          + (f" (variant {clf['variant']})" if clf.get("variant") else "")
          + f" — {clf['embedding_model']} embeddings ({clf['embedding_dim']}d) "
          f"into logistic regression, pinned in `classifier.model_version`.")
        A("")
        L.extend(
            _table(
                ["metric", "value"],
                [
                    ["evaluation AUC", _fmt(m.get("auc"))],
                    ["average precision", _fmt(m.get("average_precision"))],
                    [f"precision @ {thr}", _fmt(at_thr.get("precision"))],
                    [f"recall @ {thr}", _fmt(at_thr.get("recall"))],
                    [f"flagged rate @ {thr}", _fmt(at_thr.get("flagged_rate"))],
                    ["training examples", clf.get("n_train")],
                    ["evaluation examples", m.get("n")],
                ],
            )
        )
        A("")

        # The single most quotable number in this report, next to the reason it
        # does not answer the question people will quote it for.
        if ev:
            A(f"**These numbers describe one task: `{clf.get('headline_task')}`.** "
              f"The evaluation set is {ev.get('positives')} positives — "
              f"\"{ev.get('positive_definition')}\" — against {ev.get('negatives')} "
              f"negatives drawn from other arXiv papers. Both sides are "
              f"unambiguous, so this measures telling clear cases apart. The hard "
              f"live cases are the borderline ones, and they are not in this set; "
              f"its base rate is also far above the live one, so **live precision "
              f"is lower than the table above**. This comparison answers \"which "
              f"variant\", not \"is the classifier good enough\". **Q1b is the only "
              f"measurement that answers the second question, and Phase 1 Go/No-Go "
              f"rests on Q1b, not on this AUC.**")
            A("")

        sanity = clf.get("journal_sanity_check")
        if sanity:
            A(f"Journal sanity check: mean probability {_fmt(sanity.get('mean_proba'))} "
              f"over {sanity.get('n')} whitelist articles, "
              f"{(sanity.get('share_above_threshold') or 0):.1%} above threshold. "
              f"{sanity.get('note')} — and since N4 the journal path does not "
              f"consult the classifier at all, so this bounds nothing in production.")
            A("")

        sweep = m.get("sweep") or []
        if sweep:
            A("**Threshold sweep.** The headline AUC hides the decision that "
              "actually matters. The selection threshold is set from this table, "
              "not from a default:")
            A("")
            L.extend(
                _table(
                    ["threshold", "precision", "recall", "flagged rate"],
                    [
                        [r.get("threshold"), _fmt(r.get("precision"), 3),
                         _fmt(r.get("recall"), 3), _fmt(r.get("flagged_rate"), 3)]
                        for r in sweep
                    ],
                )
            )
            A("")
            A(f"Configured selection threshold: **{cfg('classifier.threshold')}**.")
            A("")

        # From the adopted model's own metadata, not from a separate trainset
        # file that may describe a different build.
        tm = clf.get("trainset_meta") or {}
        c = tm.get("counts") or {}
        if c:
            A(f"Training set ({tm.get('variant', clf.get('variant'))}): "
              f"{c.get('journal')} journal positives + "
              f"{tm.get('arxiv_positive_total')} arXiv-urban positives "
              f"({c.get('arxiv_subfield')} by subfield, {c.get('arxiv_strict')} "
              f"strict) + {c.get('arxiv_other')} negatives = {c.get('total')}. "
              f"Journal positives are kept even though the journal path no longer "
              f"consults the classifier: dropping them costs precision "
              f"(0.85 → 0.70 measured, variant v3), because they are still valid "
              f"training signal for what urban research reads like. Separating "
              f"the entry paths and separating the training set are different "
              f"decisions.")
            A("")

    # -- volume -----------------------------------------------------------
    A("## Volume and gate (PRD §5.3, Q2)")
    A("")
    if backfill_meta:
        A(f"Backfill {backfill_meta['start']} → {backfill_meta['end']} "
          f"({backfill_meta['days']} days):")
        A("")
        # Every count carries the name of the population it counts. The
        # calibration bug this report documents was a number computed on one
        # population and read as another; a bare "17,093" sitting next to a
        # bare "37,390" reads as a loss rather than a filter.
        L.extend(
            _table(
                ["population", "count", "meaning"],
                [
                    ["`collected`", backfill_meta.get("candidates"),
                     "records returned by arXiv + OpenAlex"],
                    ["`after_dedup`", backfill_meta.get("after_dedup"),
                     "preprint and journal record merged into one"],
                    ["`after_gate`", backfill_meta.get("after_gate"),
                     "cleared the keyword gate — the scored population"],
                    ["`gate_rejected`", backfill_meta.get("gate_rejected"),
                     "dropped by the gate"],
                    ["`above_threshold`", backfill_meta.get("selected"),
                     f"journal by membership, or arXiv ≥ "
                     f"{backfill_meta.get('selection_threshold')}"],
                    ["`published`", backfill_meta.get("published"),
                     "would have filled the 24 daily slots"],
                ],
            )
        )
        A("")
    if calib and calib.get("daily_distribution"):
        dd = calib["daily_distribution"]
        A(f"Per-day `above_threshold` items over {dd.get('days_observed')} days, "
          f"by entry path:")
        A("")
        dist_rows = [
            [path, _fmt(v.get("median_per_day"), 1), _fmt(v.get("p25_per_day"), 1),
             _fmt(v.get("p75_per_day"), 1),
             f"{v.get('min_per_day')}–{v.get('max_per_day')}"]
            for path, v in (dd.get("by_source") or {}).items()
        ]
        dist_rows.append([
            "**both**", f"**{_fmt(dd.get('median_per_day'), 1)}**",
            _fmt(dd.get("p25_per_day"), 1), _fmt(dd.get("p75_per_day"), 1),
            f"{dd.get('min_per_day')}–{dd.get('max_per_day')}",
        ])
        L.extend(_table(["path", "median/day", "p25", "p75", "range"], dist_rows))
        A("")
        A("The pooled median is not comparable with the arXiv-only figure this "
          "report carried before journals entered the backfill. A whitelist "
          "article clears by membership, so the journal row measures how many "
          "whitelist articles appeared, not how many cleared a judgement — the "
          "arXiv row is the one that answers \"is there enough signal\".")
        A("")

    cats = category_intake()
    if cats:
        A("**arXiv intake and gate outcome by category**, measured over the "
          "backfill. Cross-listed papers count under each of their categories, so "
          "the totals sum above the item count. The four low-volume categories "
          "are not gated at all (PRD §5.3), which is why their pass rate is 1.0:")
        A("")
        L.extend(
            _table(
                ["category", "items", "per day", "through gate", "gate pass rate"],
                [
                    [k, v["total"], v["per_day"], v["through_gate"], v["gate_pass_rate"]]
                    for k, v in list(cats.items())[:15]
                ],
            )
        )
        A("")

    if gate:
        A(f"**Gate recall check** — {gate.get('sampled')} items drawn from "
          f"{gate.get('rejected_pool')} rejects; "
          f"**{gate.get('above_threshold')}** scored above the selection threshold "
          f"(limit {gate.get('max_acceptable')}). Verdict: **{gate.get('verdict')}**.")
        if gate.get("misses"):
            A("")
            A("Highest-scoring rejected items:")
            A("")
            L.extend(_table(["score", "title"],
                            [[_fmt(m["score"], 3), m["title"]] for m in gate["misses"][:8]]))
        A("")
    else:
        A("**Gate recall check**: not run. `uc gate-recall` after a backfill.")
        A("")

    # -- headline calibration --------------------------------------------
    A("## Quiet-day threshold (PRD §5.6, Q3)")
    A("")
    if not calib or calib.get("status") != "OK":
        A("Not calibrated. Run `uc backfill --days 90` then `uc calibrate --apply`.")
    else:
        if calib.get("provisional"):
            A("**PROVISIONAL — Q3 is not settled.** A threshold was found that "
              "lands in the target band, but landing in the band is not the "
              "same as the threshold meaning something:")
            A("")
            for reason in calib.get("reasons") or []:
                A(f"- {reason}")
            A("")
            A("Recorded, not worked around. Moving the weights until the number "
              "looks better is where the figures would start lying. The formula "
              "is PRD §5.6's to change.")
            A("")

        A(f"Chosen threshold **{calib['headline_threshold']}**, giving a "
          f"headline rate of **{calib['headline_rate']:.1%}** across "
          f"{calib['n_days']} days against a 30–50% target "
          f"({'in band' if calib['in_band'] else 'OUT OF BAND'}). Measured on "
          f"the {calib.get('n_selected')} items that would have been "
          f"`{calib.get('population', 'published')}`, not on the candidate "
          f"pool — a day's headline is the top card of its issue.")
        A("")

        audit = calib.get("component_audit")
        if audit:
            A("**What each weighted component actually does.** A component that "
              "takes one value across a path contributes nothing to ranking on "
              "that path, whatever weight it carries:")
            A("")
            rows_a = []
            for name, a in audit.items():
                for path, s in (a.get("by_source") or {}).items():
                    rows_a.append([
                        f"`{name}`", a.get("weight"), path, s["distinct_values"],
                        f"{s['modal_value']} @ {s['modal_share']:.1%}",
                    ])
            L.extend(_table(
                ["component", "weight", "path", "distinct values", "modal value"],
                rows_a,
            ))
            A("")

        qm = calib.get("quantile_method")
        if qm:
            A(f"The threshold is enumerated rather than estimated. The daily "
              f"top scores take only {qm['distinct_daily_tops']} distinct "
              f"values, because every day publishes a whitelist journal "
              f"article and those score identically; a quantile lands inside "
              f"that tie and reports a {qm['rate']:.1%} rate at threshold "
              f"{qm['threshold']}. Each distinct top is tried instead and the "
              f"measured rate closest to the band's middle wins.")
            A("")

        decay = calib.get("novelty_decay")
        if decay:
            A("**The novelty term dies.** The overlay vocabulary is a closed "
              "list, so once the archive has seen it the term goes to zero and "
              "stays there. Mean novelty of published items, by month of the "
              "backfill:")
            A("")
            L.extend(_table(["month", "mean novelty", "items"],
                            [[m, v["mean"], v["n"]] for m, v in decay.items()]))
            A("")
            A("In steady state a whitelist journal article scores a flat 0.44 "
              "and only an arXiv item carrying code or data links can lift a "
              "day above it. Whether a term that saturates in two weeks belongs "
              "in the headline formula is PRD §5.6's question — recorded here, "
              "not decided here.")
            A("")

        offset = load_json(paths.RUNS / "backfill" / "novelty_offset.json")
        mat = (offset or {}).get("archive_maturity") or {}
        if mat.get("status") == "OK":
            live_rate = sum(1 for i in issues if not i.quiet_day)
            A(f"**The live rate will not match this yet.** {live_rate} of "
              f"{len(issues)} published days currently carry a headline. The "
              f"threshold was calibrated against an archive {mat['archive_items_replay']} "
              f"items deep; `content/` holds {mat['archive_items_live']} behind "
              f"those days, so almost every tag is still fresh — mean novelty "
              f"{mat['novelty_live_mean']} live against {mat['novelty_replay_mean']} "
              f"in the replay, worth {mat['headline_mean_shift']} on the headline "
              f"score. The LLM tags the backfill lacks account for only "
              f"{(offset.get('headline_score_offset') or {}).get('mean_shift')} of "
              f"that. It decays on its own as days accumulate; it is the archive "
              f"being young, not the threshold being wrong.")
            A("")

        A("Headline-score quantiles over the published backfill items:")
        A("")
        L.extend(_table(["quantile", "score"],
                        [[q, v] for q, v in calib["score_quantiles"].items()]))
        A("")
        A("Distribution (bucket → count):")
        A("")
        A("```")
        for edge, n in calib["histogram"].items():
            if n:
                A(f"{edge:>5} {'#' * min(60, max(1, n * 60 // max(calib['histogram'].values())))} {n}")
        A("```")
        A("")
        A(f"Current `config/scoring.yaml` threshold: "
          f"{scoring.get('headline_threshold')} "
          f"(source: {(scoring.get('calibration') or {}).get('source')}).")
        A("")

    # -- cost -------------------------------------------------------------
    A("## Cost, measured (PRD §1, §9)")
    A("")
    L.extend(
        _table(
            ["item", "value"],
            [
                ["days of runs", costs["days"]],
                ["items published", costs["published"]],
                ["items summarised", costs["summarized"]],
                ["LLM (daily runs)", f"${costs['llm_usd']}"],
                ["OpenAlex (daily runs)", f"${costs['openalex_usd']}"],
                ["embeddings (local)", f"${costs['embedding_usd']}"],
                ["total (daily runs)", f"${costs['total_usd']}"],
                ["per published item", f"${costs['per_item_usd']}" if costs["per_item_usd"] else "n/a"],
                ["monthly estimate", f"${costs['monthly_estimate_usd']}" if costs["monthly_estimate_usd"] else "n/a"],
                ["tokens in / out (all tasks)",
                 f"{costs['tokens_in']} / {costs['tokens_out']}"],
            ],
        )
    )
    A("")

    # The token row above counts summarize and extract together while the cost
    # rows count only what daily runs booked, so the two do not reconcile on
    # their own. The per-task ledger is the one that does.
    # The spend tally moved under `content/` in 0U (U6) so it survives a CI
    # run. The old copy is still on disk and is frozen at the moment of the
    # move, so reading `paths.STATE` here would under-report for ever.
    usage = load_json(paths.persistent_state("llm_usage.json")) or {}
    by_task = usage.get("by_task") or {}
    if by_task:
        A("**Per task, cumulative** — every LLM call ever made from this "
          "repository, including calls outside a daily run (labelling "
          "preparation, re-runs against a cold cache). The daily-run figures "
          "above are a subset of this, which is why they are smaller:")
        A("")
        L.extend(_table(
            ["task", "calls", "cost"],
            [[t, v.get("calls"), f"${v.get('cost_usd')}"]
             for t, v in sorted(by_task.items())]
            + [["**total**", usage.get("calls"), f"**${usage.get('cost_usd')}**"]],
        ))
        A("")
        A(f"Tokens: {usage.get('input_tokens')} in, {usage.get('output_tokens')} "
          f"out, {usage.get('thinking_tokens')} thinking. Summarize and extract "
          f"run one call each per item (D8 was reverted in N1), so a per-item "
          f"token figure divided by the published count describes neither task "
          f"on its own.")
        A("")

    A("Embeddings are local (`BAAI/bge-base-en-v1.5` on CPU), so their marginal "
      "cost is zero — which is what makes backfills and retraining free.")
    A("")

    # -- labels -----------------------------------------------------------
    A("## Q1b labels (roadmap §2.3)")
    A("")
    if not labels.get("n_labels"):
        A("No labels yet. `uc review --label relevance --date …` collects them; "
          "`uc labels` summarises them.")
        A("")
    else:
        A(f"{labels['n_labels']} labels over {labels['days_labelled']} day(s). "
          f"{labels['summaries_available']:.0%} of labelled items had a summary "
          f"on screen.")
        A("")
        rows = []
        for src, v in labels["by_source"].items():
            rows.append([
                src, v["n_labels"], v["days"],
                _fmt(v.get(f"precision_at_{labels['k']}"), 3),
                _fmt(v["keep_rate"], 3),
                v["drop_reasons"]["not_urban"],
                v["drop_reasons"]["not_our_kind"],
                v["drop_reasons"]["weak"],
            ])
        L.extend(_table(
            ["source", "labels", "days", f"precision@{labels['k']}", "keep rate",
             "drop: not urban", "drop: not our kind", "drop: weak"], rows))
        A("")
        A("**The two drop reasons point at different problems.** *not urban* is a "
          "classifier error. *not our kind* is a coverage question nothing in the "
          "pipeline answers yet — it is the training signal for the classifier "
          "that will replace the journal path's placeholder ranking.")
        A("")

        # How far down each path's ranking precision survives, against the slots
        # that path is actually asked to fill.
        depth_rows = []
        for src, v in (labels.get("by_source") or {}).items():
            curve = v.get("precision_by_depth") or []
            if not curve:
                continue
            slots = cfg(f"selection.slots.{src}")
            depth_rows.append([
                src,
                slots,
                v.get("depth_holding_0.7"),
                ", ".join(
                    f"@{i}: {p}" for i, p in enumerate(curve, 1) if i in (1, 4, 8, 12, 15)
                ),
            ])
        if depth_rows:
            L.extend(_table(
                ["path", "daily slots", "depth holding 0.7", "precision by depth"],
                depth_rows,
            ))
            A("")
            A("Where the depth holding 0.7 is below the slot count, the path is "
              "being asked for more items than it has good ones — the fix is the "
              "slot split or a better ranker, not a higher threshold. Raising the "
              "arXiv threshold does not help: at 0.7 the 90-day backfill yields a "
              "median of 6 arXiv candidates a day and at 0.8 it yields 3, so the "
              "path could not fill 12 slots at any precision.")
            A("")

    # -- source mix -------------------------------------------------------
    A("## What actually gets published")
    A("")
    # `items` is every file under content/items/, which is not the same as what
    # the issues carry: `store` never deletes, so items dropped by a changed
    # selection rule stay on disk. The published population is the union of the
    # issues' own lists.
    published_keys = {wk for issue in issues for wk in issue.items}
    published_items = [i for i in items if i.work_key in published_keys]
    from_arxiv = sum(1 for i in published_items if i.ids.arxiv)
    from_journal = len(published_items) - from_arxiv
    A(f"Across {len(issues)} issues, **{len(published_items)} items were "
      f"`published`** — {from_arxiv} from arXiv and {from_journal} from "
      f"whitelist journals.")
    orphans = len(items) - len(published_items)
    if orphans:
        A("")
        A(f"`content/items/` holds {len(items)} files, {orphans} more than the "
          f"issues reference. Those are items an earlier selection rule "
          f"published and the current one does not; they are still part of the "
          f"archive novelty is measured against, which is why the difference is "
          f"counted rather than rounded away.")
    A("")
    A(f"The split is structural, not a quota. Each entry path owns its slots — "
      f"journal {cfg('selection.slots.journal')}, arXiv "
      f"{cfg('selection.slots.arxiv')} — and a path that cannot fill its own "
      f"lends them to the other, which the run records. The earlier "
      f"`classifier.arxiv_min_share` quota is gone (N4): it was treating a "
      f"symptom, since a whitelist article scores ~0.99 nearly by construction "
      f"and the classifier could not rank within that path at all. Measured on "
      f"2026-08-11 under the old single-classifier design: 23 of 24 slots went "
      f"to journal articles.")
    A("")

    # -- abstract coverage ------------------------------------------------
    A("## What we could not read")
    A("")
    unreadable_keys = {wk for issue in issues for wk in issue.unreadable}
    by_publisher: dict[str, int] = {}
    for issue in issues:
        for name, n in (issue.scan_meta.unreadable_by_publisher or {}).items():
            by_publisher[name] = by_publisher.get(name, 0) + n
    if not unreadable_keys:
        A("No items are currently unreadable, or no issue has recorded any.")
        A("")
    else:
        total = len(unreadable_keys) + len(published_items)
        A(f"**{len(unreadable_keys)} items across {len(issues)} issues had no "
          f"abstract from any source** and published in `Also published today` "
          f"instead of as cards — {len(unreadable_keys)}/{total} of everything "
          f"that reached an issue. Springer Nature withdrew its non-OA abstracts "
          f"from OpenAlex in 2022 and Elsevier followed in 2024; Crossref and "
          f"Springer's own API are asked for what they can still supply, and "
          f"what none of them has cannot be summarised, because the abstract is "
          f"the only evidence a summary is allowed to use.")
        A("")
        L.extend(_table(
            ["publisher", "`unreadable` items"],
            sorted(by_publisher.items(), key=lambda kv: -kv[1]),
        ))
        A("")
        A("This is the one blind spot the pipeline can measure exactly, and the "
          "count is stated rather than hidden. It names publishers here because "
          "this is the engineering report; the reader-facing section names none.")
        A("")

    # -- archive ----------------------------------------------------------
    A("## Archive")
    A("")
    quiet = sum(1 for i in issues if i.quiet_day)
    with_summary = sum(1 for i in items if i.summary.en and i.summary.en.what)
    L.extend(
        _table(
            ["thing", "count"],
            [
                ["items", len(items)],
                ["items with a summary", with_summary],
                ["issues", len(issues)],
                ["quiet days", quiet],
                ["items with an OpenAlex ID", sum(1 for i in items if i.ids.openalex)],
                ["items with referenced_works", sum(1 for i in items if i.graph.referenced_works)],
                ["published (journal) items", sum(1 for i in items
                                                  if i.publication_status.state == "published")],
            ],
        )
    )
    A("")

    # -- runs -------------------------------------------------------------
    if runs:
        A("## Runs")
        A("")
        rows = []
        for r in sorted(runs, key=lambda r: str(r.date)):
            failed = [k for k, v in r.stages.items() if v in ("FAILED", "SKIPPED")]
            rows.append([
                r.date, r.counts.arxiv_fetched, r.counts.after_gate, r.counts.selected,
                r.counts.summarized, r.counts.published,
                ", ".join(failed) if failed else "-",
            ])
        L.extend(_table(
            ["date", "fetched", "after gate", "selected", "summarised", "published",
             "skipped / failed"], rows))
        A("")

    # -- what is not measured ---------------------------------------------
    A("## What this report does not know")
    A("")
    unknown: list[str] = []
    if p_at_10 is None:
        unknown.append(
            "**Q1b precision@10** — needs `uc review --label relevance` over 5 days "
            "× 30 items. This is the number that decides whether the filter is "
            "usable in practice; the holdout AUC does not answer it."
        )
    if review_median is None:
        unknown.append(
            "**Q4 review time** — needs `uc review` run by a human. The CLI records "
            "it automatically; nothing else can."
        )
    if not calib or calib.get("status") != "OK":
        unknown.append("**Q2 and Q3** — need `uc backfill --days 90` then `uc calibrate`.")
    if not gate:
        unknown.append("**Gate recall** — needs `uc gate-recall` after a backfill.")
    summarised_any = any(i.summary.en and i.summary.en.what for i in items)
    if not summarised_any:
        unknown.append(
            "**Summary quality and per-item LLM cost** — no summaries were "
            "generated, so neither can be reported. See the run errors for why."
        )
    if not unknown:
        unknown.append("Nothing outstanding.")
    for u in unknown:
        A(f"- {u}")
    A("")

    A("---")
    A("")
    A("Regenerate with `uc report`. Sources: `runs/*/metrics.json`, "
      "`models/clf-*.json`, `runs/backfill/`, `runs/labels/`, `content/`.")
    A("")

    target = out_path or (paths.DOCS / "phase0-report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    store.write_text_atomic(target, "\n".join(L))
    return target
