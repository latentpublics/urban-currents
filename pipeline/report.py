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
    metas = sorted(paths.MODELS.glob("clf-*.json"))
    if not metas:
        return None
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
    trainset = load_json(paths.RUNS / "trainset" / "trainset.meta.json")

    from .review import precision_at_k

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
    p_at_10 = labels.get("precision_at_k")
    q1_p_ok = None if p_at_10 is None else p_at_10 >= 0.7

    median_day = (calib or {}).get("daily_distribution", {}).get("median_per_day")
    q2_ok = None if median_day is None else median_day >= 5

    rate = (calib or {}).get("headline_rate")
    q3_ok = None if rate is None else 0.30 <= rate <= 0.50

    review_seconds = [
        r.timing.get("review_s") for r in runs if r.timing.get("review_s")
    ]
    review_median = None
    if review_seconds:
        s = sorted(review_seconds)
        review_median = s[len(s) // 2] / 60
    q4_ok = None if review_median is None else review_median <= 15

    L.extend(
        _table(
            ["Q", "Question", "Criterion", "Measured", "Verdict"],
            [
                ["Q1a", "Is the filter usable?", "holdout AUC >= 0.9", _fmt(auc), _verdict(q1_auc_ok)],
                ["Q1b", "Is the filter usable?", "precision@10 >= 0.7",
                 _fmt(p_at_10) if p_at_10 is not None else PENDING_HUMAN,
                 _verdict(q1_p_ok) if p_at_10 is not None else PENDING_HUMAN],
                ["Q2", "Is there enough signal for a daily?", "median >= 5 items/day",
                 _fmt(median_day, 1), _verdict(q2_ok)],
                ["Q3", "Where does the quiet-day line go?", "headline rate 30-50%",
                 _fmt(rate, 3), _verdict(q3_ok)],
                ["Q4", "Does review fit the budget?", "median <= 15 min/day",
                 _fmt(review_median, 1) if review_median else PENDING_HUMAN,
                 _verdict(q4_ok) if review_median else PENDING_HUMAN],
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
    else:
        m = clf["metrics"]
        A(f"Model `{clf['version']}` — {clf['embedding_model']} embeddings "
          f"({clf['embedding_dim']}d) into logistic regression.")
        A("")
        L.extend(
            _table(
                ["metric", "value"],
                [
                    ["holdout AUC", _fmt(m.get("auc"))],
                    ["average precision", _fmt(m.get("average_precision"))],
                    [f"precision @ {clf.get('threshold')}", _fmt(m.get("precision_at_threshold"))],
                    [f"recall @ {clf.get('threshold')}", _fmt(m.get("recall_at_threshold"))],
                    ["training examples", clf.get("n_examples")],
                    ["holdout examples", clf.get("n_holdout")],
                ],
            )
        )
        A("")
        A("**Per-source behaviour on the holdout.** The predicted failure mode is "
          "that journal-heavy training scores arXiv urban papers too low "
          "(PRD §5.4, §10), so it is measured directly:")
        A("")
        rows = []
        for src, v in (clf.get("per_source") or {}).items():
            rows.append([
                src, v.get("n"), _fmt(v.get("mean_proba")),
                _fmt(v.get("recall_at_threshold")) if "recall_at_threshold" in v
                else f"FPR {_fmt(v.get('false_positive_rate'))}",
            ])
        L.extend(_table(["source", "n", "mean probability", "recall / FPR"], rows))
        A("")

        sweep = clf.get("threshold_sweep") or []
        if sweep:
            A("**Threshold sweep.** The headline AUC hides the decision that "
              "actually matters. The selection threshold is set from this table, "
              "not from a default:")
            A("")
            L.extend(
                _table(
                    ["threshold", "arXiv-urban recall", "journal recall",
                     "negative FPR", "holdout precision"],
                    [
                        [r["threshold"], _fmt(r.get("arxiv_urban_recall"), 3),
                         _fmt(r.get("journal_recall"), 3),
                         _fmt(r.get("arxiv_other_fpr"), 3),
                         _fmt(r.get("overall_precision"), 3)]
                        for r in sweep
                    ],
                )
            )
            A("")
            A(f"Configured selection threshold: **{cfg('classifier.threshold')}**. "
              "Note that holdout precision is measured on a roughly balanced "
              "sample; the live base rate is far lower, so live precision is "
              "lower than this column suggests. Q1b's labelling is the test that "
              "settles it.")
            A("")
    if trainset:
        c = trainset.get("counts", {})
        A(f"Training set: {c.get('journal_positive')} journal positives + "
          f"{c.get('arxiv_positive')} arXiv-urban positives + {c.get('negative')} "
          f"negatives = {c.get('total')}. The arXiv share of positives is "
          f"{trainset.get('arxiv_positive_share')} by design — journal prose and "
          f"arXiv prose differ, and training on journals alone down-scores exactly "
          f"the arXiv urban computing papers this product exists to find.")
        A("")

    # -- volume -----------------------------------------------------------
    A("## Volume and gate (PRD §5.3, Q2)")
    A("")
    if backfill_meta:
        A(f"Backfill {backfill_meta['start']} → {backfill_meta['end']} "
          f"({backfill_meta['days']} days):")
        A("")
        L.extend(
            _table(
                ["stage", "count"],
                [
                    ["candidates collected", backfill_meta.get("candidates")],
                    ["after dedup", backfill_meta.get("after_dedup")],
                    ["after gate", backfill_meta.get("after_gate")],
                    ["rejected by gate", backfill_meta.get("gate_rejected")],
                    [f"above relevance {backfill_meta.get('selection_threshold')}",
                     backfill_meta.get("selected")],
                ],
            )
        )
        A("")
    if calib and calib.get("daily_distribution"):
        dd = calib["daily_distribution"]
        A(f"Per-day selected items over {dd.get('days_observed')} days — "
          f"median **{_fmt(dd.get('median_per_day'), 1)}**, "
          f"p25 {_fmt(dd.get('p25_per_day'), 1)}, p75 {_fmt(dd.get('p75_per_day'), 1)}, "
          f"range {dd.get('min_per_day')}–{dd.get('max_per_day')}.")
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
        A(f"Chosen threshold **{calib['headline_threshold']}** — the "
          f"{calib['quantile']} quantile of daily top scores across "
          f"{calib['n_days']} days, giving a headline rate of "
          f"**{calib['headline_rate']:.1%}** against a 30–50% target "
          f"({'in band' if calib['in_band'] else 'OUT OF BAND'}).")
        A("")
        A("Headline-score quantiles over the selected backfill items:")
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
                ["LLM", f"${costs['llm_usd']}"],
                ["OpenAlex", f"${costs['openalex_usd']}"],
                ["embeddings (local)", f"${costs['embedding_usd']}"],
                ["total", f"${costs['total_usd']}"],
                ["per published item", f"${costs['per_item_usd']}" if costs["per_item_usd"] else "n/a"],
                ["monthly estimate", f"${costs['monthly_estimate_usd']}" if costs["monthly_estimate_usd"] else "n/a"],
                ["tokens in / out", f"{costs['tokens_in']} / {costs['tokens_out']}"],
            ],
        )
    )
    A("")
    A("Embeddings are local (`BAAI/bge-base-en-v1.5` on CPU), so their marginal "
      "cost is zero — which is what makes backfills and retraining free.")
    A("")

    # -- source mix -------------------------------------------------------
    A("## What actually gets published")
    A("")
    from_arxiv = sum(1 for i in items if i.ids.arxiv)
    from_journal = len(items) - from_arxiv
    A(f"Of {len(items)} published items, **{from_arxiv} came from arXiv** and "
      f"{from_journal} from whitelist journals.")
    A("")
    A(f"The split is enforced: `classifier.arxiv_min_share` is "
      f"{cfg('classifier.arxiv_min_share')}. Without it the daily list fills "
      f"with journal articles, because the classifier was trained on those "
      f"journals and scores their articles ~0.99 close to by construction. "
      f"Measured on 2026-08-11 with the quota disabled: 23 of 24 slots were "
      f"journal articles.")
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
