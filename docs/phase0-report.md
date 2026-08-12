# Urban Currents — Phase 0 report

Generated 2026-08-12T16:28:13+00:00 by `uc report`. Every figure below is computed from files in this repository; anything not measured says so.

## The four questions (PRD §1)

| Q | Question | Criterion | Measured | Verdict |
|---|---|---|---|---|
| Q1a | Is the filter usable? | holdout AUC >= 0.9 | 0.9761 | PASS |
| Q1b | Is the filter usable? | precision@10 >= 0.7 | PENDING-HUMAN | PENDING-HUMAN |
| Q2 | Is there enough signal for a daily? | median >= 5 items/day | 28 | PASS |
| Q3 | Where does the quiet-day line go? | headline rate 30-50% | 0.4 | PASS |
| Q4 | Does review fit the budget? | median <= 15 min/day | PENDING-HUMAN | PENDING-HUMAN |

> Q1b and Q4 require a human at the keyboard: `uc review --label relevance` for precision@10, `uc review` for the timing. They are marked PENDING-HUMAN rather than guessed.

## Relevance classifier (PRD §5.4)

Model `clf-2026-08-12` — BAAI/bge-base-en-v1.5 embeddings (768d) into logistic regression.

| metric | value |
|---|---|
| holdout AUC | 0.9761 |
| average precision | 0.9815 |
| precision @ 0.35 | 0.9068 |
| recall @ 0.35 | 0.9359 |
| training examples | 7897 |
| holdout examples | 1580 |

**Per-source behaviour on the holdout.** The predicted failure mode is that journal-heavy training scores arXiv urban papers too low (PRD §5.4, §10), so it is measured directly:

| source | n | mean probability | recall / FPR |
|---|---|---|---|
| arxiv_other | 800 | 0.1242 | FPR 0.0938 |
| arxiv_urban | 211 | 0.6765 | 0.7678 |
| journal | 569 | 0.9544 | 0.9982 |

**Threshold sweep.** The headline AUC hides the decision that actually matters. The selection threshold is set from this table, not from a default:

| threshold | arXiv-urban recall | journal recall | negative FPR | holdout precision |
|---|---|---|---|---|
| 0.1 | 0.934 | 1 | 0.351 | 0.732 |
| 0.15 | 0.886 | 1 | 0.24 | 0.797 |
| 0.2 | 0.839 | 1 | 0.168 | 0.848 |
| 0.25 | 0.82 | 0.998 | 0.142 | 0.867 |
| 0.3 | 0.815 | 0.998 | 0.117 | 0.887 |
| 0.35 | 0.768 | 0.998 | 0.094 | 0.907 |
| 0.4 | 0.758 | 0.997 | 0.07 | 0.928 |
| 0.45 | 0.73 | 0.997 | 0.058 | 0.94 |
| 0.5 | 0.682 | 0.995 | 0.044 | 0.953 |
| 0.55 | 0.673 | 0.99 | 0.035 | 0.962 |
| 0.6 | 0.663 | 0.988 | 0.028 | 0.97 |
| 0.65 | 0.626 | 0.982 | 0.02 | 0.977 |
| 0.7 | 0.607 | 0.979 | 0.016 | 0.981 |
| 0.75 | 0.554 | 0.974 | 0.013 | 0.985 |
| 0.8 | 0.493 | 0.954 | 0.009 | 0.989 |

Configured selection threshold: **0.35**. Note that holdout precision is measured on a roughly balanced sample; the live base rate is far lower, so live precision is lower than this column suggests. Q1b's labelling is the test that settles it.

Training set: 2800 journal positives + 1097 arXiv-urban positives + 4000 negatives = 7897. The arXiv share of positives is 0.3 by design — journal prose and arXiv prose differ, and training on journals alone down-scores exactly the arXiv urban computing papers this product exists to find.

## Volume and gate (PRD §5.3, Q2)

Backfill 2026-05-14 → 2026-08-11 (90 days):

| stage | count |
|---|---|
| candidates collected | 32730 |
| after dedup | 32728 |
| after gate | 12443 |
| rejected by gate | 20285 |
| above relevance 0.35 | 2419 |

Per-day selected items over 90 days — median **28**, p25 18.2, p75 34.8, range 6–53.

**arXiv intake and gate outcome by category**, measured over the backfill. Cross-listed papers count under each of their categories, so the totals sum above the item count. The four low-volume categories are not gated at all (PRD §5.3), which is why their pass rate is 1.0:

| category | items | per day | through gate | gate pass rate |
|---|---|---|---|---|
| cs.AI | 15605 | 173.4 | 5008 | 0.321 |
| cs.LG | 13960 | 155.1 | 4140 | 0.297 |
| cs.CV | 9871 | 109.7 | 4410 | 0.447 |
| cs.CL | 4062 | 45.1 | 1128 | 0.278 |
| cs.RO | 1601 | 17.8 | 814 | 0.508 |
| stat.ML | 1538 | 17.1 | 414 | 0.269 |
| cs.CY | 1350 | 15.0 | 1350 | 1.0 |
| cs.CR | 1333 | 14.8 | 369 | 0.277 |
| cs.SE | 881 | 9.8 | 251 | 0.285 |
| cs.HC | 830 | 9.2 | 432 | 0.52 |
| stat.AP | 730 | 8.1 | 730 | 1.0 |
| cs.MA | 677 | 7.5 | 283 | 0.418 |
| cs.IR | 674 | 7.5 | 190 | 0.282 |
| stat.ME | 500 | 5.6 | 325 | 0.65 |
| cs.SD | 497 | 5.5 | 84 | 0.169 |

**Gate recall check** — 200 items drawn from 20867 rejects; **3** scored above the selection threshold (limit 3). Verdict: **GATE_OK**.

Highest-scoring rejected items:

| score | title |
|---|---|
| 0.628 | What Color is the Sky (for a non-human) ? |
| 0.483 | Honey, I Shrunk the Arc de Triomphe! |
| 0.41 | CHM-Net: Center Heatmap-driven Macro-Micro Modeling Network for MRI-based Microbial Density Stratification |

## Quiet-day threshold (PRD §5.6, Q3)

Chosen threshold **0.5872** — the 0.6 quantile of daily top scores across 90 days, giving a headline rate of **40.0%** against a 30–50% target (in band).

Headline-score quantiles over the selected backfill items:

| quantile | score |
|---|---|
| 0.1 | 0.1744 |
| 0.25 | 0.245 |
| 0.5 | 0.3508 |
| 0.75 | 0.4252 |
| 0.9 | 0.5127 |
| 0.95 | 0.5553 |
| 0.99 | 0.5988 |

Distribution (bucket → count):

```
  0.1 ######### 72
 0.15 ###################################### 302
  0.2 ################################ 259
 0.25 ################################### 278
  0.3 #################################### 288
 0.35 ############################################################ 476
  0.4 ############################### 249
 0.45 ########################## 213
  0.5 ################## 146
 0.55 ############## 114
  0.6 # 14
 0.65 # 6
  0.7 # 2
```

Current `config/scoring.yaml` threshold: 0.5872 (source: backfill).

## Cost, measured (PRD §1, §9)

| item | value |
|---|---|
| days of runs | 1 |
| items published | 24 |
| items summarised | 0 |
| LLM | $0.0 |
| OpenAlex | $0.0006 |
| embeddings (local) | $0.0 |
| total | $0.0006 |
| per published item | $3e-05 |
| monthly estimate | $0.018 |
| tokens in / out | 0 / 0 |

Embeddings are local (`BAAI/bge-base-en-v1.5` on CPU), so their marginal cost is zero — which is what makes backfills and retraining free.

## What actually gets published

Of 24 published items, **12 came from arXiv** and 12 from whitelist journals.

The split is enforced: `classifier.arxiv_min_share` is 0.5. Without it the daily list fills with journal articles, because the classifier was trained on those journals and scores their articles ~0.99 close to by construction. Measured on 2026-08-11 with the quota disabled: 23 of 24 slots were journal articles.

## Archive

| thing | count |
|---|---|
| items | 24 |
| items with a summary | 0 |
| issues | 1 |
| quiet days | 0 |
| items with an OpenAlex ID | 12 |
| items with referenced_works | 8 |
| published (journal) items | 12 |

## Runs

| date | fetched | after gate | selected | summarised | published | skipped / failed |
|---|---|---|---|---|---|---|
| 2026-08-11 | 311 | 167 | 24 | 0 | 24 | - |

## What this report does not know

- **Q1b precision@10** — needs `uc review --label relevance` over 5 days × 30 items. This is the number that decides whether the filter is usable in practice; the holdout AUC does not answer it.
- **Q4 review time** — needs `uc review` run by a human. The CLI records it automatically; nothing else can.
- **Summary quality and per-item LLM cost** — no summaries were generated, so neither can be reported. See the run errors for why.

---

Regenerate with `uc report`. Sources: `runs/*/metrics.json`, `models/clf-*.json`, `runs/backfill/`, `runs/labels/`, `content/`.
