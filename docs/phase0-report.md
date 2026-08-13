# Urban Currents — Phase 0 report

Generated 2026-08-13T00:08:29+00:00 by `uc report`. Every figure below is computed from files in this repository; anything not measured says so.

## The four questions (PRD §1)

| Q | Question | Criterion | Measured | Verdict |
|---|---|---|---|---|
| Q1a | Is the filter usable? | holdout AUC >= 0.9 | 0.9886 | PASS |
| Q1b | Is the filter usable? | precision@10 >= 0.7 (per source) | PENDING-HUMAN | PENDING-HUMAN |
| Q2 | Is there enough signal for a daily? | median >= 5 items/day | 72 | PASS |
| Q3 | Where does the quiet-day line go? | headline rate 30-50% | 0.367 | PROVISIONAL |
| Q4 | Does review fit the budget? | median <= 15 min/day | PENDING-HUMAN | PENDING-HUMAN |

> Q1b and Q4 require a human at the keyboard: `uc review --label relevance` for precision@10, `uc review` for the timing. They are marked PENDING-HUMAN rather than guessed.

## Relevance classifier (PRD §5.4)

Model `clf-v3-2026-08-13` — BAAI/bge-base-en-v1.5 embeddings (768d) into logistic regression.

| metric | value |
|---|---|
| holdout AUC | 0.9886 |
| average precision | 0.9694 |
| precision @ 0.35 | not measured |
| recall @ 0.35 | not measured |
| training examples | None |
| holdout examples | None |

**Per-source behaviour on the holdout.** The predicted failure mode is that journal-heavy training scores arXiv urban papers too low (PRD §5.4, §10), so it is measured directly:

| source | n | mean probability | recall / FPR |
|---|---|---|---|

Training set: 2800 journal positives + 1097 arXiv-urban positives + 4000 negatives = 7897. The arXiv share of positives is 0.3 by design — journal prose and arXiv prose differ, and training on journals alone down-scores exactly the arXiv urban computing papers this product exists to find.

## Volume and gate (PRD §5.3, Q2)

Backfill 2026-05-14 → 2026-08-11 (90 days):

| population | count | meaning |
|---|---|---|
| `collected` | 37390 | records returned by arXiv + OpenAlex |
| `after_dedup` | 37378 | preprint and journal record merged into one |
| `after_gate` | 17093 | cleared the keyword gate — the scored population |
| `gate_rejected` | 20285 | dropped by the gate |
| `above_threshold` | 6311 | journal by membership, or arXiv ≥ 0.35 |
| `published` | 2157 | would have filled the 24 daily slots |

Per-day `above_threshold` items over 90 days — median **72**, p25 49.8, p75 86.8, range 22–182.

**arXiv intake and gate outcome by category**, measured over the backfill. Cross-listed papers count under each of their categories, so the totals sum above the item count. The four low-volume categories are not gated at all (PRD §5.3), which is why their pass rate is 1.0:

| category | items | per day | through gate | gate pass rate |
|---|---|---|---|---|
| cs.AI | 15923 | 176.9 | 5008 | 0.315 |
| cs.LG | 14196 | 157.7 | 4140 | 0.292 |
| cs.CV | 10044 | 111.6 | 4410 | 0.439 |
| cs.CL | 4146 | 46.1 | 1128 | 0.272 |
| cs.RO | 1618 | 18.0 | 814 | 0.503 |
| stat.ML | 1560 | 17.3 | 414 | 0.265 |
| cs.CR | 1357 | 15.1 | 369 | 0.272 |
| cs.CY | 1350 | 15.0 | 1350 | 1.0 |
| cs.SE | 888 | 9.9 | 251 | 0.283 |
| cs.HC | 848 | 9.4 | 432 | 0.509 |
| stat.AP | 730 | 8.1 | 730 | 1.0 |
| cs.IR | 691 | 7.7 | 190 | 0.275 |
| cs.MA | 685 | 7.6 | 283 | 0.413 |
| cs.SD | 510 | 5.7 | 84 | 0.165 |
| stat.ME | 504 | 5.6 | 325 | 0.645 |

**Gate recall check** — 200 items drawn from 20867 rejects; **3** scored above the selection threshold (limit 3). Verdict: **GATE_OK**.

Highest-scoring rejected items:

| score | title |
|---|---|
| 0.628 | What Color is the Sky (for a non-human) ? |
| 0.483 | Honey, I Shrunk the Arc de Triomphe! |
| 0.41 | CHM-Net: Center Heatmap-driven Macro-Micro Modeling Network for MRI-based Microbial Density Stratification |

## Quiet-day threshold (PRD §5.6, Q3)

**PROVISIONAL — Q3 is not settled.** A threshold was found that lands in the target band, but landing in the band is not the same as the threshold meaning something:

- 63% of daily top scores share one value — the threshold splits a tie, not a distribution
- on the arxiv path 3 of 4 weighted components are one value for 90%+ of published items (artifact_completeness, novelty, source_multiplicity) — the weights do not describe what ranks
- on the journal path 4 of 4 weighted components are one value for 90%+ of published items (artifact_completeness, novelty, relevance, source_multiplicity) — the weights do not describe what ranks

Recorded, not worked around. Moving the weights until the number looks better is where the figures would start lying. The formula is PRD §5.6's to change.

Chosen threshold **0.444**, giving a headline rate of **36.7%** across 90 days against a 30–50% target (in band). Measured on the 2157 items that would have been `published`, not on the candidate pool — a day's headline is the top card of its issue.

**What each weighted component actually does.** A component that takes one value across a path contributes nothing to ranking on that path, whatever weight it carries:

| component | weight | path | distinct values | modal value |
|---|---|---|---|---|
| `artifact_completeness` | 0.2 | arxiv | 3 | 0.0 @ 93.5% |
| `artifact_completeness` | 0.2 | journal | 3 | 0.2 @ 98.8% |
| `novelty` | 0.2 | arxiv | 6 | 0.0 @ 97.2% |
| `novelty` | 0.2 | journal | 6 | 0.0 @ 95.9% |
| `relevance` | 0.4 | arxiv | 945 | 0.7704 @ 0.3% |
| `relevance` | 0.4 | journal | 1 | 1.0 @ 100.0% |
| `source_multiplicity` | 0.2 | arxiv | 1 | 0.0 @ 100.0% |
| `source_multiplicity` | 0.2 | journal | 2 | 0.0 @ 99.6% |

The threshold is enumerated rather than estimated. The daily top scores take only 18 distinct values, because every day publishes a whitelist journal article and those score identically; a quantile lands inside that tie and reports a 100.0% rate at threshold 0.44. Each distinct top is tried instead and the measured rate closest to the band's middle wins.

**The novelty term dies.** The overlay vocabulary is a closed list, so once the archive has seen it the term goes to zero and stays there. Mean novelty of published items, by month of the backfill:

| month | mean novelty | items |
|---|---|---|
| 2026-05 | 0.1204 | 432 |
| 2026-06 | 0.0049 | 720 |
| 2026-07 | 0.0021 | 741 |
| 2026-08 | 0.0 | 264 |

In steady state a whitelist journal article scores a flat 0.44 and only an arXiv item carrying code or data links can lift a day above it. Whether a term that saturates in two weeks belongs in the headline formula is PRD §5.6's question — recorded here, not decided here.

**The live rate will not match this yet.** 5 of 5 published days currently carry a headline. The threshold was calibrated against an archive 1989 items deep; `content/` holds 36 behind those days, so almost every tag is still fresh — mean novelty 0.8534 live against 0.0 in the replay, worth 0.1707 on the headline score. The LLM tags the backfill lacks account for only 0.0351 of that. It decays on its own as days accumulate; it is the archive being young, not the threshold being wrong.

Headline-score quantiles over the published backfill items:

| quantile | score |
|---|---|
| 0.1 | 0.2187 |
| 0.25 | 0.2924 |
| 0.5 | 0.44 |
| 0.75 | 0.44 |
| 0.9 | 0.44 |
| 0.95 | 0.44 |
| 0.99 | 0.64 |

Distribution (bucket → count):

```
  0.1 # 22
 0.15 ####### 130
  0.2 ########## 187
 0.25 ############ 223
  0.3 ############ 229
 0.35 ######### 183
  0.4 ############################################################ 1100
 0.45 # 12
  0.5 # 34
 0.55 # 6
  0.6 # 30
  0.7 # 1
```

Current `config/scoring.yaml` threshold: 0.444 (source: backfill).

## Cost, measured (PRD §1, §9)

| item | value |
|---|---|
| days of runs | 5 |
| items published | 120 |
| items summarised | 120 |
| LLM | $0.5284 |
| OpenAlex | $0.0036 |
| embeddings (local) | $0.0 |
| total | $0.532 |
| per published item | $0.00443 |
| monthly estimate | $3.192 |
| tokens in / out | 485268 / 70700 |

Embeddings are local (`BAAI/bge-base-en-v1.5` on CPU), so their marginal cost is zero — which is what makes backfills and retraining free.

## Q1b labels (roadmap §2.3)

No labels yet. `uc review --label relevance --date …` collects them; `uc labels` summarises them.

## What actually gets published

Of 156 published items, **63 came from arXiv** and 93 from whitelist journals.

The split is enforced: `classifier.arxiv_min_share` is None. Without it the daily list fills with journal articles, because the classifier was trained on those journals and scores their articles ~0.99 close to by construction. Measured on 2026-08-11 with the quota disabled: 23 of 24 slots were journal articles.

## Archive

| thing | count |
|---|---|
| items | 156 |
| items with a summary | 120 |
| issues | 5 |
| quiet days | 0 |
| items with an OpenAlex ID | 138 |
| items with referenced_works | 92 |
| published (journal) items | 96 |

## Runs

| date | fetched | after gate | selected | summarised | published | skipped / failed |
|---|---|---|---|---|---|---|
| 2026-08-05 | 406 | 224 | 24 | 24 | 24 | - |
| 2026-08-06 | 378 | 198 | 24 | 24 | 24 | - |
| 2026-08-07 | 381 | 209 | 24 | 24 | 24 | - |
| 2026-08-10 | 449 | 230 | 24 | 24 | 24 | - |
| 2026-08-11 | 311 | 167 | 24 | 24 | 24 | - |

## What this report does not know

- **Q1b precision@10** — needs `uc review --label relevance` over 5 days × 30 items. This is the number that decides whether the filter is usable in practice; the holdout AUC does not answer it.
- **Q4 review time** — needs `uc review` run by a human. The CLI records it automatically; nothing else can.

---

Regenerate with `uc report`. Sources: `runs/*/metrics.json`, `models/clf-*.json`, `runs/backfill/`, `runs/labels/`, `content/`.
