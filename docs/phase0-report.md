# Urban Currents — Phase 0 report

Generated 2026-08-20T01:00:58+00:00 by `uc report`. Every figure below is computed from files in this repository; anything not measured says so.

## The four questions (PRD §1)

| Q | Question | Criterion | Measured | Verdict |
|---|---|---|---|---|
| Q1a | Is the filter usable? | holdout AUC >= 0.9 | 0.9938 | PASS |
| Q1b | Is the filter usable? | precision@10 >= 0.7 (per source) | arxiv: 0.6, journal: 0.66 | FAIL |
| Q2 | Is there enough signal for a daily? | median >= 5 items/day | arxiv: 18, journal: 51 | PASS |
| Q3 | Where does the quiet-day line go? | headline rate 30-50% | 0.367 | PROVISIONAL |
| Q4 | Does review fit the budget? | median <= 15 min/day | PENDING-HUMAN | PENDING-HUMAN |

## Relevance classifier (PRD §5.4)

Model `clf-v2-2026-08-13` (variant v2) — BAAI/bge-base-en-v1.5 embeddings (768d) into logistic regression, pinned in `classifier.model_version`.

| metric | value |
|---|---|
| evaluation AUC | 0.9938 |
| average precision | 0.9835 |
| precision @ 0.35 | 0.8522 |
| recall @ 0.35 | 0.98 |
| flagged rate @ 0.35 | 0.2875 |
| training examples | 7512 |
| evaluation examples | 800 |

**These numbers describe one task: `arxiv_urban_vs_arxiv_other`.** The evaluation set is 200 positives — "published in a whitelist journal AND has an arXiv location" — against 600 negatives drawn from other arXiv papers. Both sides are unambiguous, so this measures telling clear cases apart. The hard live cases are the borderline ones, and they are not in this set; its base rate is also far above the live one, so **live precision is lower than the table above**. This comparison answers "which variant", not "is the classifier good enough". **Q1b is the only measurement that answers the second question, and Phase 1 Go/No-Go rests on Q1b, not on this AUC.**

Journal sanity check: mean probability 0.9642 over 400 whitelist articles, 100.0% above threshold. in-sample sanity check, not a performance claim — and since N4 the journal path does not consult the classifier at all, so this bounds nothing in production.

**Threshold sweep.** The headline AUC hides the decision that actually matters. The selection threshold is set from this table, not from a default:

| threshold | precision | recall | flagged rate |
|---|---|---|---|
| 0.1 | 0.575 | 0.995 | 0.432 |
| 0.15 | 0.668 | 0.995 | 0.372 |
| 0.2 | 0.754 | 0.995 | 0.33 |
| 0.25 | 0.791 | 0.985 | 0.311 |
| 0.3 | 0.811 | 0.985 | 0.304 |
| 0.35 | 0.852 | 0.98 | 0.287 |
| 0.4 | 0.878 | 0.97 | 0.276 |
| 0.45 | 0.906 | 0.965 | 0.266 |
| 0.5 | 0.931 | 0.95 | 0.255 |
| 0.55 | 0.943 | 0.915 | 0.242 |
| 0.6 | 0.963 | 0.905 | 0.235 |
| 0.65 | 0.968 | 0.9 | 0.233 |
| 0.7 | 0.973 | 0.89 | 0.229 |
| 0.75 | 0.977 | 0.855 | 0.219 |
| 0.8 | 0.982 | 0.83 | 0.211 |
| 0.85 | 0.987 | 0.785 | 0.199 |
| 0.9 | 0.986 | 0.7 | 0.177 |

Configured selection threshold: **0.35**.

Training set (v2): 2796 journal positives + 716 arXiv-urban positives (692 by subfield, 24 strict) + 4000 negatives = 7512. Journal positives are kept even though the journal path no longer consults the classifier: dropping them costs precision (0.85 → 0.70 measured, variant v3), because they are still valid training signal for what urban research reads like. Separating the entry paths and separating the training set are different decisions.

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

Per-day `above_threshold` items over 90 days, by entry path:

| path | median/day | p25 | p75 | range |
|---|---|---|---|---|
| arxiv | 18 | 12 | 23.8 | 4–43 |
| journal | 51 | 34.5 | 64 | 11–148 |
| **both** | **72** | 49.8 | 86.8 | 22–182 |

The pooled median is not comparable with the arXiv-only figure this report carried before journals entered the backfill. A whitelist article clears by membership, so the journal row measures how many whitelist articles appeared, not how many cleared a judgement — the arXiv row is the one that answers "is there enough signal".

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

**The live rate will not match this yet.** 40 of 63 published days currently carry a headline. The threshold was calibrated against an archive 1989 items deep; `content/` holds 36 behind those days, so almost every tag is still fresh — mean novelty 0.8534 live against 0.0 in the replay, worth 0.1707 on the headline score. The LLM tags the backfill lacks account for only 0.0351 of that. It decays on its own as days accumulate; it is the archive being young, not the threshold being wrong.

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
| days of runs | 66 |
| items published | 1027 |
| items summarised | 1053 |
| LLM (daily runs) | $6.0209 |
| OpenAlex (daily runs) | $0.0512 |
| embeddings (local) | $0.0 |
| total (daily runs) | $6.0721 |
| per published item | $0.00591 |
| monthly estimate | $2.76 |
| tokens in / out (all tasks) | 4352655 / 597222 |

**Per task, cumulative** — every LLM call ever made from this repository, including calls outside a daily run (labelling preparation, re-runs against a cold cache). The daily-run figures above are a subset of this, which is why they are smaller:

| task | calls | cost |
|---|---|---|
| extract | 1451 | $2.853387 |
| headline | 47 | $0.074923 |
| summarize | 1267 | $4.508542 |
| synthesis | 166 | $0.630303 |
| **total** | 2933 | **$8.074441** |

Tokens: 3017025 in, 406906 out, 0 thinking. Summarize and extract run one call each per item (D8 was reverted in N1), so a per-item token figure divided by the published count describes neither task on its own.

Embeddings are local (`BAAI/bge-base-en-v1.5` on CPU), so their marginal cost is zero — which is what makes backfills and retraining free.

## Q1b labels (roadmap §2.3)

148 labels over 5 day(s). 100% of labelled items had a summary on screen.

| source | labels | days | precision@10 | keep rate | drop: not urban | drop: not our kind | drop: weak |
|---|---|---|---|---|---|---|---|
| arxiv | 73 | 5 | 0.6 | 0.548 | 21 | 7 | 5 |
| journal | 75 | 5 | 0.66 | 0.587 | 3 | 18 | 10 |

**The two drop reasons point at different problems.** *not urban* is a classifier error. *not our kind* is a coverage question nothing in the pipeline answers yet — it is the training signal for the classifier that will replace the journal path's placeholder ranking.

| path | daily slots | depth holding 0.7 | precision by depth |
|---|---|---|---|
| arxiv | 12 | 8 | @1: 1.0, @4: 0.85, @8: 0.7, @12: 0.6, @15: 0.5333 |
| journal | 12 | 8 | @1: 0.8, @4: 0.7, @8: 0.75, @12: 0.6167, @15: 0.5867 |

Where the depth holding 0.7 is below the slot count, the path is being asked for more items than it has good ones — the fix is the slot split or a better ranker, not a higher threshold. Raising the arXiv threshold does not help: at 0.7 the 90-day backfill yields a median of 6 arXiv candidates a day and at 0.8 it yields 3, so the path could not fill 12 slots at any precision.

## What actually gets published

Across 63 issues, **1037 items were `published`** — 280 from arXiv and 757 from whitelist journals.

`content/items/` holds 2274 files, 1237 more than the issues reference. Those are items an earlier selection rule published and the current one does not; they are still part of the archive novelty is measured against, which is why the difference is counted rather than rounded away.

The split is structural, not a quota. Each entry path owns its slots — journal 12, arXiv 12 — and a path that cannot fill its own lends them to the other, which the run records. The earlier `classifier.arxiv_min_share` quota is gone (N4): it was treating a symptom, since a whitelist article scores ~0.99 nearly by construction and the classifier could not rank within that path at all. Measured on 2026-08-11 under the old single-classifier design: 23 of 24 slots went to journal articles.

## What we could not read

**1226 items across 63 issues had no abstract from any source** and published in `Also published today` instead of as cards — 1226/2263 of everything that reached an issue. Springer Nature withdrew its non-OA abstracts from OpenAlex in 2022 and Elsevier followed in 2024; Crossref and Springer's own API are asked for what they can still supply, and what none of them has cannot be summarised, because the abstract is the only evidence a summary is allowed to use.

| publisher | `unreadable` items |
|---|---|
| Elsevier | 1080 |
| Springer | 126 |
| Springer Nature | 37 |
| Taylor & Francis | 37 |
| Sage | 21 |
| Copernicus | 18 |
| Wiley | 8 |
| ASCE | 2 |
| 10.1215 | 2 |
| Taylor & Francis (Routledge) | 2 |
| 10.11361 | 1 |
| 10.12813 | 1 |

This is the one blind spot the pipeline can measure exactly, and the count is stated rather than hidden. It names publishers here because this is the engineering report; the reader-facing section names none.

## Archive

| thing | count |
|---|---|
| items | 2274 |
| items with a summary | 1046 |
| issues | 63 |
| quiet days | 23 |
| items with an OpenAlex ID | 2146 |
| items with referenced_works | 1873 |
| published (journal) items | 2030 |

## Runs

| date | fetched | after gate | selected | summarised | published | skipped / failed |
|---|---|---|---|---|---|---|
| 2026-06-12 | 368 | 183 | 18 | 18 | 18 | enrich.springer |
| 2026-06-13 | 207 | 109 | 15 | 15 | 15 | enrich.springer |
| 2026-06-14 | 241 | 127 | 16 | 16 | 16 | enrich.springer |
| 2026-06-15 | 470 | 256 | 18 | 18 | 18 | enrich.springer |
| 2026-06-16 | 435 | 215 | 19 | 19 | 19 | enrich.springer |
| 2026-06-17 | 409 | 204 | 17 | 17 | 17 | enrich.springer |
| 2026-06-18 | 440 | 209 | 21 | 21 | 21 | enrich.springer |
| 2026-06-19 | 308 | 158 | 20 | 20 | 20 | enrich.springer |
| 2026-06-20 | 174 | 87 | 9 | 9 | 9 | enrich.springer |
| 2026-06-21 | 182 | 71 | 13 | 13 | 13 | enrich.springer |
| 2026-06-22 | 471 | 240 | 19 | 19 | 19 | enrich.springer |
| 2026-06-23 | 428 | 221 | 18 | 18 | 18 | enrich.springer |
| 2026-06-24 | 379 | 215 | 20 | 20 | 20 | enrich.springer |
| 2026-06-25 | 377 | 182 | 16 | 16 | 16 | enrich.springer |
| 2026-06-26 | 341 | 187 | 18 | 18 | 18 | enrich.springer |
| 2026-06-27 | 185 | 99 | 13 | 13 | 13 | enrich.springer |
| 2026-06-28 | 233 | 102 | 11 | 11 | 11 | enrich.springer |
| 2026-06-29 | 483 | 272 | 19 | 19 | 19 | enrich.springer |
| 2026-06-30 | 485 | 284 | 20 | 20 | 20 | enrich.springer |
| 2026-07-01 | 435 | 300 | 19 | 19 | 19 | enrich.springer |
| 2026-07-02 | 453 | 251 | 15 | 15 | 15 | enrich.springer |
| 2026-07-03 | 301 | 154 | 19 | 19 | 19 | enrich.springer |
| 2026-07-04 | 172 | 88 | 15 | 15 | 15 | enrich.springer |
| 2026-07-05 | 201 | 90 | 17 | 17 | 17 | enrich.springer |
| 2026-07-06 | 405 | 209 | 15 | 15 | 15 | enrich.springer |
| 2026-07-07 | 385 | 226 | 18 | 18 | 18 | enrich.springer |
| 2026-07-08 | 335 | 200 | 18 | 18 | 18 | enrich.springer |
| 2026-07-09 | 330 | 183 | 19 | 19 | 19 | enrich.springer |
| 2026-07-10 | 302 | 160 | 13 | 13 | 13 | enrich.springer |
| 2026-07-11 | 179 | 89 | 12 | 12 | 12 | enrich.springer |
| 2026-07-12 | 176 | 84 | 11 | 11 | 11 | - |
| 2026-07-13 | 380 | 225 | 16 | 16 | 16 | enrich.springer |
| 2026-07-14 | 333 | 196 | 15 | 15 | 15 | enrich.springer |
| 2026-07-15 | 347 | 184 | 17 | 17 | 17 | enrich.springer |
| 2026-07-16 | 324 | 195 | 15 | 15 | 15 | enrich.springer |
| 2026-07-17 | 285 | 162 | 16 | 16 | 16 | enrich.springer |
| 2026-07-18 | 146 | 90 | 17 | 17 | 17 | enrich.springer |
| 2026-07-19 | 136 | 83 | 12 | 12 | 12 | enrich.springer |
| 2026-07-20 | 363 | 178 | 13 | 13 | 13 | enrich.springer |
| 2026-07-21 | 321 | 171 | 13 | 13 | 13 | enrich.springer |
| 2026-07-22 | 328 | 179 | 17 | 17 | 17 | enrich.springer |
| 2026-07-23 | 352 | 182 | 13 | 13 | 13 | enrich.springer |
| 2026-07-24 | 277 | 193 | 19 | 19 | 19 | enrich.springer |
| 2026-07-25 | 167 | 99 | 13 | 13 | 13 | enrich.springer |
| 2026-07-26 | 188 | 98 | 15 | 15 | 15 | enrich.springer |
| 2026-07-27 | 366 | 207 | 20 | 20 | 20 | enrich.springer |
| 2026-07-28 | 369 | 206 | 16 | 16 | 16 | enrich.springer |
| 2026-07-29 | 363 | 197 | 14 | 14 | 14 | enrich.springer |
| 2026-07-30 | 489 | 242 | 16 | 16 | 16 | enrich.springer |
| 2026-07-31 | 344 | 170 | 17 | 17 | 17 | enrich.springer |
| 2026-08-01 | 203 | 173 | 15 | 15 | 15 | enrich.springer |
| 2026-08-02 | 256 | 118 | 12 | 12 | 12 | enrich.springer |
| 2026-08-03 | 518 | 273 | 17 | 17 | 17 | enrich.springer |
| 2026-08-04 | 508 | 260 | 16 | 16 | 16 | enrich.springer |
| 2026-08-05 | 406 | 224 | 24 | 24 | 24 | enrich.springer |
| 2026-08-06 | 381 | 200 | 24 | 24 | 24 | enrich.springer |
| 2026-08-07 | 384 | 210 | 24 | 24 | 24 | enrich.springer |
| 2026-08-08 | 221 | 113 | 16 | 16 | 16 | enrich.springer |
| 2026-08-09 | 228 | 164 | 11 | 11 | 11 | enrich.springer |
| 2026-08-10 | 452 | 232 | 24 | 23 | 24 | enrich.springer |
| 2026-08-11 | 410 | 212 | 18 | 18 | 24 | enrich.springer |
| 2026-08-13 | 0 | 0 | 0 | 0 | 0 | - |
| 2026-08-14 | 0 | 0 | 0 | 0 | 0 | - |
| 2026-08-18 | 2176 | 1101 | 16 | 0 | 0 | enrich.springer, summarize |
| 2026-08-19 | 2098 | 1094 | 24 | 24 | 10 | enrich.springer |
| 2026-08-20 | 1731 | 974 | 24 | 24 | 5 | enrich.springer |

## What this report does not know

- **Q4 review time** — needs `uc review` run by a human. The CLI records it automatically; nothing else can.

---

Regenerate with `uc report`. Sources: `runs/*/metrics.json`, `models/clf-*.json`, `runs/backfill/`, `runs/labels/`, `content/`.
