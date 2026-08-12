# Urban Currents — Phase 0 report

Generated 2026-08-12T14:27:31+00:00 by `uc report`. Every figure below is computed from files in this repository; anything not measured says so.

## The four questions (PRD §1)

| Q | Question | Criterion | Measured | Verdict |
|---|---|---|---|---|
| Q1a | Is the filter usable? | holdout AUC >= 0.9 | 0.9761 | PASS |
| Q1b | Is the filter usable? | precision@10 >= 0.7 | PENDING-HUMAN | PENDING-HUMAN |
| Q2 | Is there enough signal for a daily? | median >= 5 items/day | not measured | not measured |
| Q3 | Where does the quiet-day line go? | headline rate 30-50% | not measured | not measured |
| Q4 | Does review fit the budget? | median <= 15 min/day | PENDING-HUMAN | PENDING-HUMAN |

> Q1b and Q4 require a human at the keyboard: `uc review --label relevance` for precision@10, `uc review` for the timing. They are marked PENDING-HUMAN rather than guessed.

## Relevance classifier (PRD §5.4)

Model `clf-2026-08-12` — BAAI/bge-base-en-v1.5 embeddings (768d) into logistic regression.

| metric | value |
|---|---|
| holdout AUC | 0.9761 |
| average precision | 0.9815 |
| precision @ 0.5 | 0.953 |
| recall @ 0.5 | 0.9103 |
| training examples | 7897 |
| holdout examples | 1580 |

**Per-source behaviour on the holdout.** The predicted failure mode is that journal-heavy training scores arXiv urban papers too low (PRD §5.4, §10), so it is measured directly:

| source | n | mean probability | recall / FPR |
|---|---|---|---|
| arxiv_other | 800 | 0.1242 | FPR 0.0437 |
| arxiv_urban | 211 | 0.6765 | 0.6825 |
| journal | 569 | 0.9544 | 0.9947 |

Training set: 2800 journal positives + 1097 arXiv-urban positives + 4000 negatives = 7897. The arXiv share of positives is 0.3 by design — journal prose and arXiv prose differ, and training on journals alone down-scores exactly the arXiv urban computing papers this product exists to find.

## Volume and gate (PRD §5.3, Q2)

**arXiv intake by category** (counted across every collected item; cross-listed papers appear under each of their categories):

| category | items |
|---|---|
| cs.AI | 5 |
| cs.CV | 4 |
| cs.CY | 3 |
| cs.SI | 2 |
| cs.LO | 1 |
| cs.LG | 1 |
| cs.NI | 1 |

**Gate recall check**: not run. `uc gate-recall` after a backfill.

## Quiet-day threshold (PRD §5.6, Q3)

Not calibrated. Run `uc backfill --days 90` then `uc calibrate --apply`.
## Cost, measured (PRD §1, §9)

| item | value |
|---|---|
| days of runs | 1 |
| items published | 24 |
| items summarised | 0 |
| LLM | $0.0 |
| OpenAlex | $0.0003 |
| embeddings (local) | $0.0 |
| total | $0.0003 |
| per published item | $1e-05 |
| monthly estimate | $0.009 |
| tokens in / out | 0 / 0 |

Embeddings are local (`BAAI/bge-base-en-v1.5` on CPU), so their marginal cost is zero — which is what makes backfills and retraining free.

## Archive

| thing | count |
|---|---|
| items | 26 |
| items with a summary | 0 |
| issues | 1 |
| quiet days | 0 |
| items with an OpenAlex ID | 14 |
| items with referenced_works | 10 |
| published (journal) items | 14 |

## Runs

| date | fetched | after gate | selected | summarised | published | skipped / failed |
|---|---|---|---|---|---|---|
| 2026-08-11 | 311 | 167 | 24 | 0 | 24 | - |

---

Regenerate with `uc report`. Sources: `runs/*/metrics.json`, `models/clf-*.json`, `runs/backfill/`, `runs/labels/`, `content/`.
