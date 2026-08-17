# Operations — Urban Currents Phase 0

What a day costs a human, where a human is required, and what to do when a stage
fails.

**The pipeline can now run itself, and does not.** `uc daily` is one command
that collects a window, decides an outcome, publishes, and sends; the workflow
files that would call it every morning are committed with their `schedule:`
blocks commented out. Turning them on is the checklist below.

## The first command after being away

```bash
uv run uc status
```

Answers "is anything wrong" before "what happened": the last successful run, the
dates with no issue, cumulative spend, and the window the next run will cover.
Exits non-zero when there are unpublished dates, so it also works as a check.

`last_success` and `last_issue` are different facts and both are reported. Every
issue published before the outcome model existed has no run-log row, so a null
`last_success` beside eight published issues means "the log starts later", not
"nothing has ever worked".

```bash
uv run uc catch-up     # retry the missed dates still inside the horizon
uv run uc weekly       # seven days of outcomes, spend and sends
```

## Where a human is required

| # | Task | Frequency | Budget | Command |
|---|---|---|---|---|
| 1 | Review the journal whitelist | once, then when re-generated | ~20 min | edit `vocab/sources/journals.yaml` |
| 2 | Curate bootstrapped vocabulary candidates | once, then occasionally | ~30 min | edit `vocab/methods.yaml`, `vocab/data.yaml` |
| 3 | Daily review of the issue | daily | **≤ 15 min (Q4)** | `uc review --date YYYY-MM-DD` |
| 4 | Relevance labelling for Q1 | 5 days × 30 items | ~15 min/day | `uc review --label relevance --date …` |
| 5 | Drain `unmatched.jsonl` into the vocabulary | weekly | ~10 min | read `runs/*/unmatched.jsonl` |
| 6 | Re-calibrate the quiet-day threshold | after a backfill | ~2 min | `uc calibrate --apply` |

Tasks 1, 2 and 5 are the ones that decay if skipped: the whitelist drives the
training set, and the vocabulary drives every overlay tag. Task 3 is the one the
whole Phase 0 measurement rests on — see Q4.

**The 15-minute review budget is a measurement, not a promise.** `uc review`
records elapsed time into `runs/{run_id}/metrics.json` because self-reported
review times are always under-reported.

## Daily run

```bash
uv run uc daily                          # collect the window, decide, publish, send
uv run uc daily --dry-run                # everything, writing and sending nothing
uv run uc review --date 2026-08-14       # human checkpoint (opens the preview)
```

`uc daily` picks its own window. The issue is dated by **when we first saw the
papers**, not when they appeared: journal indexing runs p50 1 day and p90 2 days
behind publication (measured over 4,674 stored responses —
`scripts/indexing_lag.py`), and **arXiv's `submittedDate` index is three days
behind** — asked on 2026-08-18 it returned 0 for each of the previous three days
and 221–453 per day from D-4 back, weekends included
(`scripts/arxiv_visibility.py`).

So a run covers `[today-7, today-1]`, wide enough for the slower of the two.
The tail beyond that is picked up by later runs, because an already-published
item is skipped rather than published twice.

It exits **1** when the day was `not_published`, **75** when another run holds
the lock, and 0 otherwise.

**`--dry-run` still summarises, and therefore still costs.** It runs every
stage and writes nothing — measured at $0.14 and about 7 minutes for a full
7-day window. A dry run that skipped the expensive stage would not be testing
the thing most likely to break.

A dry run also leaves **no row in `content/runs_log/`**. The log answers "did
this day get covered", and a rehearsal's answer is no however much work it did;
the record of it is in `runs/{run_id}/metrics.json`.

### The three outcomes

| outcome | meaning | issue file | email |
|---|---|---|---|
| `published` | we looked, and there was something | written | sent |
| `quiet` | **we looked**, and there was almost nothing | written | sent |
| `not_published` | **we did not look** | none | none, alert instead |

**A failed day is not a quiet day.** `quiet` requires every required source to
have finished OK, a candidate population actually counted (zero is a count), no
failed stage, and the budget intact. Anything less writes no issue and logs why
in `content/runs_log/YYYY-MM-DD.json`. See `pipeline/outcome.py`.

### Stage by stage

Still supported, and still the point of the design — re-running `summarize`
never requires re-collecting:

Or stage by stage, which is the point of the design — re-running `summarize`
never requires re-collecting:

```bash
uv run uc collect   --date 2026-08-14
uv run uc dedup     --date 2026-08-14
uv run uc gate      --date 2026-08-14
uv run uc classify  --date 2026-08-14
uv run uc select    --date 2026-08-14
uv run uc link      --date 2026-08-14
uv run uc summarize --date 2026-08-14
uv run uc score     --date 2026-08-14
uv run uc issue     --date 2026-08-14
uv run uc preview   --date 2026-08-14 --open
```

Useful flags: `--fixture` (built-in sample papers, no network), `--no-llm` (skip
the API call), `--limit N` (cap items summarised this run).

## One-time setup

```bash
uv sync --extra embed                                   # includes torch; large
cp .env.example .env                                    # then fill in the keys
uv run python scripts/build_journal_whitelist.py        # → vocab/sources/journals.yaml
#   ... review the `# REVIEW:` entries by hand ...
uv run python scripts/build_trainset.py                 # → runs/trainset/
uv run python scripts/train_classifier.py               # → models/clf-{date}.*
uv run python scripts/bootstrap_vocab.py                # → vocabulary candidates
uv run uc backfill --days 90                            # → runs/backfill/ (no LLM)
uv run uc calibrate --apply                             # → config/scoring.yaml
uv run uc gate-recall                                   # → runs/gate_recall.json
```

## Turning the schedule on

`.github/workflows/daily.yml` and `weekly.yml` are committed with
`workflow_dispatch` only; their `schedule:` blocks are commented out. The
comparison behind choosing GitHub Actions is in `docs/scheduler-options.md`.

**The order matters.** It is arranged so that nothing can reach a stranger
before a human has read what it would have said. Do not skip ahead to step 6.

1. **Run it by hand, dry.** Actions → daily → Run workflow, `dry_run: true`.
   Nothing is written, nothing is sent. Confirms the install, the model cache
   and the keys.
2. **Run it by hand, live, with the file backend.** `dry_run: false`. An issue
   is written and committed; the mail is written to a `.eml` inside the runner
   and thrown away with it. Check the commit and `uc status`.
3. **Set `UC_ALERT_RECIPIENT`** in repository secrets. Failure alerts start
   working. Nothing reaches readers yet — alerts are operational mail and go
   only to that address.
4. **Uncomment `schedule:` in `daily.yml`.** It now runs itself, publishes to
   the archive, and mails nobody. Leave it here for a week and read the
   archive each morning as a stranger would.
5. **Uncomment `schedule:` in `weekly.yml`.** One summary mail a week to the
   operator. Confirms the mail path end to end with an audience of one.
6. **Only then**: pick a provider (`docs/email-delivery-options.md`), buy the
   domain, set up SPF/DKIM/DMARC, fill in `UC_SMTP_*`, and change
   `deliver.backend` to `smtp`. **This is the step that can reach someone who
   did not ask.**

Two things to know before step 4:

- **GitHub disables scheduled workflows after 60 days without repository
  activity.** Whether the workflow's own commits reset that clock is not
  something the documentation makes clear, so treat a missing weekly summary as
  a possible symptom of it. `uc status` shows a stale `last_success` either way.
- **The scheduler is best-effort and runs late under load.** That costs nothing
  here — the window is three days wide and `uc catch-up` retries for a week.

### If the bot cannot push

`content/` is committed by the workflow. When someone has pushed while the run
was working, the job rebases and retries **once**; a second failure stops the
job and alerts, because two conflicts in a row means a real one, and a conflict
in published content is a thing a person must look at.

**Never force push.** Discarding someone else's commit to make the bot's push
succeed is the one failure mode here that running again cannot undo.

## Failure handling

Every stage records `OK` / `SKIPPED` / `PARTIAL` / `FAILED` in
`runs/{run_id}/metrics.json` under `stages`, with the reason in `errors`.
**A failing stage never stops the run** — a partial issue beats no issue.

| Symptom | Meaning | Action |
|---|---|---|
| `collect.openalex: SKIPPED` | `OPENALEX_KEY` missing | the arXiv side still runs; add the key and re-run `uc collect` |
| `summarize: SKIPPED` | `ANTHROPIC_API_KEY` missing | cards publish without the two-layer summary; add the key and re-run `uc summarize` |
| `summarize: PARTIAL` | hit a call cap | raise `llm.max_summaries_per_run`, or accept it and re-run tomorrow |
| `classify.model: heuristic-v0` | no trained model found | train one; until then relevance scores are keyword density, not probabilities |
| `enrich` finds nothing | OpenAlex has not indexed the preprint yet | normal. The retry queue in `runs/state/openalex_enrich_pending.json` tries again on later days |
| `BudgetExceeded` in errors | 80% of the OpenAlex daily budget | stop for the day; the budget resets at midnight UTC |
| An item shows "Summary pending review." | LLM output violated the schema twice | `review.status` is `pending`; fix by hand in review or re-run summarize after a prompt change |
| `uc daily` exits 75 | another run holds the lock | wait. A lock whose owner is dead is reclaimed automatically; one held by a live process refuses on purpose |
| `status: not_published` | **we could not see the day** | `reasons` in `content/runs_log/` names which of the four conditions failed. `uc catch-up` retries it |
| an alert arrives with `alert_failed` in the run log | the mail could not go out | the run log is still correct — the alert is a copy of it, never the record |
| `uc status` shows `last_success: null` with issues in the archive | those issues predate the outcome model | expected. `last_issue` is the other half of the answer |
| `silent_sources: ["collect.arxiv"]` | a source finished **OK** and returned nothing across the whole window | **the failure that reports success.** Check `daily.lookback_days` against the source's indexing lag, then the source itself. It does not block publication — the other source's papers are real — so nothing else will tell you |

## Cost control

- **LLM.** Every call is cached at `runs/cache/{prompt_version}/{work_key}.json`.
  Re-running summarize costs nothing unless the prompt version changed. Two hard
  caps live in `config/pipeline.yaml`: `llm.max_summaries_per_run` and
  `llm.max_summaries_total` (cumulative, tracked in
  `runs/state/llm_usage.json`).
- **Editing a prompt requires bumping `llm.prompt_version`**, otherwise the cache
  serves stale responses generated by the old prompt.
- **OpenAlex.** `meta.cost_usd` is accumulated per response; a stage stops at 80%
  of `openalex.daily_budget_usd`. A free key is $1/day. Enrichment uses only free
  DOI singleton lookups by default; `--deep` enables title search, which costs
  search rates and was measured at ~$0.09/day for a low hit rate.
- **Embeddings are local and free.** That is what makes re-running the backfill
  and retraining the classifier a non-decision.
- **The backfill never summarises.** If it ever does, that is a bug.

## What must never happen

- Hand-editing anything under `content/`. It is pipeline output. Use `uc review`.
- Committing `.env`, or printing a key into a log, report, or metrics file.
- A test that reaches a real API.
- Renaming an OpenAlex-derived field (`referenced_works`, `topics`,
  `primary_location`, `cited_by_count`) — Phase 1 depends on those names.
- Changing a `work_key` after it has been assigned.

## Files worth knowing

| Path | What |
|---|---|
| `content/items/{work_key}.json` | one paper, permanent and mutable |
| `content/issues/YYYY-MM-DD.json` | one edition, immutable once published |
| `content/runs_log/YYYY-MM-DD.json` | **what the run concluded, including the days with no issue** |
| `content/deliveries/YYYY-MM-DD.json` | what was sent, to how many, and the hash of the body |
| `content/_retired/` | wrong-but-kept files. Validated, read by no aggregate |
| `content/entities/{facet}/{id}.json` | tag nodes, derived from items |
| `content/graph/edges.jsonl` | derived edges, `uc graph` |
| `runs/{run_id}/raw/` | verbatim API responses |
| `runs/{run_id}/metrics.json` | counts, costs, timings, stage statuses |
| `runs/{run_id}/stages/*.jsonl` | per-stage output, the resume points |
| `runs/{run_id}/unmatched.jsonl` | overlay candidates with no vocabulary entry |
| `runs/cache/` | LLM response cache |
| `runs/labels/relevance.jsonl` | keep/drop labels behind Q1 |
| `runs/state/` | LLM usage total, OpenAlex enrichment queue |
| `models/clf-{date}.json` | classifier metrics and training metadata |
