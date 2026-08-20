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
`last_success` beside a populated archive means "the log starts later", not
"nothing has ever worked".

```bash
uv run uc review --pending   # judge what was held while you were away
uv run uc catch-up           # retry the missed dates still inside the horizon
uv run uc weekly             # seven days of outcomes, spend and sends
```

## Reviewing, now that nobody reviews daily

The operating assumption changed in Phase 0L. Review used to be **daily, before
publication, and over everything**; it is now **occasional, after publication,
and over a sample**. Nothing waits for a human, so the selection policy carries
the doubt the daily pass used to carry.

| command | what it is for |
|---|---|
| `uc review --pending` | the held queue, oldest first, resumable. **The command for coming back** |
| `uc review --sample --since D` | read a stratified sample of what already went out |
| `uc review --date D` | the full review of one issue, when you want to look at a specific day |
| `uc review --relabel weak` | split the pre-M1 `drop_weak` labels into method and results |

**There is deliberately no `--latest`.** An argument-free "show me today" would
be a standing invitation to check every morning, which is the habit this design
exists to remove. `--pending` asks for no date on purpose: a week away leaves a
week of held items, and remembering which dates those were is the friction being
removed.

### The held queue

When the pipeline is not sure about an item it **holds** it rather than
publishing it, and the day goes out with a hole. The hole is the intended
outcome — a slot filled with something we are unsure of is worth less than a
shorter issue, and a reader cannot tell the two apart.

Held items are **not** carried into a later issue. They are waiting for a
judgement, not owed to readers. Two kinds:

- **withheld** — it was going to be published and a rule pulled it. Costs the
  issue an item, and is offered first in `--pending`.
- **near miss** — it was never going to be published but sits close enough to
  the line that a judgement is worth having. Costs nothing.

The rules live in `pipeline/held.py` and are tuned in `config/pipeline.yaml`
under `held:`. **The queue is the labelling queue is the training set**: it puts
scarce attention exactly where the pipeline is least sure, instead of at the top
of a ranking it already gets right.

```bash
uv run python scripts/held_rate.py    # how much the rules would hold back
```

**If the withheld rate goes over 30%, the rules are too wide** — at that point
they are not a filter, they are a different editorial policy adopted by
accident. Measured 2026-08-18: **0.2917 over one day**, and 4 of the 7 withheld
were plainly urban papers arriving through an environmental-science subfield.
That is the first thing to look at.

## Where a human is required

| # | Task | Frequency | Budget | Command |
|---|---|---|---|---|
| 1 | Review the journal whitelist | once, then when re-generated | ~20 min | edit `vocab/sources/journals.yaml` |
| 2 | Curate bootstrapped vocabulary candidates | once, then occasionally | ~30 min | edit `vocab/methods.yaml`, `vocab/data.yaml` |
| 3 | Judge the held queue | **when you come back**, not daily | ~10 min a sitting | `uc review --pending` |
| 4 | Relevance labelling for Q1 | 5 days × 30 items | ~15 min/day | `uc review --label relevance --date …` |
| 5 | Drain `unmatched.jsonl` into the vocabulary | weekly | ~10 min | read `runs/*/unmatched.jsonl` |
| 6 | Re-calibrate the quiet-day threshold | after a backfill | ~2 min | `uc calibrate --apply` |

Tasks 1, 2 and 5 are the ones that decay if skipped: the whitelist drives the
training set, and the vocabulary drives every overlay tag.

**Task 3 is no longer daily.** Q4 was redefined in Phase 0L: it used to ask
whether a day could be reviewed in fifteen minutes, and now asks whether the
thing can run for a week unattended without publishing something the editor
would retract. Nothing blocks on a human — the held queue absorbs the doubt
instead, and gets judged whenever someone is back. See `docs/phase0-ledger.md`
for what the old definition measured (nothing: zero days carried a `review_s`).

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

**Both schedules are on.** `daily.yml` has fired at 21:00 UTC since
2026-08-19, and `weekly.yml` was enabled at Sunday 22:00 UTC in 0U — the weekly
one as much for the heartbeat as for the summary, since a repository whose
schedules GitHub has quietly disabled looks exactly like a quiet week.
`deadman.yml` watches for that at 09:00 UTC. The checklist below is kept
because it is the order to follow when turning delivery on, which has **not**
happened: `deliver.backend` is still `file`. The comparison behind choosing
GitHub Actions is in `docs/scheduler-options.md`.

**The order matters.** It is arranged so that nothing can reach a stranger
before a human has read what it would have said. Do not skip ahead to step 6.

0. **Put the keys in repository secrets.** *This step was missing and it is why
   the first real attempt failed.* Step 1 used to claim it "confirms the keys"
   while saying nowhere how they get there.

   **Settings → Secrets and variables → Actions → New repository secret**

   | secret | value | without it |
   |---|---|---|
   | `OPENALEX_KEY` | same as your local `.env` | **journal collection fails and no issue is published** |
   | `GOOGLE_API_KEY` | same as your local `.env` | cards publish with no summary |
   | `CONTACT_EMAIL` | a contact address | only used in the OpenAlex request header |
   | `SPRINGER_API_KEY` | optional | ~12 journal abstracts a day go unrecovered |

   `UC_ALERT_RECIPIENT` comes at step 3 and `UC_SMTP_*` at step 6, deliberately.
   GitHub masks a secret once saved — to change one, delete it and add it again.

1. **Run it by hand, cheap.** Actions → daily → Run workflow, `dry_run: true`
   **and `smoke: true`**. Expect **3–6 minutes**; the first run is slower
   because the 440MB embedding model is not cached yet.

   `smoke` narrows the window to two days and caps summaries at three. Without
   it, step 1 collects a full seven-day window and summarises all of it — the
   most expensive thing the pipeline does, with the result discarded. That is
   the run that was killed at 45 minutes on the first attempt.
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

### Where each step happens in GitHub

| step | where |
|---|---|
| 0 · 3 · 6 (keys) | **Settings → Secrets and variables → Actions** |
| 1 · 2 | **Actions → `daily` in the sidebar → `Run workflow`** |
| 4 · 5 | edit `.github/workflows/daily.yml` (or `weekly.yml`) → delete the `#` on the two `schedule:` lines → commit |
| 6 (config) | `deliver.backend` in `config/pipeline.yaml` |

`Run workflow` opens a small panel with `date` (blank means today) and `dry_run`
(defaults to `true`). **Step 2 is the same button with `dry_run` set to `false`.**

Two things to know before step 4:

- **GitHub disables scheduled workflows after 60 days without repository
  activity.** Whether the workflow's own commits reset that clock is not
  something the documentation makes clear, so treat a missing weekly summary as
  a possible symptom of it. `uc status` shows a stale `last_success` either way.
- **The scheduler is best-effort and runs late under load.** That costs nothing
  here — the window is seven days wide and `uc catch-up` retries for a week.

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
| `summarize: SKIPPED` | `GOOGLE_API_KEY` missing | cards publish without the two-layer summary; add the key and re-run `uc summarize`. **The key depends on `llm.provider`** — Gemini is the default and wants `GOOGLE_API_KEY`; the Anthropic path wants `ANTHROPIC_API_KEY` |
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
