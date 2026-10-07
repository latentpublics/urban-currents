# Operations — Urban Currents Phase 0

What a day costs a human, where a human is required, and what to do when a stage
fails.

**The pipeline runs itself, publishes to a real address, and mails one
operator.** `uc daily` collects a window,
decides an outcome, publishes and sends; `daily.yml` has called it at 21:00 UTC
since 2026-08-19, `weekly.yml` since 0U, and `deadman.yml` watches for the case
where neither fires. Since 1P (2026-10-05) `deliver.backend` is `smtp` through
Resend, and that reaches **`UC_ALERT_RECIPIENT` only** — failure alerts, the
third-day silence mail and the weekly summary. The issue itself still goes to
nobody: it is sent to `UC_PREVIEW_RECIPIENT`, which does not exist. Until 1P
everything was `file`, written into the runner and discarded with it.
`uc status` says which state you are in, on the `[ALERTS]` line, using the
backend that will actually be used rather than the one in the config — and,
if that differs, which secret is missing.

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
uv run uc missing-days       # days inside the horizon with no run-log row at all
uv run uc catch-up           # retry the missed dates still inside the horizon
uv run uc weekly             # seven days of outcomes, spend and sends
```

`missing-days` answers a question `status` cannot. `status` reads the rows in
`content/runs_log/` and tells you what they say; a day that was never attempted
has no row to read, so it appears nowhere in `status` and nowhere in the archive
either — it is simply absent. See **A day with no row** below.

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

**Judgements are the scarcest thing this project produces, and they live in
`runs/labels/`.** `uc status` prints a `[LABELS]` line if any judgement file is
outside version control, and stays silent otherwise. It exists because
`held_review.jsonl` — 122 judgements — sat untracked for three weeks while a
decision those judgements had already answered went on being cited as
unmeasured. `.gitignore` keeps `runs/labels/*.jsonl` tracked and excludes only
`*_pool.jsonl`, which are regenerable candidate lists rather than judgements.

**There is deliberately no `--latest`.** An argument-free "show me today" would
be a standing invitation to check every morning, which is the habit this design
exists to remove. `--pending` asks for no date on purpose: a week away leaves a
week of held items, and remembering which dates those were is the friction being
removed.

### The held queue

When the pipeline is not sure about an item it **files** it for a judgement, and
a rule that is set to withhold also takes it out of the day, which then goes out
with a hole. The hole is the intended outcome for a withholding rule — a slot
filled with something we are unsure of is worth less than a shorter issue, and a
reader cannot tell the two apart.

Held items are **not** carried into a later issue. They are waiting for a
judgement, not owed to readers. Two kinds:

- **withheld** — it was going to be published and a rule pulled it. Costs the
  issue an item, and is offered first in `--pending`.
- **near miss** — it was never going to be published but sits close enough to
  the line that a judgement is worth having. Costs nothing.

**As of batch 1H no rule withholds, so `withheld` is 0 and that is normal.**
`uc status` and the weekly mail will show a queue made entirely of near misses.
Two rules could withhold and neither does: `off_subfield` has an empty deny-list
(every subfield on it was overturned by targeted labels, batch 0Q), and
`at_the_floor` was switched off in 1H on an editorial call: arXiv has one floor
and 0.80 is it. Above the floor is published.

Both rules still run and still file, and `uc status` marks them `inert` — which
means "withholds nothing", not "does nothing". `at_the_floor` is the busiest
rule in the queue. That is deliberate: [0.80, 0.83) can only stop being an
unmeasured window if the items in it keep arriving for judgement. One line in
`config/pipeline.yaml` (`held.at_the_floor_withholds`, `held.off_subfield_withholds`)
turns either back on.

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
That is the first thing to look at. The check is one-sided and stays that way:
with nothing withheld the rate is 0 and says nothing at all.

## Where a human is required

| # | Task | Frequency | Budget | Command |
|---|---|---|---|---|
| 1 | Review the journal whitelist | once, then when re-generated | ~20 min | edit `vocab/sources/journals.yaml` |
| 2 | Curate bootstrapped vocabulary candidates | once, then occasionally | ~30 min | edit `vocab/methods.yaml`, `vocab/data.yaml` |
| 3 | Judge the held queue | **when you come back**, not daily | ~10 min a sitting | `uc review --pending` |
| 4 | Relevance labelling for Q1 | 5 days × 30 items | ~15 min/day | `uc review --label relevance --date …` |
| 5 | Drain `unmatched.jsonl` into the vocabulary | weekly | ~10 min | read `runs/*/unmatched.jsonl` |
| 6 | Re-calibrate the quiet-day threshold | after a backfill | ~2 min | `uc calibrate --apply` |
| 7 | **Read an alert mail** and act on it | when one arrives (since 1P) | ~5 min | the mail names the command; `uc status` |
| 8 | Set or rotate a mail secret — **values are set by a person, never by the agent** | when the provider or key changes | ~5 min | see "Mail provider" |

Tasks 1, 2 and 5 are the ones that decay if skipped: the whitelist drives the
training set, and the vocabulary drives every overlay tag.

**Task 3 is no longer daily.** Q4 was redefined in Phase 0L: it used to ask
whether a day could be reviewed in fifteen minutes, and now asks whether the
thing can run for a week unattended without publishing something the editor
would retract. Nothing blocks on a human — the held queue absorbs the doubt
instead, and gets judged whenever someone is back. See `docs/phase0-ledger.md`
for what the old definition measured (nothing: zero days carried a `review_s`).

## What a run leaves behind

`content/runs_log/YYYY-MM-DD.json` is the committed record of a run. Since
batch 1L it carries what the sources actually did, not just what the day
concluded:

| field | what it means |
|---|---|
| `silent_sources` | the source finished **OK** and returned nothing |
| `failed_sources` | **every request to that source failed** — we were blind, not quiet |
| `source_failures` | the exception type and message, per source |
| `source_observations` | HTTP status, the feed's own result total, windows tried and failed |
| `held_withheld` / `held_near_miss` | the day's held-queue tallies |

`silent_sources` and `failed_sources` are different facts and are never merged.
A dead arXiv window used to produce zero items with the stage still green, so it
was filed as silence — the source politely having nothing — when in fact we
could not see. 2026-09-07, 09-12 and 09-13 are all recorded the old way and
cannot now be told apart.

🔴 **A missing field means "this row predates 1L", not zero.** `held_withheld:
null` says the run never reached selection, or the row was written before these
fields existed; `held_withheld: 0` says it selected and held nothing. Reading
the first as the second rebuilds exactly the ambiguity these fields removed.
**Rows before 2026-09-15 have none of them.** They are not backfilled — a value
we did not record is not a value we can invent.

**An arXiv failure does not stop the day.** The issue still goes out on whatever
the other sources found, because journal articles collected on a day arXiv was
unreachable are real papers and withholding them would trade a partial issue for
none. The failure is recorded, not acted on.

The full run directory — `metrics.json`, `stages/`, the unmatched and dropped
lists — is uploaded by the daily workflow as a **90-day artifact**, named
`run-YYYY-MM-DD` on the workflow run page. It is not committed; `runs/` stays
gitignored. Raw API responses and the rendered email are deliberately excluded.

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

## The site

Built from `content/` by `uc site` and deployed by `.github/workflows/pages.yml`
to **<https://latentpublics.com/urban-currents/>**.

| | |
|---|---|
| Source | GitHub Actions (`Settings -> Pages -> Source: GitHub Actions`) |
| Custom domain | **Leave empty.** The domain belongs to `latentpublics.github.io`; setting it here would take `latentpublics.com` for this repository and push the organisation site off it |
| Trigger | `workflow_run` on a successful `daily`, plus `workflow_dispatch` |
| Indexed | **Allowed since 2026-10-06** (G5). `site.published: true` in `config/pipeline.yaml`. Allowed is not the same as indexed — see below |

A day that ends `not_published` does not deploy — `daily` exits non-zero, so
the `workflow_run` condition is false. That is intended. A *quiet* day does
publish, an empty issue and all, and appears on the site as a day we looked and
found nothing; a `not_published` day is one the pipeline could not vouch for,
and the site is an archive of what we can vouch for.

### Making it public (G5)

One line: `site.published: true` in `config/pipeline.yaml`. That turns
`robots.txt` from `Disallow: /` to `Allow: /` with a `Sitemap:` line, and drops
`noindex` from every page. Do not edit `robots.txt` or the templates by hand —
they are generated, and the switch exists so the two cannot disagree.

**Done on 2026-10-06** and deployed the same morning by dispatching `pages`.

Three things worth knowing, all measured rather than assumed:

* **Turning it on is cheaper than turning it off.** Setting the line back to
  `false` restores `noindex` on the next deploy, but a search engine's copy, its
  snapshot and links from elsewhere outlive it. Treat `false` as "stop new
  indexing", not "unpublish".
* **A commit is not a deploy.** `pages.yml` builds from `main` only when a
  `daily` run succeeds or when it is dispatched by hand
  (**Actions → `pages` → Run workflow**). After changing this line, dispatch
  `pages` and then check the live site — do not dispatch it to overlap the
  21:00 UTC `daily`.
* **The `robots.txt` under `/urban-currents/` is not the one crawlers read.**
  Crawlers read `https://latentpublics.com/robots.txt` only, which belongs to
  the organisation site and has said `Allow: /` throughout. So the sub-path
  `robots.txt` has never blocked anything, and its `Sitemap:` line will not be
  discovered from there. **What kept this site out of search was the
  `<meta name="robots" content="noindex, nofollow">` on every page**, and that
  is what the switch now removes. To have the sitemap read, submit
  `https://latentpublics.com/urban-currents/sitemap.xml` in Search Console.

To check the state from anywhere:

```
curl -sS https://latentpublics.com/urban-currents/robots.txt
curl -sS https://latentpublics.com/urban-currents/ | grep -i 'name="robots"'
curl -sSI https://latentpublics.com/urban-currents/sitemap.xml
```

Published, the first prints `Allow: /` and a `Sitemap:` line, the second
prints nothing, and the third is `200`. Not appearing in search for some days
after the switch is normal, not a fault.

## Turning the schedule on

**Both schedules are on.** `daily.yml` has fired at 21:00 UTC since
2026-08-19, and `weekly.yml` was enabled at Sunday 22:00 UTC in 0U — the weekly
one as much for the heartbeat as for the summary, since a repository whose
schedules GitHub has quietly disabled looks exactly like a quiet week.
`deadman.yml` watches for that at 09:00 UTC. The checklist below is kept
because it is the order delivery was turned on in. Steps 0–5 were done by
2026-08-19; step 6 was done for **operator mail only** in 1P — see
"Mail provider" below. Reader mail is a seventh step that has not been taken.

GitHub Actions was chosen in 0k over a laptop and a VPS, and the deciding
argument was the **shape of the failure**, not cost or convenience: a laptop
that sleeps misses the day silently, which is the state this whole phase exists
to remove, while a late Actions run is absorbed by the seven-day window and
`uc catch-up`.

That last clause was too generous and 2026-08-26 collected on it. A late
Actions run is absorbed **as long as the day it files under is the day it was
due for**, and until the 08-26 batch it was not: the run took its date from the
clock, the 21:00 UTC slot is three hours from midnight, and a three-and-a-half
hour delay produced an issue dated the next day and a gap where 08-26 should
have been. A scheduled run is now dated from its cron slot instead
(`uc slot-date`), catch-up walks the calendar so a day with no row is retried
like any other, and the deadman looks for holes as well as for staleness.

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
6. **Only then**: pick a provider — the comparison in 0k narrowed it to
   Amazon SES and Resend, and the domain question was still open — buy the
   domain, set up SPF/DKIM/DMARC, fill in `UC_SMTP_*`, and change
   `deliver.backend` to `smtp`. *Done in 1P with Resend; the values are under
   "Mail provider" below.* With `UC_PREVIEW_RECIPIENT` unset this reaches the
   operator and nobody else.
7. **Reader mail: add `UC_PREVIEW_RECIPIENT`.** Not taken. **This is the step
   that can reach someone who did not ask** — the moment that secret exists,
   the next published issue is mailed to it. It is one address; there is no
   subscriber list and nothing reads one.

### Mail provider (1P)

| | |
|---|---|
| provider | Resend, SMTP |
| host · port | `smtp.resend.com` · `587` (STARTTLS). 465/2465 are implicit TLS, which `SmtpBackend` does not speak |
| `UC_SMTP_USER` | the literal string **`resend`** — not an email address. An address here fails authentication; it is the most common mistake |
| `UC_SMTP_PASSWORD` | a Resend API key |
| sender (`deliver.sender`) | `Urban Currents <no-reply@send.latentpublics.com>` |
| verified domain | `send.latentpublics.com` — Resend refuses a From on any domain it has not verified. Its DKIM/SPF state is visible only in the Resend dashboard |

Two recipient switches, independent of the backend and of each other:

| path | recipient | if unset |
|---|---|---|
| alerts, silence mail, weekly summary | `UC_ALERT_RECIPIENT` | `no_alert_recipient`, nothing sent |
| the issue | `UC_PREVIEW_RECIPIENT` | `no_recipients`, nothing sent |

Setting a secret without leaving the value in shell history:

```
gh secret set UC_SMTP_USER     --repo latentpublics/urban-currents --body "resend"
gh secret set UC_SMTP_PASSWORD --repo latentpublics/urban-currents
```

The second line has no `--body` on purpose; `gh` prompts for the value. The
web route is **Settings → Secrets and variables → Actions → New repository
secret**.

If one of the three (`deliver.smtp.host`, `UC_SMTP_USER`, `UC_SMTP_PASSWORD`)
is missing, the backend falls back to `file` for that run and says which, by
name: a `[DELIVER]` line on stderr, `fell_back_from`/`missing` in the send
result, and a line under `[ALERTS]` in `uc status`. A missing password costs a
send, not a day — and it must not look like alerting is on.

**To check the path end to end:** Actions → `weekly` → Run workflow. It mails
the weekly summary to `UC_ALERT_RECIPIENT` and goes red if the send did not
land. Check the inbox **and the spam folder**, and the From line.

**Changing provider:** `deliver.smtp.host`/`port`, `deliver.sender` (on the new
provider's verified domain), and the two `UC_SMTP_*` secrets. Nothing else
names Resend.

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
- **The scheduler is best-effort and runs late under load.** It costs nothing
  now, and it is worth knowing why the earlier version of this sentence was
  wrong. It said the seven-day window and `uc catch-up` absorbed any lateness.
  They absorb the *collection*; they did not absorb the *date*. The 21:00 UTC
  slot is three hours from midnight, the run took its date from the clock, and
  on 2026-08-26 a three-and-a-half-hour delay published the day's issue as
  08-27 and left 08-26 with no issue and no row. A scheduled run is now dated
  from its cron slot (`uc slot-date`), so lateness has to reach the *next* slot
  before it can misfile anything.

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
| a date has **no row at all** | the day was never attempted, or was attempted under another date | `uc missing-days`. Catch-up retries it on the next daily run; past `daily.catch_up_days` it cannot be recovered. See **A day with no row** |
| an alert arrives with `alert_failed` in the run log | the mail could not go out | the run log is still correct — the alert is a copy of it, never the record |
| `uc status` shows `last_success: null` with issues in the archive | those issues predate the outcome model | expected. `last_issue` is the other half of the answer |
| `silent_sources: ["collect.arxiv"]` | a source finished **OK** and returned nothing across the whole window | **the failure that reports success.** Check `daily.lookback_days` against the source's indexing lag, then the source itself. It does not block publication — the other source's papers are real — so nothing else will tell you |

### A day with no row

Every alarm in this repository except one reads a run-log row and judges it.
The exception exists because of 2026-08-26, which has **no row** — the run that
should have carried that date published itself as 08-27 — and a day with no row
was invisible to all of them at once. `uc status` had nothing to list, the
workflow's interrupted-row net asked about the wrong day and found it already
written, `uc catch-up` built its queue from the rows that existed, and the
deadman asked how old the newest row was and got a reassuring answer.

```bash
uv run uc missing-days               # the last 7 days, ignoring the newest one
uv run uc missing-days --days 30     # wider, for looking rather than alarming
uv run uc missing-days --as-of 2026-08-27 --grace 0
```

It exits 1 when it finds a gap, which is how `deadman.yml` goes red. The
`--grace` window keeps it from firing on a day whose run is merely late:
by default the newest day it will name is one whose slot was about 36 hours
ago.

### The deadman said the pipeline was dead and it was not

2026-09-07, and it was the watchdog that was wrong. The alert read *"No run has
been recorded since 2026-09-06 (38h ago)"* while the 09-06 run had finished
normally at 22:54 UTC — fifteen hours earlier — and the 09-07 run went out
seven hours later.

Two things caused it and both are fixed. The freshness check measured a row's
age from **midnight on the row's own date**, but a run that covers a day
records at 22:50–23:26 *on* that day, so a healthy archive always read 33 hours
old at the 09:00 slot, against a 36-hour limit. Three hours of slack, for a job
whose own cron is routinely late — this repository's `daily` arrives at +2:19
every single day, and the deadman that morning was about five hours behind.

It now reads the row's own `recorded_at`, and the limit is 30 hours. A healthy
morning reads 9.5–10.5, so the job can be roughly twenty hours late before it
accuses anything; one missed 21:00 slot reads 33.6 or more, so it is still
caught at the very next 09:00 check, exactly as before. **Detection did not
move — only the false-alarm margin did.** Raising the old limit instead would
have traded the other way: on a once-a-day check the date measurement steps in
whole days, so the next threshold up would have delayed detection by 24 hours.

The arithmetic is in `.github/scripts/deadman-freshness.sh` rather than inside
the workflow, so it can be run with a fabricated clock;
`tests/test_deadman_freshness.py` replays both that morning and the real missed
day of 2026-08-21.

### A source we promise to read returned nothing

Different direction, same morning. `collect.arxiv` finished successfully and
fetched zero items, so 2026-09-07's issue covers the journals alone. Three
things are true at once and it is worth keeping them apart:

* **the day still publishes**, deliberately — the journal papers are real and
  withholding them would trade a partial issue for none;
* **it says so**: the issue page, the archive row and `silent_sources` in the
  API all name the source, and the scan line stops claiming arXiv categories it
  read nothing from;
* **nobody is mailed for one such day.** A required source silent for three
  days running gets one mail, on the third day and not again; every silent day
  appears in the weekly summary for as long as it lasts.

> **🔴 Until 2026-10-05 that mail reached nobody, and there was a stretch when
> that mattered.** `deliver.backend` was `file`, so the "third day" mail was
> written as an `.eml` into the Actions runner, and the runner was then
> destroyed. arXiv refused every request from 2026-09-17 to 09-27. The alarm
> worked: the silence was detected, the mail was decided on 09-19
> (`silent_alert.status: alert_undeliverable` in that run's log), and it went
> nowhere. An operator who read "no mail came, so it is fine" during those
> eleven days was reading the absence of a mail that could not have arrived.
> The *"Failure alerts cannot reach anyone"* line the workflow printed on every
> run is an `::error::` **annotation** — it shows red in the log, but the step
> passes, the job stays green, and GitHub mails nobody. All eleven of those
> runs finished green.
>
> **Since 1P the mail goes to `UC_ALERT_RECIPIENT` through Resend** (see "Mail
> provider"). Absence of mail now means something only while `uc status` says
> *"alerts reach a person"*; if it says *"reach nobody"*, it names what is
> missing.
>
> **What does reach a person since 1N:** if a required source has failed
> outright — every request died, `failed_sources` in the run-log row — on
> **two consecutive days**, the last step of `daily` fails the job, and GitHub
> mails a failed scheduled run to the account that last edited its `cron:`
> line (subject to that account's notification settings — not verified from
> here). The issue
> for that day has already been published and committed by then; the red cross
> is about the source, not the day. `uv run uc dead-sources --date <day>` asks
> the same question locally.
>
> Silence (a source that answered and had nothing) does **not** turn the job
> red; only failure does. A silent source reaches a person through the
> third-day mail (once per streak) and the weekly summary.

**Reading the failure.** `source_failures` in `content/runs_log/<day>.json`
carries the collector's own sentence. Since 1N the two shapes read differently:

* *"arXiv rejected the request immediately: HTTP 406 …(not retried)"* — a 4xx
  refusal. One request, no backoff, because asking again gets the same answer.
  Look at what we send, or at whether arXiv changed what it accepts.
* *"arXiv request failed after 3 attempts: …"* — a 5xx, a timeout or an empty
  body, retried with backoff. Usually arXiv under load; usually transient.

Before 1N both were written as the second, which is why the eleven days of 406
looked like a flaky server in the record. 429 has its own sentence (*"rate
limited … giving up rather than waiting past the day's budget"*) and was never
part of this.

**A 406 gets one second request, in a different shape (1Q).** The 406 came
back on 2026-10-06, the day after a 200 on the same code. A request that does
not change cannot by itself fail one day and succeed the next, so 1N's
explicit `Accept` is not *the cause* of the 406 — but whether arXiv is
refusing that request's *shape* is still open. So on a 406, and only a 406,
the collector sends one more request: the one it sent before 1N, header for
header (no `Accept` of its own, so httpx's `*/*` in httpx's position — an
explicit `Accept: */*` is not the same bytes). It is not a retry: the refused
request is never sent twice. 400, 401, 403 and 404 get no second request; 429
and 5xx behave as before. It happens at most once per collection window, after
the usual five-second gap; if the second request is refused too, the window
ends there.

Read it in `source_observations["collect.arxiv"]` of that day's run-log row:

| `accept_fallback[].fallback_status` | What it means |
|---|---|
| **200** (`recovered: true`, `accept_shape: "pre-1N"`) | arXiv refuses the `Accept` header 1N added. **The day was recovered** — arXiv papers are in that issue. Bring this to whoever owns the collector: the header should go. |
| **406** (`recovered: false`) | The request's shape is not the trigger. What is left is on arXiv's side or the path to it: our IP, our rate, a firewall in front of the API. Read `rejected_headers` and `fallback_headers` — `server`, `via`, `cf-*`, `x-*` say who answered. |
| `null` with `fallback_error` | The second request never completed. Inconclusive; wait for the next 406. |

`rejected_headers` is kept for every 4xx, not only 406. Header values are
scrubbed before they are written, `Set-Cookie` is reduced to
`set_cookie_present`, and nothing named like a credential is kept at all.
`accept_shape` is on every row from 1Q on: `"1N"` on a normal day.

Three causes look identical from here and the alert says so rather than
guessing: the source is down, our query stopped matching, or the window we ask
for sits inside the source's indexing lag. `uc run --date <day> --dry-run`
tells them apart.

**What to do about one.** Catch-up retries days with no row automatically on
the next daily run, so a gap that is still there has already survived an
attempt. Dispatch `daily` by hand from the Actions tab with that `date` while
it is still inside `daily.catch_up_days` — seven days. Past the horizon the
sources have moved on and the day cannot be recovered; it stays a gap, and the
archive says so rather than pretending otherwise.

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
