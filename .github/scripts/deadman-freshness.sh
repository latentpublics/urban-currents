#!/usr/bin/env bash
#
# Deadman question 1: is the archive still being written to? (1E, B)
#
# Lifted out of `deadman.yml` so it can be **tested**, which the arithmetic it
# contains had never been. It stays plain shell — `ls`, `sed`, `date` and
# nothing else — because this is the check that has to survive the project
# being unbuildable. Question 2 in the workflow needs `uv sync` to run at all;
# this one must not.
#
# ── What it measures, and what it used to ────────────────────────────────────
#
# It reads the newest row's own `recorded_at` — the moment the run wrote it.
# Until 1E it measured from the row's **date at midnight UTC**, and that is a
# different quantity: the date is the day the issue covers, and the run that
# covers it lands 22:50–23:26 UTC on that day. So a healthy archive was always
# reported as 33 hours old at the 09:00 slot, against a 36-hour limit — three
# hours of margin for a watchdog whose own scheduler is routinely hours late.
# On 2026-09-07 the deadman ran late, measured 38 hours, and mailed an alarm
# about a pipeline that had finished normally fifteen hours earlier and would
# run again seven hours later.
#
# The comment that chose the date over the file's mtime is still right and does
# not apply here: *"a fresh checkout gives every file today's mtime, so mtime
# here would always look healthy."* mtime is a property of the checkout.
# `recorded_at` is a property of the run — it travels in the committed row, and
# a fresh checkout cannot change it.
#
# ── The fallback is loud on purpose ──────────────────────────────────────────
#
# Every row in `content/runs_log/` has carried `recorded_at` since the first
# one; `Outcome.as_dict()` always writes it. If one ever does not, this falls
# back to the old midnight measurement — which over-reads the age by up to a
# day and so errs toward alarming — and **says that it did**. A fallback that
# works silently is how you stop finding out what made it fire.
set -euo pipefail

log_dir="${RUNS_LOG_DIR:-content/runs_log}"
max_age="${MAX_AGE_HOURS:-30}"
now_epoch="${NOW_EPOCH:-$(date -u +%s)}"

summary() {
  [ -n "${GITHUB_STEP_SUMMARY:-}" ] && printf '%s\n' "$1" >> "$GITHUB_STEP_SUMMARY"
  return 0
}

latest="$(ls -1 "${log_dir}"/*.json 2>/dev/null | sort | tail -1 || true)"
if [ -z "$latest" ]; then
  echo "::error::${log_dir}/ is empty — the pipeline has never recorded a run, or the checkout is wrong."
  exit 1
fi

day="$(basename "$latest" .json)"

# `sed`, not a JSON parser: the row is written by `json.dumps(indent=2,
# sort_keys=True)`, so the field is one line and its shape is fixed by the
# writer rather than guessed at here.
recorded_at="$(sed -n 's/.*"recorded_at": *"\([^"]*\)".*/\1/p' "$latest" | head -1)"

measured_from="recorded_at"
latest_epoch=""
if [ -n "$recorded_at" ]; then
  latest_epoch="$(date -u -d "$recorded_at" +%s 2>/dev/null || true)"
fi
if [ -z "$latest_epoch" ]; then
  measured_from="the row's date at midnight"
  latest_epoch="$(date -u -d "${day} 00:00:00" +%s)"
  echo "::warning::${day}.json has no readable recorded_at, so its age is measured from midnight UTC on its own date. That over-reads the age by up to a day."
  summary "**\`${day}.json\` has no readable \`recorded_at\`.** Its age is measured from midnight on the row's date instead, which over-reads it by up to a day."
fi

age_hours=$(( (now_epoch - latest_epoch) / 3600 ))

echo "newest run-log row: ${day}, written ${recorded_at:-unknown} (${age_hours}h old by ${measured_from}, limit ${max_age}h)"
summary "## Deadman"
summary ""
summary "Newest run-log row: \`${day}\`, written \`${recorded_at:-unknown}\` — ${age_hours}h ago (limit ${max_age}h)."

if [ "$age_hours" -gt "$max_age" ]; then
  echo "::error::Nothing has been written to the archive for ${age_hours}h — the newest row is ${day}, written ${recorded_at:-unknown}. The daily schedule may be disabled — check the Actions tab, and see docs/OPERATIONS.md."
  summary ""
  summary "**Nothing has been written to the archive for ${age_hours} hours.**"
  summary ""
  summary "This job does not know why. The three usual causes:"
  summary ""
  summary "- GitHub disabled the schedule (60 days without repository activity)"
  summary "- a syntax error in a workflow file, which fails silently"
  summary "- an account or billing problem"
  summary ""
  summary "Re-run \`daily\` by hand from the Actions tab to tell them apart."
  exit 1
fi
