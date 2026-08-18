#!/usr/bin/env bash
# Drive the backfill to completion across process lifetimes (phase 0L, N1).
#
# `uc backfill-issues` checkpoints after every day, so a killed run loses at
# most the day in flight. What it does not do is restart itself — and a sixty-day
# backfill at ~2.5 minutes a day outlives any single background task here. The
# first attempt was killed at twelve minutes with three days done.
#
# So this loops: run the command, and if the checkpoint has not reached the
# target, run it again. Each pass resumes from the checkpoint. It stops when the
# target is met, when a pass makes no progress twice running (which means
# something is wrong that retrying will not fix), or when the command reports it
# has hit the spend ceiling.
set -u

cd "$(dirname "$0")/.." || exit 1

TARGET="${1:-60}"
CHECKPOINT="runs/state/backfill_issues.json"

done_count() {
  python -c "
import json,sys
try:
    d=json.load(open('$CHECKPOINT'))
    print(len(d.get('done') or []))
except Exception:
    print(0)
"
}

spent() {
  python -c "
import json
try:
    print('%.4f' % json.load(open('$CHECKPOINT')).get('spend_usd', 0.0))
except Exception:
    print('0.0000')
"
}

stale=0
while true; do
  before=$(done_count)
  if [ "$before" -ge "$TARGET" ]; then
    echo "DONE: $before day(s), \$$(spent) spent"
    break
  fi

  echo "--- pass starting at $before/$TARGET day(s), \$$(spent) spent"
  uv run uc backfill-issues --days "$TARGET" 2>&1 | tail -25

  after=$(done_count)
  if [ "$after" -le "$before" ]; then
    stale=$((stale + 1))
    echo "--- no progress (attempt $stale)"
    # Twice with nothing gained is a real blockage, not a slow day.
    if [ "$stale" -ge 2 ]; then
      echo "BLOCKED: two passes made no progress at $after/$TARGET"
      break
    fi
  else
    stale=0
  fi
done

echo "FINAL: $(done_count) day(s), \$$(spent) spent"
