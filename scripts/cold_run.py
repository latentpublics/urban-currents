"""How long does a completely cold run take? (hotfix 2, H8)

CI was killed at 45 minutes. The local number everyone was quoting — `daily_s:
253.3`, four minutes — was measured on a machine with 783 cached LLM responses
and a 440MB embedding model already on disk. **Those are not the same run**, and
the gap between them was being filled with guesses.

This measures the cold one. Two caches are taken away:

- `runs/cache/` — the LLM response cache, keyed by prompt version and work key.
  Moved aside, so every summary and extraction is paid for again. **This costs
  real money** (~$0.14 at the last measurement) and that is the point: the
  expense is the thing being measured.
- `HF_HOME` — pointed at an empty directory so the embedding model downloads
  from scratch. The user's real cache is 6.5GB and shared with other work, so it
  is redirected rather than moved.

Runs `--dry-run` against a sandboxed `UC_CONTENT`, so nothing reaches the
archive. Restores the LLM cache in a `finally` — an interrupted measurement must
not leave the project without its cache.

Usage:
    uv run python scripts/cold_run.py            # the full cold measurement
    uv run python scripts/cold_run.py --warm     # same run, caches left in place
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "runs" / "cache"
PARKED = ROOT / "runs" / "cache.warm-parked"


def stage_timings(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        "timing": m.get("timing", {}),
        "counts": m.get("counts", {}),
        "cost": m.get("cost", {}),
        "stages": m.get("stages", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", action="store_true", help="leave the caches in place")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sandbox = Path(tempfile.mkdtemp(prefix="uc-cold-content-")) / "content"
    shutil.copytree(ROOT / "content", sandbox)

    env = dict(os.environ)
    env["UC_CONTENT"] = str(sandbox)
    env["PYTHONUNBUFFERED"] = "1"

    # The run directory has to be fresh or the per-stage numbers are fiction.
    #
    # `Run.timed()` **accumulates**: `timing[key] = timing.get(key, 0) + elapsed`.
    # Re-running a date adds to the same counters, so after four runs of
    # 2026-08-18 the file claimed `collect_s: 1842` against a `daily_s` of 426 —
    # a stage apparently taking four times the run that contained it. `daily_s`
    # is assigned rather than accumulated, which is why it stayed honest and why
    # the contradiction was visible at all.
    from datetime import date as _date

    run_dir = ROOT / "runs" / f"run_{_date.today()}"
    parked_run = ROOT / "runs" / f"run_{_date.today()}.parked"
    if run_dir.exists():
        if parked_run.exists():
            shutil.rmtree(parked_run)
        run_dir.rename(parked_run)
        print(f"parked the run directory at {parked_run} (timings must start at zero)")

    hf_home = None
    if not args.warm:
        hf_home = Path(tempfile.mkdtemp(prefix="uc-cold-hf-"))
        env["HF_HOME"] = str(hf_home)
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        if CACHE.exists():
            if PARKED.exists():
                shutil.rmtree(PARKED)
            CACHE.rename(PARKED)
            print(f"parked the LLM cache at {PARKED}")

    print(f"{'warm' if args.warm else 'COLD'} run starting; content sandbox {sandbox}")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            ["uv", "run", "uc", "daily", "--dry-run"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.monotonic() - started
        print(proc.stdout[-3000:])
        if proc.returncode != 0:
            print(f"[exit {proc.returncode}]")
            print(proc.stderr[-2000:])
    finally:
        # Always give the cache back. A measurement that leaves the project
        # slower than it found it is not a measurement anyone will run twice.
        if PARKED.exists():
            if CACHE.exists():
                shutil.rmtree(CACHE)
            PARKED.rename(CACHE)
            print(f"restored the LLM cache to {CACHE}")
        # The measured run directory stays; the parked one is the history and is
        # put back beside it under its original name only if nothing took it.
        if parked_run.exists() and not run_dir.exists():
            parked_run.rename(run_dir)

    from datetime import date

    run_dir = ROOT / "runs" / f"run_{date.today()}"
    detail = stage_timings(run_dir)

    out = {
        "mode": "warm" if args.warm else "cold",
        "wall_seconds": round(elapsed, 1),
        "wall_minutes": round(elapsed / 60, 2),
        **detail,
    }
    target = ROOT / "runs" / ("cold_run.json" if not args.warm else "warm_run.json")
    target.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"\nwall clock: {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    print("stage timings:")
    for k, v in sorted((detail.get("timing") or {}).items(), key=lambda kv: -float(kv[1] or 0)):
        print(f"  {k:20} {v}")
    print(f"cost: {detail.get('cost')}")
    print(f"→ {target}")


if __name__ == "__main__":
    main()
