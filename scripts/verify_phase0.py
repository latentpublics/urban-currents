"""Phase 0 final verification — the nine checks, run as one command.

Each check prints PASS / FAIL / SKIPPED with the evidence it used, and the whole
thing exits non-zero if any check fails. Written as a script rather than a
checklist so the result can be reproduced instead of trusted.

Usage:
    uv run python scripts/verify_phase0.py --date 2026-08-11
    uv run python scripts/verify_phase0.py --date 2026-08-11 --skip-e2e
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.models import CANONICAL_PREFIXES  # noqa: E402
from pipeline.validate import validate_content  # noqa: E402


@dataclass
class Check:
    name: str
    status: str = "PENDING"
    detail: str = ""
    evidence: list[str] = field(default_factory=list)


class CardCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards = 0
        self.external: list[str] = []
        self.open_tags: list[str] = []
        self.unbalanced = 0
        self.has_title = False

    VOID = {"meta", "link", "br", "hr", "img", "input", "source"}

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "article" and "uc-card" in (d.get("class") or ""):
            self.cards += 1
        if tag == "title":
            self.has_title = True
        if tag in ("img", "script", "iframe", "link"):
            src = d.get("src") or d.get("href") or ""
            if src.startswith(("http://", "https://", "//")):
                self.external.append(src)
        if tag not in self.VOID:
            self.open_tags.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self.open_tags and self.open_tags[-1] == tag:
            self.open_tags.pop()
        elif tag in self.open_tags:
            while self.open_tags and self.open_tags.pop() != tag:
                self.unbalanced += 1
        else:
            self.unbalanced += 1


def _last_json(out: str) -> Optional[dict]:
    """The last JSON object printed on stdout, ignoring log lines around it."""
    start = out.rfind("\n{")
    if start == -1:
        start = out.find("{")
    if start == -1:
        return None
    end = out.rfind("}")
    if end == -1 or end < start:
        return None
    try:
        return json.loads(out[start : end + 1])
    except json.JSONDecodeError:
        return None


def run(cmd: list[str], content: Optional[Path] = None) -> tuple[int, str]:
    """Run a command, optionally with the published archive redirected.

    `content` points `UC_CONTENT` at a sandbox so a verification run cannot
    write into the real archive. Phase 0h's run left a ghost issue behind —
    `content/issues/2026-08-14.json`, a quiet day with no items, authored by a
    test rather than by a day's work — and running verification daily would
    accumulate those.
    """
    env = dict(os.environ)
    if content is not None:
        env["UC_CONTENT"] = str(content)
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def sandbox_content() -> Path:
    """A copy of the archive for the live run to write into.

    Copied rather than empty: the immutability guard and the status-change path
    only do anything when a day already exists, and a verification that never
    exercises them is verifying a simpler pipeline than the real one.
    """
    target = Path(tempfile.mkdtemp(prefix="uc-verify-content-")) / "content"
    shutil.copytree(paths.CONTENT, target)
    return target


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_pytest() -> Check:
    c = Check("1. pytest passes")
    code, out = run(["uv", "run", "pytest", "-q"])
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:]
    c.status = "PASS" if code == 0 else "FAIL"
    c.detail = tail[0] if tail else "no output"
    return c


def check_e2e(d: date, skip: bool, content: Optional[Path] = None) -> Check:
    """`uc daily` against live APIs — the command a scheduler actually calls.

    This used to run `uc run --date <today>`, a single publication date. That is
    no longer how the pipeline works and the difference is not cosmetic: a single
    recent date sits inside both sources' indexing lag, so the check began
    failing with `collect: EMPTY` on a pipeline that was working correctly. A
    verification that exercises a path production does not use can only be right
    by accident (phase 0k, final verification).

    The pass condition is the outcome, not the stage list: `published` or `quiet`
    both mean we looked. `not_published` means we did not, and that is a failure
    here however cleanly it was handled.
    """
    c = Check("2. real end-to-end run against live APIs (uc daily)")
    if skip:
        c.status = "SKIPPED"
        c.detail = "--skip-e2e"
        return c
    code, out = run(["uv", "run", "uc", "daily"], content=content)

    result = _last_json(out)
    status = (result or {}).get("status")
    c.evidence = [
        f"status: {status}",
        f"covers: {(result or {}).get('covers_from')} → {(result or {}).get('covers_to')}",
        f"candidates: {(result or {}).get('candidates')}",
        f"published: {(result or {}).get('published')}",
        f"reasons: {(result or {}).get('reasons')}",
        f"seconds: {(result or {}).get('seconds')}",
    ]
    if result is None:
        c.status = "FAIL"
        c.detail = f"no result object; exit {code}"
    elif status in ("published", "quiet"):
        c.status = "PASS"
        c.detail = (
            f"{status}: {result.get('published')} items from "
            f"{result.get('candidates')} candidates over "
            f"{result.get('covers_from')}..{result.get('covers_to')}"
        )
    else:
        c.status = "FAIL"
        c.detail = f"{status}: {result.get('reasons')}"
    return c


def check_outputs(d: date, content: Optional[Path] = None) -> Check:
    """What the live run produced — checked where the run was told to write it."""
    c = Check("2b. expected artefacts exist")
    root = content or paths.CONTENT
    items = sorted((root / "items").glob("*.json"))
    issue = root / "issues" / f"{d}.json"
    preview = paths.RUNS / f"run_{d}" / "preview.html"
    email = paths.RUNS / f"run_{d}" / "email.html"
    where = "sandbox" if content else "content"
    c.evidence = [
        f"{where}/items: {len(items)} files",
        f"{where}/issues/{d}.json: {'yes' if issue.exists() else 'NO'}",
        f"runs/run_{d}/preview.html: {'yes' if preview.exists() else 'NO'}",
        f"runs/run_{d}/email.html: {'yes' if email.exists() else 'NO'}",
    ]
    c.status = (
        "PASS"
        if (items and issue.exists() and preview.exists() and email.exists())
        else "FAIL"
    )
    c.detail = "; ".join(c.evidence)
    return c


def check_schema() -> Check:
    c = Check("3. every file under content/ validates")
    result = validate_content()
    c.status = "PASS" if result.ok else "FAIL"
    c.detail = f"{result.checked} files, {len(result.errors)} errors"
    c.evidence = result.errors[:10]
    return c


def check_free_strings() -> Check:
    c = Check("4. zero free strings in entities")
    bad = []
    n_tags = 0
    for item in store.iter_items():
        for facet in ("topics", "people", "orgs", "methods", "data", "tools", "places"):
            for ref in getattr(item.entities, facet):
                n_tags += 1
                if not ref.id.startswith(CANONICAL_PREFIXES):
                    bad.append(f"{item.work_key}: {facet}: {ref.id!r}")
    c.status = "PASS" if not bad else "FAIL"
    c.detail = f"{n_tags} tags checked, {len(bad)} without a canonical prefix"
    c.evidence = bad[:10]
    return c


def check_preview(d: date, content: Optional[Path] = None) -> Check:
    c = Check("5. preview parses and its card count matches items_published")
    preview = paths.RUNS / f"run_{d}" / "preview.html"
    # Read the issue from wherever the run was told to write it. This used to
    # read the real archive unconditionally while the run wrote to the sandbox,
    # so the check could only pass when a same-dated issue happened to exist in
    # `content/` already — right answer, wrong reason (phase 0k).
    issue = _load_issue_from(d, content)
    if not preview.exists() or issue is None:
        c.status = "FAIL"
        c.detail = (
            f"preview {'ok' if preview.exists() else 'missing'}, "
            f"issue {'ok' if issue else 'missing'} in "
            f"{'sandbox' if content else 'content'}"
        )
        return c

    parser = CardCounter()
    parser.feed(preview.read_text(encoding="utf-8"))
    ok = (
        parser.cards == issue.scan_meta.items_published
        and parser.cards == len(issue.items)
        and not parser.external
        and parser.unbalanced == 0
        and parser.has_title
    )
    c.status = "PASS" if ok else "FAIL"
    c.detail = (
        f"cards={parser.cards}, items_published={issue.scan_meta.items_published}, "
        f"issue.items={len(issue.items)}, external refs={len(parser.external)}, "
        f"unbalanced tags={parser.unbalanced}"
    )
    c.evidence = parser.external[:5]
    return c


def _load_issue_from(d: date, content: Optional[Path]):
    """One issue, from the sandbox when there is one and from the archive when not."""
    from pipeline.models import Issue

    root = content or paths.CONTENT
    path = root / "issues" / f"{d}.json"
    if not path.exists():
        return None
    try:
        return Issue.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unparseable issue is a failed check
        return None


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _content_snapshot() -> dict[str, bytes]:
    return {
        p.relative_to(paths.ROOT).as_posix(): p.read_bytes()
        for p in sorted(paths.CONTENT.rglob("*"))
        if p.is_file()
    }


def check_idempotency(d: date, skip: bool, content: Optional[Path] = None) -> Check:
    """The same day twice: identical archive, and nothing sent a second time.

    Two exclusions, both deliberate. `runs_log/` records that a second attempt
    happened — `attempts` and `recorded_at` change by design, and **a log that
    did not change when you ran again would be a broken log**. `deliveries/` is
    checked separately and more strictly: not "unchanged bytes" but "no extra
    send", which is the property that matters to a reader.
    """
    c = Check("6. running the same date twice leaves content/ unchanged, and sends nothing")
    if skip:
        c.status = "SKIPPED"
        c.detail = "--skip-e2e"
        return c

    # Measured inside the sandbox the e2e run wrote into: idempotency is a
    # property of the pipeline, and asking it of an archive the run was
    # forbidden to touch would pass for the wrong reason.
    root = content or paths.CONTENT
    # `state/` joins them for the same reason `runs_log/` is here: it records
    # facts about **runs**, not about content. Cumulative LLM spend goes up when
    # you run again — that is what cumulative means — and a spend figure that
    # did not move on a second run would be the broken one. It lives under
    # `content/` only because that is the one directory CI keeps (0U, U6).
    volatile = ("runs_log/", "deliveries/", "state/")

    def archive() -> dict[str, bytes]:
        return {
            k: v
            for k, v in _snapshot(root).items()
            if not k.replace("\\", "/").startswith(volatile)
        }

    def sends() -> int:
        total = 0
        for path in sorted((root / "deliveries").glob("*.json")):
            try:
                total += len(json.loads(path.read_text(encoding="utf-8")).get("sends") or [])
            except json.JSONDecodeError:
                continue
        return total

    before, sends_before = archive(), sends()
    code, out = run(["uv", "run", "uc", "daily"], content=content)
    after, sends_after = archive(), sends()

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    identical = not (added or removed or changed)
    no_extra_send = sends_after == sends_before

    c.status = "PASS" if identical and no_extra_send and code == 0 else "FAIL"
    c.detail = (
        f"{len(before)} files before, {len(after)} after; "
        f"+{len(added)} -{len(removed)} ~{len(changed)}; "
        f"sends {sends_before} → {sends_after}"
    )
    c.evidence = [
        f"second run: {(_last_json(out) or {}).get('status')}",
        f"delivery: {((_last_json(out) or {}).get('delivery') or {}).get('status')}",
        "runs_log/, deliveries/ and state/ excluded from the byte comparison; "
        "sends counted instead",
    ] + (added + removed + changed)[:8]
    return c


def check_report() -> Check:
    c = Check("7. uc report writes docs/phase0-report.md")
    code, out = run(["uv", "run", "uc", "report"])
    target = paths.DOCS / "phase0-report.md"
    ok = code == 0 and target.exists() and target.stat().st_size > 500
    c.status = "PASS" if ok else "FAIL"
    c.detail = (
        f"{target.relative_to(paths.ROOT)}: "
        f"{target.stat().st_size if target.exists() else 0} bytes"
    )
    return c


def check_costs() -> Check:
    c = Check("9. measured cost tally")
    from pipeline.report import cost_summary, load_runs

    runs = load_runs()
    costs = cost_summary(runs)

    extra = {}
    for name, path in (
        ("trainset", paths.RUNS / "trainset" / "trainset.meta.json"),
        ("backfill", paths.RUNS / "backfill" / "backfill.meta.json"),
    ):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            extra[name] = data.get("openalex_cost_usd", 0.0)
    # `persistent_state`, not `paths.STATE` (0U, U6). The tally moved under
    # `content/` so it survives CI; the old copy is still on disk here and
    # reading it would report a number frozen at the moment of the move.
    llm_state = paths.persistent_state("llm_usage.json")
    llm_total = 0.0
    if llm_state.exists():
        llm_total = json.loads(llm_state.read_text(encoding="utf-8")).get("cost_usd", 0.0)

    grand = costs["total_usd"] + sum(extra.values())
    c.status = "PASS"
    c.detail = f"total measured spend ${grand:.4f} (LLM ${llm_total:.4f})"
    c.evidence = [
        f"daily runs: ${costs['total_usd']:.4f} over {costs['days']} day(s)",
        *[f"{k}: ${v:.4f}" for k, v in extra.items()],
        f"LLM cumulative ({llm_state}): ${llm_total:.4f}",
        f"per published item: {costs['per_item_usd']}",
        f"monthly estimate from daily runs: {costs['monthly_estimate_usd']}",
    ]
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()
    d = date.fromisoformat(args.date)

    # The live run writes into a copy. Nothing this script does can add a day to
    # the real archive, which is how phase 0h ended up with a ghost issue.
    sandbox = None if args.skip_e2e else sandbox_content()

    checks = [
        check_pytest(),
        check_e2e(d, args.skip_e2e, content=sandbox),
        check_outputs(d, content=sandbox),
        check_schema(),
        check_free_strings(),
        check_preview(d, content=sandbox),
        check_idempotency(d, args.skip_e2e, content=sandbox),
        check_report(),
        check_costs(),
    ]

    print("\n" + "=" * 78)
    print("PHASE 0 FINAL VERIFICATION")
    print("=" * 78)
    for c in checks:
        print(f"\n[{c.status}] {c.name}")
        if c.detail:
            print(f"        {c.detail}")
        for e in c.evidence:
            print(f"        - {e}")

    failed = [c for c in checks if c.status == "FAIL"]
    print("\n" + "-" * 78)
    print(
        f"{sum(1 for c in checks if c.status == 'PASS')} passed, "
        f"{len(failed)} failed, "
        f"{sum(1 for c in checks if c.status == 'SKIPPED')} skipped"
    )

    out = paths.RUNS / "verification.json"
    out.write_text(
        json.dumps(
            [{"name": c.name, "status": c.status, "detail": c.detail,
              "evidence": c.evidence} for c in checks],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"written to {out.relative_to(paths.ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
