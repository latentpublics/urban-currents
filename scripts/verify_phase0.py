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
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

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


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


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


def check_e2e(d: date, skip: bool) -> Check:
    c = Check("2. real end-to-end run against live APIs")
    if skip:
        c.status = "SKIPPED"
        c.detail = "--skip-e2e"
        return c
    code, out = run(["uv", "run", "uc", "run", "--date", str(d)])
    stages = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("[")]
    failed = [s for s in stages if "FAILED" in s or "EMPTY" in s]
    skipped = [s for s in stages if "SKIPPED" in s]
    c.evidence = stages
    if code != 0 or failed:
        c.status = "FAIL"
        c.detail = f"{len(failed)} stage(s) failed: {failed}"
    elif skipped:
        c.status = "PASS"
        c.detail = f"all stages ran; skipped: {[s.split(']')[1].strip() for s in skipped]}"
    else:
        c.status = "PASS"
        c.detail = "all stages OK"
    return c


def check_outputs(d: date) -> Check:
    c = Check("2b. expected artefacts exist")
    items = store.all_item_files()
    issue = store.issue_path(d)
    preview = paths.RUNS / f"run_{d}" / "preview.html"
    c.evidence = [
        f"content/items: {len(items)} files",
        f"content/issues/{d}.json: {'yes' if issue.exists() else 'NO'}",
        f"runs/run_{d}/preview.html: {'yes' if preview.exists() else 'NO'}",
    ]
    c.status = "PASS" if (items and issue.exists() and preview.exists()) else "FAIL"
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


def check_preview(d: date) -> Check:
    c = Check("5. preview parses and its card count matches items_published")
    preview = paths.RUNS / f"run_{d}" / "preview.html"
    issue = store.load_issue(d)
    if not preview.exists() or issue is None:
        c.status = "FAIL"
        c.detail = "preview or issue missing"
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


def _content_snapshot() -> dict[str, bytes]:
    return {
        p.relative_to(paths.ROOT).as_posix(): p.read_bytes()
        for p in sorted(paths.CONTENT.rglob("*"))
        if p.is_file()
    }


def check_idempotency(d: date, skip: bool) -> Check:
    c = Check("6. running the same date twice leaves content/ unchanged")
    if skip:
        c.status = "SKIPPED"
        c.detail = "--skip-e2e"
        return c
    before = _content_snapshot()
    code, _ = run(["uv", "run", "uc", "run", "--date", str(d)])
    after = _content_snapshot()

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])

    c.status = "PASS" if not (added or removed or changed) and code == 0 else "FAIL"
    c.detail = (
        f"{len(before)} files before, {len(after)} after; "
        f"+{len(added)} -{len(removed)} ~{len(changed)}"
    )
    c.evidence = (added + removed + changed)[:10]
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
    llm_state = paths.STATE / "llm_usage.json"
    llm_total = 0.0
    if llm_state.exists():
        llm_total = json.loads(llm_state.read_text(encoding="utf-8")).get("cost_usd", 0.0)

    grand = costs["total_usd"] + sum(extra.values())
    c.status = "PASS"
    c.detail = f"total measured spend ${grand:.4f} (LLM ${llm_total:.4f})"
    c.evidence = [
        f"daily runs: ${costs['total_usd']:.4f} over {costs['days']} day(s)",
        *[f"{k}: ${v:.4f}" for k, v in extra.items()],
        f"LLM cumulative (runs/state/llm_usage.json): ${llm_total:.4f}",
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

    checks = [
        check_pytest(),
        check_e2e(d, args.skip_e2e),
        check_outputs(d),
        check_schema(),
        check_free_strings(),
        check_preview(d),
        check_idempotency(d, args.skip_e2e),
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
