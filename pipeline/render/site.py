"""Home and archive pages, built from `content/` alone (phase 0j, W4).

**This is the schema test.** A home page and an archive row ask for things a
daily issue never had to answer — how many of a day's papers released code, how
many released data, which one represents the day — and the answer to "does
`content/` have it" is only half yes. Finding that out is the point of building
these two pages now rather than in Phase 1.

What an archive row needs, and where it comes from:

| field | source | present |
|---|---|---|
| date, `items_published`, `quiet_day` | `Issue` | yes |
| `unreadable_count` | `Issue.scan_meta` | yes |
| representative item | `Issue.headline.work_key` | yes |
| **code count / data count** | **nowhere** | **derived here** |

The counts are derived at render time by walking `content/items/` and reading
each item's badges. They are deliberately *not* added to `Issue`:

- An issue is immutable once published (D127). A field added today is empty for
  every issue already written, so the archive would show counts for recent days
  and blanks for older ones — a schema change that looks like a data gap.
- The derivation is cheap and stays cheap. Measured: 0.072s to load all 224
  items, 0.0015s to derive every issue's counts from them. At 1,847 items that
  is about 0.6s, and the walk is linear.
- The counter-argument is real and recorded in the report: a derived value has
  to be recomputed by every consumer, and Phase 1's Astro build would do this
  walk again. If it ever needs to be authoritative rather than convenient, it
  belongs in `scan_meta` **at write time for new issues only**, with old issues
  left null and the null meaning "not recorded then" rather than "zero".

**Orphans are excluded.** 12 items in `content/items/` are referenced by no
issue at all (D127). They are real records and they are not part of any day, so
counting them would inflate every aggregate. They are counted separately and
never deleted.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .. import paths, store
from ..config import cfg
from ..models import Issue, Item, PIPELINE_VERSION

TEMPLATE_DIR = Path(__file__).parent / "templates"

# How many days the home page's archive strip shows.
HOME_ARCHIVE_DAYS = 14

# The publish-range rule. p10-p90 over days that published anything: it drops
# the one exceptional day at each end without pretending the spread is narrower
# than it is. The quiet days are excluded from the range and reported as their
# own count — a range that includes zero is not a range, it is two facts.
RANGE_LOW, RANGE_HIGH = 0.10, 0.90


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_issues() -> list[Issue]:
    out: list[Issue] = []
    for p in sorted((paths.CONTENT / "issues").glob("*.json")):
        try:
            out.append(Issue.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - one unreadable issue is not fatal
            continue
    return sorted(out, key=lambda i: i.date)


def item_index() -> dict[str, Item]:
    return {it.work_key: it for it in store.iter_items()}


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return round(s[lo] + (s[hi] - s[lo]) * frac)


def archive_rows(
    issues: Optional[list[Issue]] = None, items: Optional[dict[str, Item]] = None
) -> list[dict]:
    """One row per issue, newest first. Everything a row shows, and nothing else."""
    issues = load_issues() if issues is None else issues
    items = item_index() if items is None else items

    rows = []
    for issue in issues:
        keys = list(issue.items)
        code = sum(1 for k in keys if k in items and "code" in items[k].badges)
        data = sum(1 for k in keys if k in items and "data" in items[k].badges)
        lead = items.get(issue.headline.work_key or "")
        rows.append({
            "date": str(issue.date),
            # `published`, never `items`: Jinja resolves `row.items` to the
            # dict's own `items` method, so the template gets a bound method and
            # fails somewhere unrelated. This is the second template in this
            # batch to hit it.
            "published": len(keys),
            "quiet": issue.quiet_day or not keys,
            "unreadable": issue.scan_meta.unreadable_count,
            # The representative title, whole. Trimming happens in CSS so the
            # DOM keeps the full string and a narrow screen only *looks*
            # shorter — a truncated string in the markup is a truncated string
            # for a screen reader too.
            "lead_title": lead.bibliography.title if lead else "",
            "lead_key": lead.work_key if lead else None,
            "code": code,
            "data": data,
        })
    return list(reversed(rows))


def archive_stats(rows: list[dict], items: dict[str, Item]) -> dict[str, Any]:
    """The figures the chrome quotes. Every one of them measured, none written in.

    The home page's own sentence about itself has to come from the archive, or
    the service is doing the thing it exists not to do.
    """
    published = [r["published"] for r in rows if not r["quiet"]]
    in_issues: set[str] = set()
    for issue in load_issues():
        in_issues |= set(issue.items) | set(issue.unreadable)

    return {
        "days": len(rows),
        "published_days": len(published),
        "quiet_days": len(rows) - len(published),
        "range_low": _quantile(published, RANGE_LOW),
        "range_high": _quantile(published, RANGE_HIGH),
        "range_rule": f"p{int(RANGE_LOW * 100)}-p{int(RANGE_HIGH * 100)} of days that published",
        "total_published": sum(published),
        "journals": _journal_count(),
        "arxiv_categories": len(cfg("arxiv.categories", []) or []),
        "code_total": sum(r["code"] for r in rows),
        "data_total": sum(r["data"] for r in rows),
        # Not part of any issue. Reported so the aggregate can be checked
        # against the item count and never silently absorbed into it.
        "orphan_items": sum(1 for k in items if k not in in_issues),
        "items_total": len(items),
    }


def _journal_count() -> int:
    """Shared with the issue stage — one definition, one number (phase 0k X0-1)."""
    from ..run_stages import _journal_count as count

    return count()


def build_home(out: Optional[Path] = None) -> Path:
    issues = load_issues()
    items = item_index()
    rows = archive_rows(issues, items)
    stats = archive_stats(rows, items)

    latest = next((i for i in reversed(issues) if i.items), None)
    lead_item = items.get(latest.headline.work_key or "") if latest else None

    env = _env()
    css = (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8")
    html = env.get_template("home.html.j2").render(
        stats=stats,
        rows=rows[:HOME_ARCHIVE_DAYS],
        max_items=max([r["published"] for r in rows] or [1]),
        latest=latest,
        # The full headline line, not a regenerated short one. There is room.
        lead_line=latest.headline.line if latest else None,
        lead_title=lead_item.bibliography.title if lead_item else None,
        synthesis=latest.synthesis if latest else None,
        css=Markup(css),
        pipeline_version=PIPELINE_VERSION,
    )
    target = out or (paths.ROOT / "site" / "index.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8", newline="\n")
    return target


def build_archive(out: Optional[Path] = None) -> Path:
    issues = load_issues()
    items = item_index()
    rows = archive_rows(issues, items)
    stats = archive_stats(rows, items)

    by_month: dict[str, list[dict]] = {}
    for row in rows:
        by_month.setdefault(row["date"][:7], []).append(row)

    env = _env()
    css = (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8")
    html = env.get_template("archive.html.j2").render(
        stats=stats,
        months=[
            {"key": k, "label": _month_label(k), "rows": v}
            for k, v in sorted(by_month.items(), reverse=True)
        ],
        max_items=max([r["published"] for r in rows] or [1]),
        css=Markup(css),
        pipeline_version=PIPELINE_VERSION,
    )
    target = out or (paths.ROOT / "site" / "archive.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8", newline="\n")
    return target


_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _month_label(key: str) -> str:
    year, month = key.split("-")
    return f"{_MONTHS[int(month) - 1]} {year}"


REVIEW_SCREENS = (
    ("2026-08-05", "A thin day — the synthesis box carries measured zeros and no paragraph"),
    ("2026-08-06", "A canon anchor, one repeat author"),
    ("2026-08-07", "Two canon anchors, no coupling"),
    ("2026-08-10", "Two archive couplings"),
    ("2026-08-11", "The fullest day — coupling with shared references, four institutions"),
)


def build_design_review(out: Optional[Path] = None) -> Path:
    """Every screen stacked in one file, from real `content/` and nothing else.

    The order puts the thin day first and the full day last, so the question the
    design has to answer — does an issue with nothing to say still look like an
    issue — is the first thing visible rather than something to hunt for.
    """
    import re

    from ..metrics import Run
    from ..render.preview import render_issue

    css = (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8")
    items = item_index()

    def body_of(html: str) -> str:
        m = re.search(r"<body>(.*)</body>", html, re.S)
        return m.group(1) if m else html

    parts: list[str] = []
    for day, note in REVIEW_SCREENS:
        issue = store.load_issue(date.fromisoformat(day))
        if issue is None:
            continue
        day_items = [items[k] for k in issue.items if k in items]
        unreadable = [items[k] for k in issue.unreadable if k in items]
        parts.append((f"Issue {day} — {note}", body_of(render_issue(issue, day_items, unreadable))))

    home = build_home(out=paths.ROOT / "site" / "index.html")
    archive = build_archive(out=paths.ROOT / "site" / "archive.html")
    parts.append(("Home", body_of(home.read_text(encoding="utf-8"))))
    parts.append(("Archive", body_of(archive.read_text(encoding="utf-8"))))

    screens = "\n".join(
        f'<section class="uc-review__screen">\n'
        f'<p class="uc-review__label">{label}</p>\n{body}\n</section>'
        for label, body in parts
    )
    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Urban Currents — design review</title>\n<style>\n"
        f"{css}\n"
        ".uc-review__screen { border-top: 4px solid var(--uc-ink); margin: 3rem 0 0; "
        "padding-top: 0.5rem; }\n"
        ".uc-review__label { font-size: 0.75rem; letter-spacing: 0.06em; "
        "text-transform: uppercase; color: var(--uc-meta); margin: 0 0 1.5rem; }\n"
        "</style>\n</head>\n<body>\n"
        f"{screens}\n</body>\n</html>\n"
    )
    target = out or (paths.DOCS / "design-review.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8", newline="\n")
    return target
