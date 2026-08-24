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
from datetime import date, date as dt_date, timedelta
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .. import paths, store
from .api import build_api
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
    """One row per issue **and per day we could not see**, newest first.

    A day with no issue is not a gap in the list. A gap would read as "nothing
    happened", which is exactly the claim we could not make — so it gets a row
    saying the sources did not answer, drawn differently from a quiet day.
    Same grey chip for both would undo the whole outcome model.

    ★ A **backfilled** day is a third kind and says so (phase 0Y, Y1).
    `backfill_issues.py` writes `backfilled: true` and explains why — *"so no
    aggregate mixes the two kinds without saying so"* — and the aggregates have
    kept that promise while the screen has not: nothing under `render/` read
    the flag, so a day assembled weeks later looked exactly like a day the
    pipeline watched. It is not an apology, it is part of the method, and the
    row states it rather than hiding it.

    ★ And a day can carry **two** facts (0Y, Y1-2). A date may have both a
    `not_published` run-log row — the live run that morning withheld the day —
    and a backfilled issue written afterwards. Before this, the issue silently
    replaced the record: the `missing` row below is only added for dates with
    no issue, so filling a withheld day erased the evidence that
    `REQUIRED_SOURCES` had done its job. `pages.yml` states the principle this
    breaks — *"the archive is the record"* — so the row now says both.
    """
    from ..outcome import NOT_PUBLISHED, all_logs

    issues = load_issues() if issues is None else issues
    items = item_index() if items is None else items

    # Dates whose live run reached `not_published`, with the reason it gave.
    # Most backfilled dates have no run log at all — the pipeline was not
    # running yet — and those simply do not appear here.
    withheld = {
        log["date"]: (log.get("reasons") or ["the sources did not answer"])[0]
        for log in all_logs()
        if log.get("status") == NOT_PUBLISHED
    }

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
            # Derived from the count, never from the stored flag (0Z, Z1).
            # `quiet_day` on disk answers "did anything clear the headline
            # bar", and 2026-08-09 (11 papers) and 2026-08-21 (9) both carry it
            # as true. Deriving here means those files cannot put "a quiet day"
            # on a day that published nine papers.
            "quiet": issue.is_quiet,
            # The other fact, said in its own words rather than borrowed from
            # the first. A day with papers and no headline is not quiet.
            "unranked": not issue.is_quiet and not issue.has_headline,
            "unreadable": issue.scan_meta.unreadable_count,
            # The representative title, whole. Trimming happens in CSS so the
            # DOM keeps the full string and a narrow screen only *looks*
            # shorter — a truncated string in the markup is a truncated string
            # for a screen reader too.
            "lead_title": lead.bibliography.title if lead else "",
            "lead_key": lead.work_key if lead else None,
            "code": code,
            "data": data,
            "missing": False,
            "reason": None,
            # Assembled after the fact rather than watched. Also means the day
            # skipped arXiv enrichment (`enrich_arxiv=not backfilled`), so the
            # contents differ slightly from a live day — one more reason the
            # reader is told which kind they are looking at.
            "backfilled": bool(getattr(issue, "backfilled", False)),
            # Set when the live run for this date withheld it. Both facts are
            # true and the row prints both (Y1-2).
            "withheld": withheld.get(str(issue.date)),
        })

    published_dates = {r["date"] for r in rows}
    for log in all_logs():
        if log.get("status") != NOT_PUBLISHED or log["date"] in published_dates:
            continue
        reasons = log.get("reasons") or []
        rows.append({
            "date": log["date"],
            "published": 0,
            "quiet": False,
            "missing": True,
            # The reader is told what happened, not made to guess from a blank.
            "reason": reasons[0] if reasons else "the sources did not answer",
            "unreadable": 0,
            "lead_title": "",
            "lead_key": None,
            "code": 0,
            "data": 0,
            "backfilled": False,
            "withheld": None,
            "unranked": False,
        })

    # Recency is **additive**, not a fifth state (0Z, Z2). published / quiet /
    # missing / backfilled say what happened on a day; recency says whether our
    # knowledge of that day is still settling. A day can be quiet *and* recent,
    # or withheld *and* recent, and collapsing the two into one label would be
    # the same mistake Z1 exists to undo. So no state wins — the row keeps its
    # kind and gains a note.
    cutoff = str(recent_cutoff())
    for r in rows:
        r["recent"] = r["date"] >= cutoff

    rows.sort(key=lambda r: r["date"])
    return list(reversed(rows))


def archive_stats(rows: list[dict], items: dict[str, Item]) -> dict[str, Any]:
    """The figures the chrome quotes. Every one of them measured, none written in.

    The home page's own sentence about itself has to come from the archive, or
    the service is doing the thing it exists not to do.
    """
    # A day we could not see is neither published nor quiet, and counting it as
    # either would put it back into an aggregate the outcome model just took it
    # out of. This caught itself in test: `published_days` read 1 for an archive
    # whose only row was a failure.
    published = [
        r["published"] for r in rows if not r["quiet"] and not r.get("missing")
    ]
    in_issues: set[str] = set()
    for issue in load_issues():
        in_issues |= set(issue.items) | set(issue.unreadable)

    return {
        "days": len(rows),
        "published_days": len(published),
        "quiet_days": sum(1 for r in rows if r["quiet"]),
        "missing_days": sum(1 for r in rows if r.get("missing")),
        # Counted separately, not subtracted. `backfill_issues.py` promised
        # that no aggregate would mix the two kinds "without saying so"; this
        # is the saying so. `published_days` still counts every day that
        # published, because it did, and this says how many of those nobody
        # watched at the time (0Y, Y1).
        "backfilled_days": sum(1 for r in rows if r.get("backfilled")),
        "range_low": _quantile(published, RANGE_LOW),
        "range_high": _quantile(published, RANGE_HIGH),
        "range_rule": f"p{int(RANGE_LOW * 100)}-p{int(RANGE_HIGH * 100)} of days that published",
        "total_published": sum(published),
        # The masthead says "since <month>", and it has to be the archive's own
        # first month rather than a date written into the template.
        "first_month": _month_label(min(r["date"] for r in rows)[:7]) if rows else "",
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


ASSET_DIR = Path(__file__).resolve().parent / "assets"


def copy_assets(target_root: Optional[Path] = None) -> list[Path]:
    """Copy the vendored fonts into the built site.

    They live in the package rather than in `site/`, because `site/` is build
    output and a clean rebuild would take them with it. Copying on every build
    means a checkout plus `uc site` produces a complete tree — and means the
    "every local url() resolves" check is testing the real thing rather than
    whatever happens to be lying in the working directory.
    """
    import shutil

    root = target_root or (paths.ROOT / "site")
    written: list[Path] = []
    for src in sorted((ASSET_DIR / "fonts").glob("*")):
        dst = root / "assets" / "fonts" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        written.append(dst)
    return written


def site_font_css(prefix: str) -> str:
    """The `@font-face` block, for **site artefacts only**.

    Takes the relative prefix rather than assuming one, because the pages sit at
    three depths — `site/index.html`, `site/issues/*.html` and
    `docs/design-review.html` — and a single hard-coded path would silently 404
    for two of them while the CSS still parsed. A missing font does not throw;
    it just quietly gives you Georgia again, which is the bug being fixed.

    Not added to `runs/*/preview.html` or to `email.html`. See the template.
    """
    env = _env()
    return env.get_template("site_fonts.css.j2").render(font_prefix=prefix)


def with_fonts(css: str, prefix: str) -> str:
    """`@font-face` first, then the stylesheet that uses the families."""
    return f"{site_font_css(prefix)}\n{css}"


def latest_issue(issues: list[Issue]) -> Optional[Issue]:
    """The issue the home page leads with (phase 0Z, Z3).

    By default the newest issue that is **older than `site.recent_days`**, so a
    first-time reader lands on a day that has stopped filling in. arXiv indexes
    about three days late, and an issue from yesterday does not yet hold
    everything posted yesterday.

    **The cost is deliberate and visible**: the front page runs three days
    behind, and the archive immediately below it shows newer dates than the
    issue above. That is a trade YJUN chose, and `site.latest_skips_recent`
    turns it off in one line without touching code.

    Falls back to the newest issue with items when the filter would leave
    nothing — a young archive should still have a front page.
    """
    with_items = [i for i in issues if i.items]
    if not with_items:
        return None
    newest = max(with_items, key=lambda i: i.date)
    if not bool(cfg("site.latest_skips_recent", True)):
        return newest
    cutoff = recent_cutoff()
    settled = [i for i in with_items if i.date < cutoff]
    return max(settled, key=lambda i: i.date) if settled else newest


def spark_bars(rows: list[dict], today: Optional[str] = None) -> list[dict]:
    """Mockup 5a's fourteen bars, oldest on the left.

    Height is the day's published count against the tallest day in the window,
    with a floor so a quiet day is still a visible mark rather than nothing —
    **a bar of zero height would say "no day here", and there was a day here.**

    The kinds are not shades of the same claim:

      `published`   an issue went out
      `quiet`       we looked and published nothing
      `missing`     we could not see the day at all
      `backfilled`  assembled later; nobody was watching that day
      `featured`    the issue shown above — not necessarily the newest (Z3)

    `missing` is drawn like `quiet` in the mockup, which has no such day in it.
    It gets its own class here so the two can never be read as one; a day nobody
    could see is not a day with nothing in it, and the whole outcome model rests
    on that distinction.

    ★ `backfilled` was missing entirely until 0Z (Z8): the list drew it with a
    dashed blue bar and a chip while the histogram drew it exactly like a live
    published day, so the same state had two different appearances forty pixels
    apart. `recent` rides along as a flag rather than a kind, for the reason
    given in `archive_rows`.

    ★ The colours come from the `--uc-state-*` tokens, which the list rows use
    too. Before 0Z the mapping was hardcoded twice, ninety lines apart, and
    `quiet` was dark grey in one and near-white in the other.
    """
    window = list(reversed(rows))  # oldest first, the way a time axis reads
    tallest = max([r["published"] for r in window] or [1]) or 1
    bars = []
    for r in window:
        if r.get("missing"):
            kind, note = "missing", f"not seen — {r.get('reason') or 'the sources did not answer'}"
        elif r["quiet"] or not r["published"]:
            kind, note = "quiet", "a quiet day"
        elif r.get("backfilled"):
            kind, note = "backfilled", f"{r['published']} items, filled in later"
        else:
            kind, note = "published", f"{r['published']} items"
        if r.get("unranked"):
            note += ", no headline"
        if r.get("recent"):
            note += "; more may follow"
        # `featured`, not `today`: with `site.latest_skips_recent` the issue on
        # the home page is deliberately a few days old, and calling its bar
        # "today" would have the page state a date it is not showing (0Z, Z3).
        if today and r["date"] == today:
            kind = "featured"
        height = max(8, round(100 * r["published"] / tallest)) if r["published"] else 12
        bars.append({
            "date": r["date"],
            "kind": kind,
            "recent": bool(r.get("recent")),
            "height": height,
            "label": f"{r['date']}: {note}",
        })
    return bars


def _tag_shift_label(synthesis) -> str:
    """What the stat rail says about tag drift — in three states, not two.

    `NO_BASELINE` means the archive behind this day was too thin to compare
    against. Printing `0` there would claim we looked and found nothing, and
    this project has now corrected that same mistake four times.
    """
    if synthesis is None or not getattr(synthesis, "deviation_status", None):
        return "not measured"
    status = synthesis.deviation_status
    if status != "OK":
        return "not measured"
    deviations = list(getattr(synthesis, "deviations", None) or [])
    if not deviations:
        return "none"
    first = deviations[0]
    tag = getattr(first, "tag", None) or getattr(first, "label", None) or "shift"
    return f"{tag} {len(deviations)}"


def build_home(out: Optional[Path] = None) -> Path:
    issues = load_issues()
    items = item_index()
    rows = archive_rows(issues, items)
    stats = archive_stats(rows, items)

    latest = latest_issue(issues)
    lead_item = items.get(latest.headline.work_key or "") if latest else None

    window = rows[:HOME_ARCHIVE_DAYS]
    latest_row = next((r for r in window if latest and r["date"] == str(latest.date)), None)
    bars = spark_bars(window, today=str(latest.date) if latest else None)

    env = _env()
    css = with_fonts(
        (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8"), "assets/fonts/"
    )
    # Derived, like every other number on this page: the description a search
    # engine shows is the same sentence the page opens with, built from the
    # archive rather than typed into the template.
    description = (
        f"{stats['arxiv_categories']} arXiv categories and {stats['journals']} journals, "
        f"read every day. {stats['published_days']} issues since {stats['first_month']}; "
        f"{stats['total_published']} papers worth your time, and the quiet days said so."
    )
    html = env.get_template("home.html.j2").render(
        head_extras=Markup(head_extras(
            description=description,
            path="",
            title="Urban Currents — a daily scan of urban data science",
            jsonld=ld_home(stats, _base_url()),
        )),
        quality=selection_quality(),
        nav=nav_links("", latest),
        stats=stats,
        rows=rows[:HOME_ARCHIVE_DAYS],
        # `or 1` on the outside too: an archive whose every row published
            # nothing gives `max(...)` of 0, and the bar width divides by
            # this (0Z). An all-quiet archive is rare but it is exactly the
            # case a young or a broken week produces.
            max_items=max([r["published"] for r in rows] or [1]) or 1,
        latest=latest,
        # The full headline line, not a regenerated short one. There is room.
        lead_line=latest.headline.line if latest else None,
        lead_title=lead_item.bibliography.title if lead_item else None,
        synthesis=latest.synthesis if latest else None,
        latest_code=latest_row["code"] if latest_row else 0,
        latest_data=latest_row["data"] if latest_row else 0,
        tag_shift=_tag_shift_label(latest.synthesis if latest else None),
        spark=bars,
        spark_from=bars[0]["date"] if bars else "",
        spark_to=bars[-1]["date"] if bars else "",
        css=Markup(css),
        pipeline_version=PIPELINE_VERSION,
    )
    target = out or (paths.ROOT / "site" / "index.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8", newline="\n")
    return target


def _month_buckets(rows: list[dict]) -> list[dict]:
    """Rows grouped by month, newest month first, each with its own tally."""
    by_month: dict[str, list[dict]] = {}
    for row in rows:
        by_month.setdefault(row["date"][:7], []).append(row)
    months = []
    for key, group in sorted(by_month.items(), reverse=True):
        months.append({
            "key": key,
            "label": _month_label(key),
            "short": _month_short(key),
            "rows": group,
            # Counted per month rather than derived in the template, because
            # "10 issues · 3 quiet" is a claim and the template is not where a
            # claim should be assembled.
            "published": sum(1 for r in group if not r["quiet"] and not r["missing"]),
            "quiet": sum(1 for r in group if r["quiet"]),
            "missing": sum(1 for r in group if r["missing"]),
        })
    return months


def _month_tabs(months: list[dict], current: str, root: str) -> list[dict]:
    """One tab per month. The current one is not a link to itself."""
    return [
        {
            "key": m["key"],
            "short": m["short"],
            "current": m["key"] == current,
            # The newest month lives at `archive.html`, so its tab points there
            # rather than at a duplicate page nothing else links to.
            "href": (
                f"{root}archive.html" if m["key"] == months[0]["key"]
                else f"{root}archive/{m['key']}.html"
            ),
        }
        for m in months
    ]


def build_archive(out: Optional[Path] = None) -> list[Path]:
    """The landing page plus one page per month.

    Returns every path written. Mockup 6b's tabs are real navigation: each is an
    `<a href>` to a page that exists, so the archive works with JavaScript off,
    in a feed reader, and for a crawler.
    """
    issues = load_issues()
    items = item_index()
    rows = archive_rows(issues, items)
    stats = archive_stats(rows, items)
    months = _month_buckets(rows)
    lead = latest_issue(issues)

    env = _env()
    raw_css = (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8")
    written: list[Path] = []

    for i, month in enumerate(months):
        newest = i == 0
        # `archive.html` sits beside `assets/`; `archive/2026-07.html` is a
        # directory deeper, and a font path that is right for one is a 404 for
        # the other.
        root = "" if newest else "../"
        target = (
            (out or (paths.ROOT / "site" / "archive.html")) if newest
            else (paths.ROOT / "site" / "archive" / f"{month['key']}.html")
        )
        month_desc = (
            f"Urban Currents issues from {month['label']}: {month['published']} "
            f"published, {month['quiet']} quiet, {month['missing']} not seen."
        )
        html = env.get_template("archive.html.j2").render(
            head_extras=Markup(head_extras(
                root,
                description=month_desc,
                path="archive.html" if newest else f"archive/{month['key']}.html",
                title=f"Urban Currents — Archive — {month['label']}",
            )),
            nav=nav_links(root, lead),
            stats=stats,
            month=month,
            tabs=_month_tabs(months, month["key"], root),
            root=root,
            # `or 1` on the outside too: an archive whose every row published
            # nothing gives `max(...)` of 0, and the bar width divides by
            # this (0Z). An all-quiet archive is rare but it is exactly the
            # case a young or a broken week produces.
            max_items=max([r["published"] for r in rows] or [1]) or 1,
            css=Markup(with_fonts(raw_css, f"{root}assets/fonts/")),
            pipeline_version=PIPELINE_VERSION,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8", newline="\n")
        written.append(target)

    return written


_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _month_short(key: str) -> str:
    """`Aug 2026` for the newest year, `Jul` within it — mockup 6b's tab row."""
    year, month = key.split("-")
    return f"{_MONTHS[int(month) - 1][:3]} {year}"


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

    from ..render.preview import render_issue

    # `docs/design-review.html` is what the fonts are actually compared in, so
    # it gets them too — from its own relative path out of `docs/` and into
    # `site/assets/`. Without this the one page built for comparing against the
    # mockup would be the one page still rendering in Georgia.
    css = with_fonts(
        (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8"),
        "../site/assets/fonts/",
    )
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

    copy_assets()
    home = build_home(out=paths.ROOT / "site" / "index.html")
    archive_pages = build_archive(out=paths.ROOT / "site" / "archive.html")
    parts.append(("Home", body_of(home.read_text(encoding="utf-8"))))
    parts.append(("Archive", body_of(archive_pages[0].read_text(encoding="utf-8"))))

    # The bodies were rendered for pages that sit in `site/`, so their relative
    # links point at `issues/…` and `archive.html` — from `docs/` those are 34
    # dead links. This is the file YJUN actually opens to compare against the
    # mockup, and a card whose title does not click is a worse review surface
    # than one whose title does.
    #
    # Rewritten rather than left, and only for links that are already relative:
    # an absolute URL is somebody else's and is not ours to repoint.
    def repoint(body: str) -> str:
        return re.sub(
            r'href="(?!https?:|mailto:|#)([^"]+)"',
            lambda m: f'href="../site/{m.group(1)}"',
            body,
        )

    screens = "\n".join(
        f'<section class="uc-review__screen">\n'
        f'<p class="uc-review__label">{label}</p>\n{repoint(body)}\n</section>'
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


def _replace_once(html: str, anchor: str, replacement: str, what: str) -> str:
    """Substitute `anchor` exactly once, and refuse to do it quietly otherwise.

    ★ This exists because of the accident it prevents (0Z-D). The site's
    navigation is injected into the issue page by finding `<main class="uc-issue"`
    and putting the nav in front of it. In 0Z-A a comment was added to
    `base.css.j2` explaining that injection, and the comment **quoted the
    anchor**. The stylesheet is inlined into `<head>`, so from that commit on
    the first match in the document was inside a CSS comment, and
    `str.replace(..., 1)` put the whole navigation there — where it is a
    comment, not a menu.

    Nothing raised. The nav markup was still in the file, so anything that
    asked whether the page *contained* a nav still passed, and a reader who
    opened an issue had no way back to the site.

    Two ways to be wrong, both now loud: **0** matches means the anchor drifted
    out of the template, **2 or more** means it is not unique any more and the
    substitution is a coin toss. Neither is a thing to find out in production.
    """
    found = html.count(anchor)
    if found != 1:
        raise RuntimeError(
            f"{what}: expected exactly one {anchor!r} to substitute, found {found}. "
            "0 means the template moved; 2+ means the anchor is also somewhere "
            "else (a CSS comment counts) and the injection would land there."
        )
    return html.replace(anchor, replacement, 1)


def issue_page_path(d: date) -> Path:
    return paths.ROOT / "site" / "issues" / f"{d}.html"


def build_issue_pages(out_dir: Optional[Path] = None) -> list[Path]:
    """One page per issue, sharing the preview's DOM exactly.

    Not a second renderer: `render_issue` produces the markup and this only adds
    the site's navigation around it. Phase 1's Astro inherits one card
    component, and a page built from a different template here would be a second
    component to keep in step with it.
    """
    from .preview import render_issue

    items = item_index()
    target_dir = out_dir or (paths.ROOT / "site" / "issues")
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    issues = load_issues()
    # The same issue the home page leads with, so `Latest` cannot point at a
    # different day from the one the front page shows (0Z, Z4).
    lead = latest_issue(issues)
    for i, issue in enumerate(issues):
        day_items = [items[k] for k in issue.items if k in items]
        unreadable = [items[k] for k in issue.unreadable if k in items]
        html = render_issue(issue, day_items, unreadable)

        previous = issues[i - 1] if i > 0 else None
        following = issues[i + 1] if i + 1 < len(issues) else None
        # The anchor carries `data-date=`, which prose about the injection
        # will not. `<main class="uc-issue"` on its own matched a CSS
        # comment that quoted it, and the nav spent a day inside the
        # stylesheet (0Z-D).
        anchor = '<main class="uc-issue" data-date='
        html = _replace_once(
            html,
            anchor,
            _issue_nav(previous, following, lead) + "\n" + anchor,
            f"issue nav for {issue.date}",
        )
        # The site copy gets the webfonts; `runs/*/preview.html` does not. Same
        # DOM, same CSS, one extra at-rule block — the preview's whole point is
        # that it is a single self-contained file, and the site's is that it
        # looks like the mockup.
        html = _replace_once(
            html,
            "<style>",
            "<style>\n" + site_font_css("../assets/fonts/"),
            f"webfonts for {issue.date}",
        )
        # The feed link and, while unpublished, the `noindex`. Injected for the
        # same reason the navigation is: this markup belongs to the site, and
        # the identical DOM is also written to `runs/` as an email.
        html = _replace_once(
            html,
            "</head>",
            head_extras(
                "../",
                description=issue_summary(issue, items),
                path=f"issues/{issue.date}.html",
                title=f"Urban Currents {issue.date}",
                og_type="article",
                jsonld=ld_issue(issue, items, _base_url(), issue_summary(issue, items)),
            )
            + "\n</head>",
            f"head extras for {issue.date}",
        )
        # ★ The site's copy titles the page with its **date** (0Z, Z6).
        #
        # `preview.html.j2` puts "Urban Currents" in the h1 because in the
        # email that is the only title there is. On the site the nav already
        # carries the brand two lines above, so the h1 repeated it and the page
        # never said, in its own heading, which day it was. Swapped here rather
        # than in the template so the email keeps the brand — and so the DOM
        # still has exactly one h1, which `test_render_contract` requires.
        html = _replace_once(
            html,
            '<h1 class="uc-issue__title">Urban Currents</h1>',
            f'<h1 class="uc-issue__title">{issue.date}</h1>',
            f"date heading for {issue.date}",
        )
        # ★ And with the h1 saying the date, the line under it that used to
        # say the date is saying it twice (0Z-E). `uc-issue__date` was the
        # masthead's dateline back when the h1 was the brand; the swap above
        # took its job. Removed here and **only here** — in the email the h1
        # is still "Urban Currents", because an email has no navigation
        # carrying the brand two lines up, and that dateline is the only
        # thing telling a reader which day they were sent.
        #
        # The anchor takes the newline and indent with it so the header does
        # not keep a blank line where the paragraph was. That makes it
        # sensitive to how the template is indented, which is the intended
        # trade: `_replace_once` turns a reindent into a failed build rather
        # than a silently unremoved line, and 0Z-D is the whole argument for
        # preferring the first.
        html = _replace_once(
            html,
            f'\n    <p class="uc-issue__date">{issue.date}</p>',
            "",
            f"dateline for {issue.date}",
        )
        path = target_dir / f"{issue.date}.html"
        path.write_text(html, encoding="utf-8", newline="\n")
        written.append(path)
    return written


def nav_links(root: str = "", latest: Optional[Issue] = None) -> list[dict]:
    """The navigation, in one place (phase 0Z, Z4).

    There are two renderers for this chrome and there have to be: the issue
    page's copy is injected by the site build because `preview.html.j2` is also
    the email, and an email must not carry links to a site it is not on. What
    there does **not** have to be is two lists of what the chrome contains —
    they had already drifted, with `uc-nav__lang` present in the templates and
    absent from the injected version.

    So the links come from here and both renderers read them.
    `test_nav_is_the_same_everywhere` fails if they diverge again.

    `Latest` points at the issue the home page leads with, not at the newest
    date in the archive: with `site.latest_skips_recent` those differ, and two
    places disagreeing about "latest" is the next bug (Z4).
    """
    target = f"{root}index.html"
    if latest is not None:
        target = f"{root}issues/{latest.date}.html"
    return [
        # The brand is a link now: it is the obvious way back to the landing
        # page and a reader will try it whether or not it works.
        {"label": "Urban Currents", "href": f"{root}index.html", "cls": "uc-nav__brand"},
        {"label": "Latest", "href": target, "cls": ""},
        {"label": "Archive", "href": f"{root}archive.html", "cls": ""},
    ]


def _issue_nav(
    previous: Optional[Issue],
    following: Optional[Issue],
    latest: Optional[Issue] = None,
) -> str:
    """Site navigation, injected rather than templated into the issue.

    The same markup is also written to `runs/` as a preview, where these links
    would point nowhere — so the chrome belongs to the site build. The *link
    list* comes from `nav_links`, which the templates use too.
    """
    parts = ['<nav class="uc-nav">']
    for link in nav_links("../", latest):
        cls = f' class="{link["cls"]}"' if link["cls"] else ""
        parts.append(f'<a{cls} href="{link["href"]}">{link["label"]}</a>')
    if previous:
        parts.append(
            f'<a class="uc-nav__prev" href="{previous.date}.html">&larr; {previous.date}</a>'
        )
    if following:
        parts.append(
            f'<a class="uc-nav__next" href="{following.date}.html">{following.date} &rarr;</a>'
        )
    parts.append('<span class="uc-nav__lang">EN</span>')
    parts.append("</nav>")
    return "\n".join(parts)


def _base_url() -> str:
    """`site.base_url` if set, otherwise empty so every link stays relative.

    An absolute URL invented before a domain exists is a link to nowhere printed
    in an email. Relative paths also mean the built site opens from the
    filesystem, which is how it is reviewed today.

    Set since 0X to `https://latentpublics.com/urban-currents` — a **sub-path**,
    which is why nothing here may emit a root-absolute `/issues/…`: it would
    resolve against the organisation's domain root and 404.
    """
    return (cfg("site.base_url", "") or "").rstrip("/")


def recent_days() -> int:
    """How many days back are still filling in (0Z, Z2). See the config."""
    return int(cfg("site.recent_days", 3) or 0)


def recent_cutoff(today: Optional[dt_date] = None) -> dt_date:
    """The oldest date that still counts as recent.

    A row is recent when its date is **on or after** this. Computed from
    today rather than from the newest issue: the claim is about arXiv's
    indexing lag now, not about how far the archive happens to reach.
    """
    today = today or dt_date.today()
    return today - timedelta(days=recent_days() - 1) if recent_days() > 0 else today + timedelta(days=1)


def is_published() -> bool:
    """Whether this site is meant to be found (phase 0X, X5).

    One value drives `robots.txt`, the `noindex` meta on every page, and
    whether the sitemap is advertised. See `config/pipeline.yaml: site.published`
    for why it is one and not three.
    """
    return bool(cfg("site.published", False))


SCHEMA = "https://schema.org"


def _periodical(base: str) -> dict:
    """The serial itself.

    ★ **`Periodical`, and deliberately not `Blog` or `Article`** (Launch A, A4).
    What this publishes is a dated issue on a fixed schedule with a stable
    scope, which is what a periodical is; a blog is a stream of posts and an
    article is one piece of writing. Getting this wrong is not cosmetic —
    structured data that overstates what a thing is, is worse than none, and
    the overstatement available here is the tempting one: calling our summaries
    scholarly articles. We do not write papers. We publish an issue that
    **mentions** other people's papers, and `mentions` is the property that
    says exactly that without claiming authorship, hosting, or endorsement.

    No `issn`. There isn't one, and inventing an identifier is the same fault
    as inventing an issue number.
    """
    node = {
        "@type": "Periodical",
        "name": "Urban Currents",
        "description": "A daily scan of urban data science research.",
        "publisher": {"@type": "Organization", "name": "Institute for Latent Publics"},
    }
    if base:
        node["@id"] = f"{base}/#periodical"
        node["url"] = f"{base}/"
    return node


def ld_home(stats: dict, base: str) -> dict:
    """`WebSite` plus the `Periodical` it publishes."""
    site = {
        "@context": SCHEMA,
        "@type": "WebSite",
        "name": "Urban Currents",
        "description": "A daily scan of urban data science research.",
        "inLanguage": "en",
        "publisher": {"@type": "Organization", "name": "Institute for Latent Publics"},
        "mainEntity": _periodical(base),
    }
    if base:
        site["url"] = f"{base}/"
    return site


def ld_issue(issue: Issue, items: dict, base: str, summary: str) -> dict:
    """One day, as an issue of the periodical.

    `mentions` carries the papers: third-party works this issue points at. They
    are `ScholarlyArticle` because that is what they are, with the author's
    identifier and the publisher's URL — and **not** `hasPart`, `citation` or
    `isBasedOn`, each of which would say something we are not entitled to say
    about a work we neither wrote nor host.

    Only `name`, `url` and `identifier`. No abstract: the same reason the API
    does not serve one.
    """
    node = {
        "@context": SCHEMA,
        "@type": "PublicationIssue",
        "name": f"Urban Currents {issue.date}",
        "datePublished": str(issue.date),
        "inLanguage": "en",
        "isPartOf": _periodical(base),
    }
    if summary:
        node["description"] = summary
    if base:
        node["url"] = f"{base}/issues/{issue.date}.html"
    mentions = []
    for key in issue.items:
        item = items.get(key)
        if item is None:
            continue
        paper = {"@type": "ScholarlyArticle", "name": item.bibliography.title}
        url = item.bibliography.primary_location.landing_page_url
        if url:
            paper["url"] = url
        if item.ids.doi:
            paper["identifier"] = item.ids.doi
        mentions.append(paper)
    if mentions:
        node["mentions"] = mentions
    return node


def ld_api(base: str, days: int) -> dict:
    """The API, as a `Dataset` with its endpoints as downloads.

    `Dataset` is accurate — a body of structured information about a topic —
    and it is the type an AI search actually looks for when deciding whether a
    site has a machine-readable original behind its prose.

    `license` points at the API page rather than at a CC BY URL, because the
    licence is **three licences** (ours, the sources', and one field under
    none of them) and a single `license` URL on the whole dataset would be the
    over-claim this property most invites.
    """
    node = {
        "@context": SCHEMA,
        "@type": "Dataset",
        "name": "Urban Currents issues",
        "description": (
            "Daily issues of Urban Currents as JSON: which papers were selected "
            "on which day, the state of each day, and the scores behind them."
        ),
        "creator": {"@type": "Organization", "name": "Institute for Latent Publics"},
        "isAccessibleForFree": True,
        "measurementTechnique": "Automated selection from arXiv and a journal whitelist",
    }
    if base:
        node["url"] = f"{base}/api.html"
        node["license"] = f"{base}/api.html"
        node["distribution"] = [
            {
                "@type": "DataDownload",
                "encodingFormat": "application/json",
                "name": name,
                "contentUrl": f"{base}/api/{path}",
            }
            for name, path in (
                ("Catalogue of every day", "index.json"),
                ("The newest issue", "latest.json"),
            )
        ]
    if days:
        node["distribution"] = (node.get("distribution") or []) + [
            {
                "@type": "DataDownload",
                "encodingFormat": "application/json",
                "name": "One issue, by date",
                "contentUrl": f"{base}/api/issues/{{date}}.json" if base else None,
            }
        ]
    return node


def _attr(value: str) -> str:
    """Escape a string for an HTML attribute.

    A paper title containing a double quote would otherwise close the attribute
    and spill the rest of the sentence into the markup as tags. Titles with
    quotes in them are not rare, and the `meta description` is built from them.
    """
    from xml.sax.saxutils import escape

    return escape(str(value), {'"': "&quot;"})


def head_extras(
    root: str = "",
    *,
    description: Optional[str] = None,
    path: Optional[str] = None,
    title: Optional[str] = None,
    og_type: str = "website",
    jsonld: Optional[dict] = None,
) -> str:
    """Everything in `<head>` that the site build owns, in one place.

    Built here rather than in the templates because the issue pages are
    rendered by `preview.html.j2`, which is also the email — and an email must
    not carry a `noindex`, a feed link, or a canonical URL relative to a site it
    is not on. Same reason the navigation and the webfonts are injected by the
    site build.

    Launch A put the rest of it — description, Open Graph, canonical and the
    structured data — through the same function, so that three page builders
    cannot end up with three ideas of what a page's description is.

    **Every string here is derived.** The callers compute `description`: the
    home page from `archive_stats`, an issue from `issue_summary`.
    `test_no_number_in_the_chrome_is_written_into_the_template` has kept that
    rule for the visible chrome since 0Z, and a `meta description` is the same
    sentence, read by someone who never opens the page.

    ★ **No image tags.** An `og:image` pointing at a file that does not exist
    is worse than no card at all: the preview renders as a broken box instead
    of as plain text. When there is an image there will be a tag.
    """
    base = _base_url()
    parts = [
        f'<link rel="alternate" type="application/atom+xml" '
        f'title="Urban Currents" href="{root}feed.xml">'
    ]
    if description:
        parts.append(f'<meta name="description" content="{_attr(description)}">')

    # Canonical and `og:url` need an absolute URL, and this deploy is a
    # **sub-path** — a root-absolute `/issues/…` would resolve against the
    # organisation's domain and 404 (see `_base_url`). With no base URL nothing
    # absolute is emitted at all, rather than a guess at one.
    canonical = f"{base}/{path}" if (base and path is not None) else None
    if canonical:
        parts.append(f'<link rel="canonical" href="{_attr(canonical)}">')
    if title:
        parts.append(f'<meta property="og:title" content="{_attr(title)}">')
        parts.append(f'<meta name="twitter:title" content="{_attr(title)}">')
    if description:
        parts.append(f'<meta property="og:description" content="{_attr(description)}">')
        parts.append(f'<meta name="twitter:description" content="{_attr(description)}">')
    if title or description:
        parts.append(f'<meta property="og:type" content="{_attr(og_type)}">')
        parts.append('<meta property="og:site_name" content="Urban Currents">')
        # `summary`, not `summary_large_image`: the large card wants an image.
        parts.append('<meta name="twitter:card" content="summary">')
    if canonical:
        parts.append(f'<meta property="og:url" content="{_attr(canonical)}">')

    if jsonld is not None:
        # ★ The only `<script>` on this site, and it executes nothing: a
        # `ld+json` block is a data island the parser hands to consumers rather
        # than to an interpreter. `test_the_whole_site_fetches_nothing` used to
        # ban `<script` outright as a proxy for "fetches nothing from anywhere
        # else"; it now bans every script that is not this type, which is what
        # it always meant.
        parts.append(
            '<script type="application/ld+json">'
            + "\n"
            + json.dumps(jsonld, indent=2, ensure_ascii=False)
            + "\n"
            + "</script>"
        )

    if not is_published():
        parts.append('<meta name="robots" content="noindex, nofollow">')
    return "\n".join(parts)


# PRD Q1b's criterion, not a measurement: the bar we said we would clear, which
# is specification and belongs in the code the way a threshold does. The
# measured value against it is read from the labels. `report.py:222` applies the
# same number to the same question; there is no config key for it because it is
# not a knob — moving it would be moving the goalposts, not tuning.
Q1B_TARGET = 0.7


def issue_summary(issue: Issue, items: Optional[dict[str, Item]] = None) -> str:
    """One sentence describing a day, for anywhere a day needs describing.

    Derived (0Z, Z1). A feed entry saying "a quiet day" over nine papers is the
    same false sentence as the archive row's, reaching further — a feed is read
    on other people's sites, and since Launch A this line is also the page's
    `meta description`, which is read by search engines and by whatever
    summarises the page for someone who never visits it. One definition, so the
    three cannot drift.
    """
    items = item_index() if items is None else items
    summary = issue.headline.line or (
        "A quiet day in urban data science." if issue.is_quiet
        else f"{len(issue.items)} papers; none cleared the headline bar."
        if issue.items else ""
    )
    lead = items.get(issue.headline.work_key or "")
    if lead and not summary:
        summary = lead.bibliography.title
    return summary


def selection_quality() -> Optional[dict]:
    """`precision@10` per entry path, measured, or nothing (Launch A, A2).

    The home page states this and states that it misses the 0.70 target we set,
    because the alternative is that somebody else notices first. Read from the
    labels rather than written into the template for the reason every other
    number on that page is: a figure typed into HTML is a figure that stops
    being true without anything failing.

    `None` when the labels are not in the checkout — the sentence is then left
    out rather than printed with a zero in it, which is the rule this project
    keeps having to restate.
    """
    try:
        from ..labeling import precision_at_k

        result = precision_at_k("relevance", 10)
    except Exception:
        return None
    by_source = result.get("by_source") or {}
    if not by_source:
        return None
    return {
        "k": result.get("k", 10),
        "n_labels": result.get("n_labels", 0),
        "days": result.get("days_labelled", 0),
        "target": Q1B_TARGET,
        "sources": {
            name: {
                "precision": d.get(f"precision_at_{result.get('k', 10)}"),
                "n_labels": d.get("n_labels", 0),
                "days": d.get("days", 0),
                "depth_holding_target": d.get("depth_holding_0.7"),
            }
            for name, d in sorted(by_source.items())
        },
    }


def build_feed(out: Optional[Path] = None) -> Path:
    """Atom — the only way to read this without giving us an address.

    It suits the service better than the mail does: no list, no unsubscribe,
    and no way for us to learn who is reading.
    """
    from xml.sax.saxutils import escape

    base = _base_url()
    issues = list(reversed(load_issues()))
    items = item_index()
    updated = f"{issues[0].date}T00:00:00Z" if issues else "1970-01-01T00:00:00Z"

    entries = []
    for issue in issues[:50]:
        link = f"{base}/issues/{issue.date}.html" if base else f"issues/{issue.date}.html"
        summary = issue_summary(issue, items)
        entries.append(
            "  <entry>\n"
            f"    <title>Urban Currents {issue.date}</title>\n"
            f'    <link href="{escape(link)}"/>\n'
            f"    <id>urn:urban-currents:issue:{issue.date}</id>\n"
            f"    <updated>{issue.date}T00:00:00Z</updated>\n"
            f"    <summary>{escape(summary)}</summary>\n"
            "  </entry>"
        )

    self_link = f'<link rel="self" href="{escape(base)}/feed.xml"/>' if base else ""
    alt_link = f'<link rel="alternate" href="{escape(base)}/"/>' if base else ""
    # RFC 4287 §4.1.1: a feed **must** contain an `author` unless every entry
    # carries one. Without it the document is not a valid Atom feed and
    # validators reject it, which is a poor way to be discovered (0X, X3).
    author = (
        "  <author>\n"
        "    <name>Urban Currents</name>\n"
        "  </author>\n"
    )
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        "  <title>Urban Currents</title>\n"
        "  <subtitle>A daily scan of urban data science</subtitle>\n"
        "  <id>urn:urban-currents:feed</id>\n"
        f"  <updated>{updated}</updated>\n"
        + author
        + f"  {self_link}\n"
        + (f"  {alt_link}\n" if alt_link else "")
        + "\n".join(entries)
        + "\n</feed>\n"
    )
    target = out or (paths.ROOT / "site" / "feed.xml")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(xml, encoding="utf-8", newline="\n")
    return target


def _inline_md(text: str) -> Markup:
    """`code` and **bold** in a short string, as HTML.

    The API document's field notes are written with backticks because that is
    how the rest of this repository writes a field name, and they were rendering
    as literal backticks on the page. Escaped **first** and then marked safe, so
    the conversion cannot turn a stray angle bracket in a field note into a tag.
    """
    import re as _re
    from markupsafe import escape as _escape

    out = str(_escape(text))
    out = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = _re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return Markup(out)


API_ENDPOINTS = (
    ("api/index.json", "Every day, newest first: date, state, how many papers, "
                       "the headline line, and the URL of the page. Days with no "
                       "issue are in here too — a gap would say nothing happened."),
    ("api/latest.json", "The newest issue in full, at an address that does not move."),
    ("api/issues/<date>.json", "One day in full. `<date>` is `YYYY-MM-DD`."),
)

# ★ Walked against the model in `test_api.py`, not maintained by hand. The two
# starred entries are the reason this section exists: both have had a name that
# disagreed with what they held, and a consumer cannot see that from the JSON.
API_FIELDS = (
    ("date", "The day the issue covers."),
    ("state", "`published`, `quiet`, `no_headline` or `not_seen`. ★ Derived here "
              "from the day's contents, not read from the stored `quiet_day` flag "
              "— that flag answers 'did anything clear the headline bar', and it "
              "is true on days that published nine papers. `quiet` here means "
              "nothing was published at all."),
    ("backfilled", "The issue was assembled after the fact from archived "
                   "candidates, because nobody was watching that morning. It is "
                   "part of the method, not a fault — but those days skipped "
                   "arXiv enrichment, so their contents differ slightly."),
    ("withheld", "The reason the live run that morning published nothing, when "
                 "there was one. A day can be both `withheld` and `backfilled`: "
                 "the run refused and the issue was built later. Both facts are "
                 "kept."),
    ("recent", "Our knowledge of this day is still settling — arXiv indexes "
               "about three days late. Papers posted around then can still "
               "appear in a *later* issue. Issues never change once published."),
    ("headline.present", "★ Whether there is a line in `headline.line`. It does "
                         "**not** mean a paper stood out: since 0Z-B the "
                         "threshold chooses the headline's *shape*, not whether "
                         "there is one. `headline.basis` records which."),
    ("headline.basis", "How the line was produced: `llm:lead` (written about one "
                       "paper), `llm:day` (written about the day), `:retry` if a "
                       "check rejected the first attempt, `fallback:<reason>` if "
                       "no attempt passed."),
    ("counts.unreadable", "Papers we could see existed and could not read, "
                          "because no source exposed an abstract. Published "
                          "because it is the one blind spot this pipeline "
                          "measures exactly."),
    ("scores.relevance", "The classifier's probability for the arXiv path. On "
                         "the journal path the paper entered because a journal "
                         "we track published it, and the classifier is not "
                         "consulted."),
    ("scores.headline", "What decides the headline's shape. A third of the "
                        "archive sits at exactly 0.44; treat small differences "
                        "as noise."),
    ("summary", "What the paper did and why it matters, written by a model from "
                "the abstract. **Not under the open licence** — see below."),
)


def build_api_docs(out: Optional[Path] = None) -> Path:
    """One page describing the JSON, its promises, and who owns what."""
    from .api import SCHEMA_VERSION, build_api  # noqa: F401  (version, not a rebuild)

    issues = load_issues()
    items = item_index()
    rows = archive_rows(issues, items)
    stats = archive_stats(rows, items)
    latest = latest_issue(issues)
    base = _base_url()

    env = _env()
    css = with_fonts(
        (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8"), "assets/fonts/"
    )
    description = (
        "Every Urban Currents issue as JSON: which papers were selected on which "
        "day, what state the day was in, and who owns which part of it."
    )
    html = env.get_template("api.html.j2").render(
        head_extras=Markup(head_extras(
            description=description,
            path="api.html",
            title="Urban Currents — API",
            jsonld=ld_api(base, len(rows)),
        )),
        nav=nav_links("", latest),
        stats=stats,
        endpoints=[{"path": p, "what": _inline_md(w)} for p, w in API_ENDPOINTS],
        fields=[{"name": n, "what": _inline_md(w)} for n, w in API_FIELDS],
        schema_version=SCHEMA_VERSION,
        # Measured after the deploy and written into the config, never guessed:
        # whether a browser may fetch these files cross-origin is the whole
        # question for anyone building on them (A1-2).
        cors_note=cfg("site.cors_note", "") or "",
        example={
            "date": str(latest.date) if latest else "",
            "url": (f"{base}/issues/{latest.date}.html" if base and latest else ""),
        },
        css=Markup(css),
        pipeline_version=PIPELINE_VERSION,
    )
    target = out or (paths.ROOT / "site" / "api.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8", newline="\n")
    return target


def build_llms_txt(out: Optional[Path] = None) -> Optional[Path]:
    """`llms.txt` — a **proposed convention, not a standard** (Launch A, A4).

    It has no RFC and no registry; it is a file some people have agreed to look
    for. That is a reason to write it plainly and to say so here, not a reason
    to skip it: the cost is a few hundred bytes and the benefit is that an
    assistant summarising this site finds the machine-readable original instead
    of scraping the prose.

    ★ Written **only when the site is published**, for the same reason the
    sitemap line is left out of `robots.txt` until then (0X, X5). A sitemap is
    an invitation to index. So is this — more so, because it is discovered by
    convention at a fixed path — and publishing an invitation next to
    `Disallow: /` states two intentions in one directory.
    """
    target = out or (paths.ROOT / "site" / "llms.txt")
    if not is_published():
        if target.exists():
            target.unlink()
        return None

    base = _base_url()
    issues = load_issues()
    latest = latest_issue(issues)
    rows = archive_rows(issues, item_index())
    stats = archive_stats(rows, item_index())

    def link(label: str, path: str, note: str) -> str:
        return f"- [{label}]({base}/{path}): {note}" if base else f"- {label} ({path}): {note}"

    lines = [
        "# Urban Currents",
        "",
        "> A daily scan of urban data science research, published by the Institute "
        "for Latent Publics. Papers are selected from arXiv by a classifier and "
        "from a journal whitelist; each day says what it looked at, what it "
        "published, and when it published nothing.",
        "",
        "Bibliography, authors and links come from the sources, never from a "
        "model. Where a measurement was not possible the line is left out rather "
        "than printed as a zero.",
        "",
        "## Machine-readable",
        "",
        link("API documentation", "api.html", "addresses, fields, licence, how to cite"),
        link("Catalogue", "api/index.json", f"every day ({stats['days']}), newest first"),
        link("Newest issue", "api/latest.json", "the current issue in full"),
        link("Atom feed", "feed.xml", "the last 50 issues"),
        "",
        "## Pages",
        "",
        link("Home", "index.html", "what this is, how papers are chosen, and how well"),
        link("Archive", "archive.html", "every issue, by month"),
    ]
    if latest is not None:
        lines.append(link("Latest issue", f"issues/{latest.date}.html", str(latest.date)))
    lines += [
        "",
        "## Licence",
        "",
        "The selection, scores and headlines are CC BY 4.0. Bibliographic records "
        "are the sources' under their own terms. The per-paper summaries are "
        "offered under no open licence — see the API documentation for why.",
        "",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return target


def build_sitemap(out: Optional[Path] = None) -> Path:
    from xml.sax.saxutils import escape

    base = _base_url()
    # Every page a reader can reach, including the month pages, which were
    # missing: `archive.html` is only the newest month and the others were
    # discoverable by a crawler following tabs but never advertised.
    rows = archive_rows()
    urls = (
        ["index.html", "archive.html", "api.html"]
        + [f"archive/{m['key']}.html" for m in _month_buckets(rows)[1:]]
        + [f"issues/{issue.date}.html" for issue in reversed(load_issues())]
    )
    body = "\n".join(
        f"  <url><loc>{escape(f'{base}/{u}' if base else u)}</loc></url>" for u in urls
    )
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{body}\n</urlset>\n"
    )
    target = out or (paths.ROOT / "site" / "sitemap.xml")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(xml, encoding="utf-8", newline="\n")
    return target


def build_robots(out: Optional[Path] = None) -> Path:
    """`Allow` or `Disallow`, from the one switch (0X, X5).

    While `site.published` is false the sitemap line is left out as well. A
    sitemap is an invitation to index, and publishing one next to
    `Disallow: /` states two different intentions in one directory.
    """
    base = _base_url()
    if not is_published():
        lines = ["User-agent: *", "Disallow: /"]
    else:
        lines = ["User-agent: *", "Allow: /"]
        if base:
            lines.append(f"Sitemap: {base}/sitemap.xml")
    target = out or (paths.ROOT / "site" / "robots.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return target


def build_site() -> dict[str, Any]:
    """Everything, in the order the links need."""
    return {
        # Assets first: the pages reference them, and a build that wrote the
        # HTML and not the fonts would look finished and render in Georgia.
        "assets": len(copy_assets()),
        "home": str(build_home()),
        "archive": [str(p) for p in build_archive()],
        "issues": len(build_issue_pages()),
        "feed": str(build_feed()),
        # The JSON and the page that documents it. Built after the pages so a
        # failure here cannot leave the site half-written, and before the
        # sitemap, which advertises `api.html`.
        "api": len(build_api()),
        "api_docs": str(build_api_docs()),
        "sitemap": str(build_sitemap()),
        "robots": str(build_robots()),
        # `None` until `site.published` — an invitation to index next to
        # `Disallow: /` states two intentions in one directory.
        "llms_txt": str(build_llms_txt() or "(unpublished)"),
        "base_url": _base_url() or "(relative)",
        # Printed so a person running `uc site` sees which of the two states
        # they just built, rather than having to open robots.txt (0X, X5).
        "published": is_published(),
    }
