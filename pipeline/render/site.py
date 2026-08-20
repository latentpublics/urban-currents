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

from datetime import date
from pathlib import Path
from typing import Any, Optional

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
    """One row per issue **and per day we could not see**, newest first.

    A day with no issue is not a gap in the list. A gap would read as "nothing
    happened", which is exactly the claim we could not make — so it gets a row
    saying the sources did not answer, drawn differently from a quiet day.
    Same grey chip for both would undo the whole outcome model.
    """
    from ..outcome import NOT_PUBLISHED, all_logs

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
            "missing": False,
            "reason": None,
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
        })

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


def spark_bars(rows: list[dict], today: Optional[str] = None) -> list[dict]:
    """Mockup 5a's fourteen bars, oldest on the left.

    Height is the day's published count against the tallest day in the window,
    with a floor so a quiet day is still a visible mark rather than nothing —
    **a bar of zero height would say "no day here", and there was a day here.**

    Three kinds, and they are not three shades of the same claim:

      `published`  an issue went out
      `quiet`      we looked and there was little to publish
      `missing`    we could not see the day at all

    `missing` is drawn like `quiet` in the mockup, which has no such day in it.
    It gets its own class here so the two can never be read as one; a day nobody
    could see is not a day with nothing in it, and the whole outcome model rests
    on that distinction.
    """
    window = list(reversed(rows))  # oldest first, the way a time axis reads
    tallest = max([r["published"] for r in window] or [1]) or 1
    bars = []
    for r in window:
        if r.get("missing"):
            kind, note = "missing", f"not seen — {r.get('reason') or 'the sources did not answer'}"
        elif r["quiet"] or not r["published"]:
            kind, note = "quiet", "a quiet day"
        else:
            kind, note = "published", f"{r['published']} items"
        if today and r["date"] == today:
            kind = "today"
        height = max(8, round(100 * r["published"] / tallest)) if r["published"] else 12
        bars.append({
            "date": r["date"],
            "kind": kind,
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

    latest = next((i for i in reversed(issues) if i.items), None)
    lead_item = items.get(latest.headline.work_key or "") if latest else None

    window = rows[:HOME_ARCHIVE_DAYS]
    latest_row = next((r for r in window if latest and r["date"] == str(latest.date)), None)
    bars = spark_bars(window, today=str(latest.date) if latest else None)

    env = _env()
    css = with_fonts(
        (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8"), "assets/fonts/"
    )
    html = env.get_template("home.html.j2").render(
        head_extras=Markup(head_extras()),
        stats=stats,
        rows=rows[:HOME_ARCHIVE_DAYS],
        max_items=max([r["published"] for r in rows] or [1]),
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
        html = env.get_template("archive.html.j2").render(
            head_extras=Markup(head_extras(root)),
            stats=stats,
            month=month,
            tabs=_month_tabs(months, month["key"], root),
            root=root,
            max_items=max([r["published"] for r in rows] or [1]),
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
    for i, issue in enumerate(issues):
        day_items = [items[k] for k in issue.items if k in items]
        unreadable = [items[k] for k in issue.unreadable if k in items]
        html = render_issue(issue, day_items, unreadable)

        previous = issues[i - 1] if i > 0 else None
        following = issues[i + 1] if i + 1 < len(issues) else None
        html = html.replace(
            '<main class="uc-issue"',
            _issue_nav(previous, following) + '\n<main class="uc-issue"',
            1,
        )
        # The site copy gets the webfonts; `runs/*/preview.html` does not. Same
        # DOM, same CSS, one extra at-rule block — the preview's whole point is
        # that it is a single self-contained file, and the site's is that it
        # looks like the mockup.
        html = html.replace(
            "<style>", "<style>\n" + site_font_css("../assets/fonts/"), 1
        )
        # The feed link and, while unpublished, the `noindex`. Injected for the
        # same reason the navigation is: this markup belongs to the site, and
        # the identical DOM is also written to `runs/` as an email.
        html = html.replace("</head>", head_extras("../") + "\n</head>", 1)
        path = target_dir / f"{issue.date}.html"
        path.write_text(html, encoding="utf-8", newline="\n")
        written.append(path)
    return written


def _issue_nav(previous: Optional[Issue], following: Optional[Issue]) -> str:
    """Site navigation, injected rather than templated into the issue.

    The same markup is also written to `runs/` as a preview, where these links
    would point nowhere — so the chrome belongs to the site build.
    """
    parts = [
        '<nav class="uc-nav">',
        '<span class="uc-nav__brand">Urban Currents</span>',
        '<a href="../index.html">Today</a>',
        '<a href="../archive.html">Archive</a>',
    ]
    if previous:
        parts.append(
            f'<a class="uc-nav__prev" href="{previous.date}.html">&larr; {previous.date}</a>'
        )
    if following:
        parts.append(
            f'<a class="uc-nav__next" href="{following.date}.html">{following.date} &rarr;</a>'
        )
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


def is_published() -> bool:
    """Whether this site is meant to be found (phase 0X, X5).

    One value drives `robots.txt`, the `noindex` meta on every page, and
    whether the sitemap is advertised. See `config/pipeline.yaml: site.published`
    for why it is one and not three.
    """
    return bool(cfg("site.published", False))


def head_extras(root: str = "") -> str:
    """The two `<head>` lines that are the same on every page.

    Built here rather than in the templates because the issue pages are
    rendered by `preview.html.j2`, which is also the email — and an email must
    not carry a `noindex` or a feed link relative to a site it is not on. Same
    reason the navigation and the webfonts are injected by the site build.
    """
    parts = [
        f'<link rel="alternate" type="application/atom+xml" '
        f'title="Urban Currents" href="{root}feed.xml">'
    ]
    if not is_published():
        parts.append('<meta name="robots" content="noindex, nofollow">')
    return "\n".join(parts)


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
        summary = issue.headline.line or (
            "A quiet day in urban data science." if issue.quiet_day else ""
        )
        lead = items.get(issue.headline.work_key or "")
        if lead and not summary:
            summary = lead.bibliography.title
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


def build_sitemap(out: Optional[Path] = None) -> Path:
    from xml.sax.saxutils import escape

    base = _base_url()
    urls = ["index.html", "archive.html"] + [
        f"issues/{issue.date}.html" for issue in reversed(load_issues())
    ]
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
        "sitemap": str(build_sitemap()),
        "robots": str(build_robots()),
        "base_url": _base_url() or "(relative)",
        # Printed so a person running `uc site` sees which of the two states
        # they just built, rather than having to open robots.txt (0X, X5).
        "published": is_published(),
    }
