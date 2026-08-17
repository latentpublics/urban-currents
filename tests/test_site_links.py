"""Every internal link resolves, and no page fetches anything (phase 0k, X4).

Before this batch `site/` held a home page and an archive whose every row
pointed at a file that did not exist. The archive's job is to be walked, so a
broken link there is not cosmetic — it is the feature failing.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urldefrag

from pipeline import paths, store
from pipeline.models import Bibliography, Headline, Issue, Item, ScanMeta
from pipeline.render.site import build_design_review, build_site

EXTERNAL = re.compile(r"^(?:https?:|mailto:|#)")
HREF = re.compile(r'href="([^"]+)"')


def _seed(repo) -> None:
    for day in (11, 12):
        item = Item(
            work_key=f"arxiv:2608.{day:05d}",
            first_published=date(2026, 8, day),
            bibliography=Bibliography(title=f"Paper for the {day}th"),
        )
        store.save_item(item, today=date(2026, 8, day))
        store.save_issue(Issue(
            date=date(2026, 8, day),
            items=[item.work_key],
            headline=Headline(present=True, work_key=item.work_key, line="A line."),
            scan_meta=ScanMeta(items_published=1),
        ))


def _internal_links(html: str) -> list[str]:
    return [
        unquote(urldefrag(href)[0])
        for href in HREF.findall(html)
        if not EXTERNAL.match(href) and urldefrag(href)[0]
    ]


def test_no_internal_link_is_broken(repo):
    _seed(repo)
    build_site()
    root = paths.ROOT / "site"

    broken: list[str] = []
    for page in root.rglob("*.html"):
        for href in _internal_links(page.read_text(encoding="utf-8")):
            if not (page.parent / href).resolve().exists():
                broken.append(f"{page.relative_to(root)} -> {href}")
    assert not broken, f"broken internal links: {broken}"


def test_every_issue_has_a_page_and_the_archive_reaches_it(repo):
    _seed(repo)
    build_site()
    root = paths.ROOT / "site"

    for issue_date in ("2026-08-11", "2026-08-12"):
        assert (root / "issues" / f"{issue_date}.html").exists()

    archive = (root / "archive.html").read_text(encoding="utf-8")
    assert 'href="issues/2026-08-11.html"' in archive
    assert 'href="issues/2026-08-12.html"' in archive


def test_an_issue_page_uses_the_same_dom_as_the_preview(repo):
    """Phase 1 inherits one card component; a second template here would be a
    second component to keep in step."""
    from pipeline.render.preview import render_issue

    _seed(repo)
    build_site()
    issue = store.load_issue(date(2026, 8, 11))
    items = [store.load_item(k) for k in issue.items]

    preview = render_issue(issue, items)
    page = (paths.ROOT / "site" / "issues" / "2026-08-11.html").read_text(encoding="utf-8")

    for cls in ("uc-issue", "uc-card", "uc-card__meta", "uc-card__body", "uc-scanmeta"):
        assert cls in preview and cls in page
    assert page.count('class="uc-card"') == preview.count('class="uc-card"')


def test_the_whole_site_fetches_nothing(repo):
    _seed(repo)
    build_site()
    build_design_review()

    pages = list((paths.ROOT / "site").rglob("*.html")) + [
        paths.DOCS / "design-review.html"
    ]
    for page in pages:
        html = page.read_text(encoding="utf-8")
        for pattern in (r"<link\b", r"<script\b", r"<img\b", r"@font-face", r"@import"):
            assert not re.search(pattern, html, re.I), f"{page.name} matches {pattern}"


def test_links_stay_relative_without_a_base_url(repo):
    """A domain we do not have must not appear in a feed or an email."""
    _seed(repo)
    build_site()
    feed = (paths.ROOT / "site" / "feed.xml").read_text(encoding="utf-8")

    # The property is about *links*, not about the document: the Atom namespace
    # is itself an http:// URI and always will be.
    assert 'href="issues/2026-08-12.html"' in feed
    hrefs = re.findall(r'href="([^"]+)"', feed)
    assert hrefs, "the feed should link to its issues"
    assert not [h for h in hrefs if h.startswith(("http://", "https://"))]


def test_the_feed_lists_the_newest_issue_first(repo):
    _seed(repo)
    build_site()
    feed = (paths.ROOT / "site" / "feed.xml").read_text(encoding="utf-8")
    assert feed.index("2026-08-12") < feed.index("2026-08-11")
    assert feed.count("<entry>") == 2


def test_sitemap_and_robots_exist(repo):
    _seed(repo)
    build_site()
    root = paths.ROOT / "site"
    assert (root / "sitemap.xml").exists()
    assert "issues/2026-08-11.html" in (root / "sitemap.xml").read_text(encoding="utf-8")
    assert "User-agent" in (root / "robots.txt").read_text(encoding="utf-8")
