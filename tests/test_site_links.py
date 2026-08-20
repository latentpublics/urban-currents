"""Every internal link resolves, and no page fetches anything (phase 0k, X4).

Before this batch `site/` held a home page and an archive whose every row
pointed at a file that did not exist. The archive's job is to be walked, so a
broken link there is not cosmetic — it is the feature failing.
"""

from __future__ import annotations

import re
from datetime import date
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
        # The property is **"fetches nothing from anywhere else"**, and the
        # original list checked for `@font-face` as a proxy for it. Since 0R
        # the site self-hosts three OFL faces out of `site/assets/fonts/`,
        # which is same-origin and makes no request off the box — so the
        # check now looks for what it always meant: a URL pointing
        # somewhere else. `preview.html` and `email.html` are covered by
        # their own test below and still carry no `@font-face` at all.
        for pattern in (r'<link[^>]*rel=["\']?(stylesheet|preload|prefetch|icon|manifest)',
                        r'<link[^>]*href=["\']?https?:',
                        r"<script\b", r"<img\b", r"@import",
                        r"url\(\s*['\"]?https?:", r"fonts\.googleapis",
                        r"fonts\.gstatic"):
            assert not re.search(pattern, html, re.I), f"{page.name} matches {pattern}"

        # And every local `url(...)` has to resolve, or the page renders in
        # the fallback face while every check above still passes — which is
        # exactly the failure this batch was called to fix.
        for ref in re.findall(r'url\("([^"]+)"\)', html):
            assert (page.parent / ref).resolve().exists(), f"{page.name}: {ref}"


def test_links_stay_relative_without_a_base_url(repo, monkeypatch):
    """A domain we do not have must not appear in a feed or an email.

    Still true, and now stated as the conditional it always was: with
    `site.base_url` empty every link is relative. 0X gave the project a domain,
    so this pins the *other* branch — the one an email and a filesystem copy
    still take — and `test_the_feed_is_absolute_under_a_base_url` pins the new
    one.
    """
    from pipeline.render import site as site_mod

    monkeypatch.setattr(
        site_mod, "cfg",
        lambda k, d=None: "" if k == "site.base_url" else (False if k == "site.published" else d),
    )
    _seed(repo)
    build_site()
    feed = (paths.ROOT / "site" / "feed.xml").read_text(encoding="utf-8")

    # The property is about *links*, not about the document: the Atom namespace
    # is itself an http:// URI and always will be.
    assert 'href="issues/2026-08-12.html"' in feed
    hrefs = re.findall(r'href="([^"]+)"', feed)
    assert hrefs, "the feed should link to its issues"
    assert not [h for h in hrefs if h.startswith(("http://", "https://"))]


def test_the_feed_is_absolute_under_a_base_url(repo):
    """And with a domain, the feed must be absolute — it is read elsewhere.

    A relative `href` in a feed resolves against whatever reader is displaying
    it, which is not this site (0X, X2/X3).
    """
    _seed(repo)
    build_site()
    feed = (paths.ROOT / "site" / "feed.xml").read_text(encoding="utf-8")

    base = "https://latentpublics.com/urban-currents"
    assert f'href="{base}/issues/2026-08-12.html"' in feed
    assert f'<link rel="self" href="{base}/feed.xml"/>' in feed
    entry_links = re.findall(r'<link href="([^"]+)"/>', feed)
    assert entry_links and all(h.startswith(base) for h in entry_links), entry_links


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
