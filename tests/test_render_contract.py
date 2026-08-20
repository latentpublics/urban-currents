"""What every generated page must be true of (phase 0j final verification).

Three properties, each of which would be easy to lose in a later change and
hard to notice: the render is a pure function of `content/`, no page asks the
network for anything, and the same input gives byte-identical output.
"""

from __future__ import annotations

import re
from datetime import date

from pipeline import store
from pipeline.models import (
    Bibliography,
    Headline,
    Issue,
    Item,
    PrimaryLocation,
    ScanMeta,
    SummaryEn,
)
from pipeline.render.inline import to_email
from pipeline.render.preview import render_issue
from pipeline.render.site import build_archive, build_design_review, build_home

# `href` is not a request: a link is followed only if a reader clicks it. What
# must not appear is anything the client fetches on its own.
FETCHING = (
    # `<link>` was on this list as a proxy for "pulls a resource", and since 0X
    # the pages carry one that does not: `<link rel="alternate"
    # type="application/atom+xml" href="feed.xml">`, which is a discovery hint
    # a client follows only if the reader asks for the feed. Same reasoning as
    # `@font-face` above — the pattern is narrowed to what it always meant
    # rather than the page being changed to fit the proxy. The kinds of `link`
    # that *do* fetch are named, and `test_the_only_link_element_is_the_feed`
    # keeps the narrowing from quietly admitting a stylesheet later.
    re.compile(r'<link[^>]*rel=["\']?(stylesheet|preload|prefetch|icon|manifest)', re.I),
    re.compile(r'<link[^>]*href=["\']?https?:', re.I),
    re.compile(r"<script\b", re.I),
    re.compile(r"<img\b", re.I),
    re.compile(r"<iframe\b", re.I),
    # Was `@font-face`. The site self-hosts three OFL faces from
    # `site/assets/fonts/` since 0R — same origin, no request off the box —
    # so the pattern is now the thing it always stood for: a URL somewhere
    # else. `preview.html` and `email.html` still carry no `@font-face`,
    # which `test_the_preview_is_one_self_contained_file` asserts directly.
    re.compile(r"url\(\s*['\"]?https?:", re.I),
    re.compile(r"fonts\.(googleapis|gstatic)", re.I),
    re.compile(r"url\(\s*['\"]?https?:", re.I),
    re.compile(r"@import", re.I),
)


def _seed(repo) -> tuple[Issue, list[Item]]:
    items = []
    for i in range(3):
        it = Item(
            work_key=f"arxiv:2608.{i:05d}",
            first_published=date(2026, 8, 11),
            bibliography=Bibliography(
                title=f"Paper {i}",
                abstract="x",
                primary_location=PrimaryLocation(
                    source_name="arXiv",
                    landing_page_url=f"https://arxiv.org/abs/2608.{i:05d}",
                ),
            ),
        )
        it.summary.en = SummaryEn(what=f"What {i}.", why=f"Why {i}.")
        it.badges = ["data"]
        store.save_item(it, today=date(2026, 8, 11))
        items.append(it)
    issue = Issue(
        date=date(2026, 8, 11),
        items=[i.work_key for i in items],
        headline=Headline(present=True, work_key=items[0].work_key, line="A line."),
        scan_meta=ScanMeta(items_published=len(items)),
    )
    store.save_issue(issue)
    return issue, items


def _pages(repo) -> dict[str, str]:
    issue, items = _seed(repo)
    web = render_issue(issue, items)
    return {
        "preview": web,
        "email": to_email(web),
        "home": build_home().read_text(encoding="utf-8"),
        "archive": build_archive()[0].read_text(encoding="utf-8"),
        "design-review": build_design_review().read_text(encoding="utf-8"),
    }


def test_no_page_fetches_anything(repo):
    for name, html in _pages(repo).items():
        for pattern in FETCHING:
            assert not pattern.search(html), f"{name} would fetch: {pattern.pattern}"


def test_the_only_link_element_is_the_feed(repo):
    """The narrowing above, held in place.

    `FETCHING` no longer bans every `<link>`, so this states what the pages are
    allowed to contain instead: one same-origin `rel="alternate"` per page and
    nothing else wearing that tag.
    """
    for name, html in _pages(repo).items():
        for tag in re.findall(r"<link[^>]*>", html, re.I):
            assert 'rel="alternate"' in tag, f"{name}: unexpected {tag}"
            assert "atom+xml" in tag, f"{name}: unexpected {tag}"
            assert "http" not in tag.split("href=")[1][:6], f"{name}: off-origin {tag}"


def test_paper_links_are_still_there(repo):
    """The rule is about requests, not about links. A card without its link is
    a card that cannot be followed to the paper."""
    pages = _pages(repo)
    assert "href=" in pages["preview"]


def test_the_render_is_a_pure_function_of_content(repo):
    """Same content, same bytes. A timestamp in the output would break this and
    would also make every rebuild look like a change in git."""
    issue, items = _seed(repo)
    first = render_issue(issue, items)
    second = render_issue(issue, items)
    assert first == second

    home_a = build_home().read_text(encoding="utf-8")
    home_b = build_home().read_text(encoding="utf-8")
    assert home_a == home_b


def test_no_page_carries_a_wall_clock_time(repo):
    """`datetime.now()` in a rendered page is the usual way purity is lost."""
    today = date.today()
    stamps = (
        re.compile(r"\d{2}:\d{2}:\d{2}"),          # a clock time
        re.compile(re.escape(today.isoformat()) + r"T"),  # an ISO timestamp for now
    )
    for name, html in _pages(repo).items():
        body = html[html.index("<body>"):] if "<body>" in html else html
        for pattern in stamps:
            assert not pattern.search(body), f"{name} carries a wall-clock time"


def test_every_page_has_exactly_one_h1(repo):
    pages = _pages(repo)
    for name in ("preview", "home", "archive"):
        assert pages[name].count("<h1") == 1, f"{name} has {pages[name].count('<h1')} h1s"
