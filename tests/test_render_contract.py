"""What every generated page must be true of (phase 0j final verification).

Three properties, each of which would be easy to lose in a later change and
hard to notice: the render is a pure function of `content/`, no page asks the
network for anything, and the same input gives byte-identical output.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from pipeline import paths, store
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
    re.compile(r"<link\b", re.I),
    re.compile(r"<script\b", re.I),
    re.compile(r"<img\b", re.I),
    re.compile(r"<iframe\b", re.I),
    re.compile(r"@font-face", re.I),
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
        "archive": build_archive().read_text(encoding="utf-8"),
        "design-review": build_design_review().read_text(encoding="utf-8"),
    }


def test_no_page_fetches_anything(repo):
    for name, html in _pages(repo).items():
        for pattern in FETCHING:
            assert not pattern.search(html), f"{name} would fetch: {pattern.pattern}"


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
