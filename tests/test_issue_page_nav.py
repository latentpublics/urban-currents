"""An issue page has a way back out of it (0Z-D).

The accident this file exists for: the site's navigation is injected into the
issue page by finding `<main class="uc-issue"` and putting the nav in front of
it. 0Z-A added a comment to `base.css.j2` that **quoted that anchor** while
explaining the injection. The stylesheet is inlined into `<head>`, so from that
commit the first match in the document was inside a CSS comment — and
`str.replace(..., 1)` put the entire navigation there, where it is a comment
and not a menu. Every link on the page was a DOI or an arXiv link. A reader who
opened an issue could not reach the home page, the archive, or the day either
side of it.

★ Why the existing tests all passed. `test_site_links.py` checks that the links
on a page resolve, and `test_nav_is_the_same_everywhere` checks that the two
renderers agree about what the nav contains. **Both are satisfied by an absent
menu**: nothing to resolve, and two identical lists. And a test asking whether
the HTML *contains* `uc-nav` would have passed too, because the markup was
there — in the wrong place. So these assertions are about **position**: in the
body, out of `<style>`, before the article.

It also holds the masthead's other half (0Z-E): the site says the date once
and the email still says it the way an email has to.

No network, no keys.
"""

from __future__ import annotations

from datetime import date

import pytest

from pipeline import paths, store
from pipeline.models import Bibliography, Headline, Issue, Item, PrimaryLocation, ScanMeta
from pipeline.render.site import _replace_once, build_issue_pages

DAYS = [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19)]


def _seed() -> None:
    for i, d in enumerate(DAYS):
        key = f"arxiv:2608.4000{i}"
        store.save_item(
            Item(
                work_key=key,
                first_published=d,
                bibliography=Bibliography(
                    title=f"A paper for {d}",
                    abstract="x",
                    primary_location=PrimaryLocation(
                        source_name="arXiv",
                        landing_page_url="https://arxiv.org/abs/2608.40000",
                    ),
                ),
            )
        )
        store.save_issue(
            Issue(
                date=d,
                items=[key],
                headline=Headline(present=True, work_key=key, line=f"A line for {d}."),
                scan_meta=ScanMeta(items_published=1, candidates_scanned=50, journals=40),
            )
        )


@pytest.fixture
def pages(repo):
    _seed()
    build_issue_pages()
    return {
        d: (paths.ROOT / "site" / "issues" / f"{d}.html").read_text(encoding="utf-8")
        for d in DAYS
    }


def _body(html: str) -> str:
    """Everything after the stylesheet closes — where a menu has to be."""
    return html[html.index("</style>") :]


# --------------------------------------------------------------------------
# The menu is there, and it is in the body
# --------------------------------------------------------------------------


def test_the_issue_page_has_a_navigation_at_all(pages):
    for d, html in pages.items():
        assert '<nav class="uc-nav">' in html, f"{d} has no navigation"


def test_the_navigation_is_not_inside_the_stylesheet(pages):
    """The whole bug in one assertion. `uc-nav` was present the entire time."""
    for d, html in pages.items():
        nav = html.index('<nav class="uc-nav">')
        assert nav > html.index("</style>"), f"{d}: the nav is inside <style>"
        assert nav < html.index('<main class="uc-issue" data-date='), (
            f"{d}: the nav is somewhere after the article"
        )


def test_a_reader_can_leave_the_page(pages):
    """Brand, Latest, Archive — the three ways out, on every issue."""
    for d, html in pages.items():
        body = _body(html)
        assert 'href="../index.html"' in body, f"{d}: no way back to the home page"
        assert 'href="../archive.html"' in body, f"{d}: no way to the archive"
        assert ">Latest<" in body, f"{d}: no way to the newest issue"
        assert ">Archive<" in body
        assert ">Urban Currents<" in body, f"{d}: the brand is not a link out"


def test_the_days_either_side_are_reachable(pages):
    """A middle day links both ways; the ends link the one way that exists."""
    first, middle, last = DAYS

    assert f'href="{first}.html"' in _body(pages[middle])
    assert f'href="{last}.html"' in _body(pages[middle])

    assert "uc-nav__prev" not in _body(pages[first]), "nothing precedes the oldest"
    assert f'href="{middle}.html"' in _body(pages[first])

    assert "uc-nav__next" not in _body(pages[last]), "nothing follows the newest"
    assert f'href="{middle}.html"' in _body(pages[last])


def test_the_date_heading_survived_the_fix(pages):
    """0Z's Z6 is deliberate and stays: the h1 is the day, not the brand."""
    for d, html in pages.items():
        assert f'<h1 class="uc-issue__title">{d}</h1>' in html
        assert '<h1 class="uc-issue__title">Urban Currents</h1>' not in html


# --------------------------------------------------------------------------
# The masthead says the date once — on the site, and only on the site
# --------------------------------------------------------------------------


def _masthead(html: str) -> str:
    start = html.index('<header class="uc-issue__masthead">')
    return html[start : html.index("</header>", start)]


def test_the_site_masthead_says_the_date_once(pages):
    """Z6 swapped the h1 to the date and left the line that used to carry it,
    so the header printed the same day twice (0Z-E)."""
    for d, html in pages.items():
        head = _masthead(html)
        assert head.count(str(d)) == 1, f"{d} appears {head.count(str(d))} times"
        assert "uc-issue__date" not in head, "the dateline is the h1's job here"


def test_the_email_keeps_its_brand_and_its_dateline(repo):
    """★ The reason this is removed in the build and not in the template.

    An email has no navigation carrying the brand two lines above the heading,
    so its h1 stays "Urban Currents" — and then the dateline is the only thing
    telling a reader which day they were sent. Same template, two outputs, and
    only one of them is allowed to lose the line.
    """
    from pipeline.render.preview import render_issue

    _seed()
    d = DAYS[1]
    issue = store.load_issue(d)
    items = [store.load_item(k) for k in issue.items]
    head = _masthead(render_issue(issue, [i for i in items if i], []))

    assert '<h1 class="uc-issue__title">Urban Currents</h1>' in head
    assert f'<p class="uc-issue__date">{d}</p>' in head


def test_the_dateline_rule_is_still_in_the_stylesheet():
    """It is unused on the site and must not be tidied away: one stylesheet
    serves both outputs, and the email is the one that still uses it."""
    import pipeline.render
    from pathlib import Path

    css = (Path(pipeline.render.__file__).parent / "templates" / "base.css.j2").read_text(
        encoding="utf-8"
    )
    assert ".uc-issue__date {" in css


def test_removing_a_dateline_that_is_not_there_raises(pages):
    """0Z-D's rule, applied to the line this batch removes: if the template
    stops emitting it, the build says so instead of quietly not removing it."""
    with pytest.raises(RuntimeError, match="found 0"):
        _replace_once("<header></header>", '<p class="uc-issue__date">x</p>', "", "a test")


# --------------------------------------------------------------------------
# And the substitution that lost it is no longer allowed to be quiet
# --------------------------------------------------------------------------


def test_a_missing_anchor_raises(pages):
    with pytest.raises(RuntimeError, match="found 0"):
        _replace_once("<p>nothing here</p>", "<main>", "x", "a test")


def test_an_ambiguous_anchor_raises(pages):
    """The actual failure: the anchor was still there, twice, and the first one
    was in a comment. Picking one silently is what cost a day of navigation."""
    doubled = '<style>/* see <main class="uc-issue"> */</style><main class="uc-issue" id="real">'
    with pytest.raises(RuntimeError, match="found 2"):
        _replace_once(doubled, '<main class="uc-issue"', "x", "a test")


def test_the_stylesheet_does_not_quote_the_anchor():
    """Belt and braces. `_replace_once` would now shout, but the comment that
    caused this is the kind of thing that gets written again."""
    import pipeline.render
    from pathlib import Path

    css = (Path(pipeline.render.__file__).parent / "templates" / "base.css.j2").read_text(
        encoding="utf-8"
    )
    assert '<main class="uc-issue"' not in css, (
        "this file is inlined into the issue page's head, so an anchor quoted "
        "here is matched before the real tag"
    )
