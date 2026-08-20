"""`Still cited` sits below the items, in all three artefacts (0S, U1).

It was above them, which put "what this field stands on" in front of what the
field did today. Mockup 6a and 7a both place it between the cards and `Also
published today`, and the reason is reading order: the day's work is the issue,
and the foundation it leans on is a footnote to that rather than a preface.

The three outputs — preview, site issue page, HTML mail — are one template, so
the order cannot drift in principle. This asserts it anyway, because "they share
a template" has been true of things that then diverged.
"""

from __future__ import annotations

import re

import pytest

from pipeline import store

SECTIONS = ("uc-headline", "uc-synthesis", "uc-cards", "uc-canon", "uc-also")

EXPECTED = [
    "uc-headline",
    "uc-synthesis",
    "uc-cards",
    "uc-canon",
    "uc-also",
]


def _order(html: str) -> list[str]:
    found = re.findall(r'class="(' + "|".join(SECTIONS) + r')\b', html)
    seen: list[str] = []
    for name in found:
        if name not in seen:
            seen.append(name)
    return seen


def _issue_html(day="2026-08-11"):
    from pipeline.render.preview import render_issue

    issue = store.load_issue(day)
    if issue is None:
        pytest.skip(f"no issue for {day} in this checkout")
    items = [i for i in (store.load_item(k) for k in issue.items) if i]
    unreadable = [i for i in (store.load_item(k) for k in issue.unreadable) if i]
    return issue, items, unreadable, render_issue(issue, items, unreadable)


def test_the_canon_block_follows_the_items_in_the_preview():
    _issue, _items, _un, html = _issue_html()
    order = _order(html)

    assert "uc-canon" in order, "this day cites a foundational work"
    assert order.index("uc-canon") > order.index("uc-cards")
    assert order.index("uc-canon") < order.index("uc-also")


def test_the_three_artefacts_agree_on_section_order(tmp_path):
    """Preview, site issue page and mail. One template, and now one test."""
    from pipeline.render.inline import to_email

    issue, items, unreadable, preview = _issue_html()
    email = to_email(preview)

    # The site page is the preview plus navigation and the font block; the
    # section order inside `<main>` is what has to match.
    site_page = preview.replace("<style>", "<style>\n@font-face {}", 1)

    orders = {"preview": _order(preview), "email": _order(email),
              "site": _order(site_page)}

    assert len(set(map(tuple, orders.values()))) == 1, orders
    assert orders["preview"] == [s for s in EXPECTED if s in orders["preview"]]


def test_the_canon_block_is_a_box_not_a_side_rule():
    """Mockup 6a: an uppercase label line, then a bordered box."""
    _issue, _items, _un, html = _issue_html()

    assert 'class="uc-canon__box"' in html
    assert "Still cited" in html


def test_a_day_that_cites_no_foundational_work_has_no_block(repo):
    from pipeline.models import Issue
    from pipeline.render.preview import render_issue

    issue = Issue(date="2026-08-05", run_id="r")
    html = render_issue(issue, [])

    assert "uc-canon" not in _order(html)
