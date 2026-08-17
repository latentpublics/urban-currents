"""One template, two outputs (phase 0j, W6).

The claim "web = email" is only worth making if it is structural. These tests
are what makes it structural: the email is derived from the same render, and its
*content* is compared against the web page word for word. Form may differ;
nothing else may.
"""

from __future__ import annotations

import re
from datetime import date
from html.parser import HTMLParser

from pipeline.models import Bibliography, Headline, Issue, Item, ScanMeta, SummaryEn
from pipeline.render.inline import (
    UnsupportedSelector,
    custom_properties,
    inline_css,
    parse_rules,
    resolve_vars,
    to_email,
)
from pipeline.render.preview import email_subject, render_issue


class _Text(HTMLParser):
    """Visible text and card structure, with all markup removed."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cards = 0
        self.titles: list[str] = []
        self.text: list[str] = []
        self._in_title = False
        self._in_skip = False

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "")
        if tag in ("style", "script"):
            self._in_skip = True
        if "uc-card" == classes:
            self.cards += 1
        if "uc-card__title" in classes:
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self._in_skip = False
        if tag == "h2":
            self._in_title = False

    def handle_data(self, data):
        if self._in_skip:
            return
        stripped = data.strip()
        if not stripped:
            return
        self.text.append(stripped)
        if self._in_title:
            self.titles.append(stripped)


def _item(n: int) -> Item:
    it = Item(
        work_key=f"arxiv:2608.{n:05d}",
        first_published=date(2026, 8, 11),
        bibliography=Bibliography(title=f"Paper {n} about cities", abstract="x"),
    )
    it.summary.en = SummaryEn(
        what=f"What {n} found.", why=f"Why {n} matters.", caveats=f"Caveat {n}."
    )
    it.badges = ["data", "preprint"]
    return it


def _issue(items: list[Item]) -> Issue:
    return Issue(
        date=date(2026, 8, 11),
        items=[i.work_key for i in items],
        headline=Headline(present=True, work_key=items[0].work_key, line="A full sentence here."),
        scan_meta=ScanMeta(items_published=len(items), candidates_scanned=100),
    )


def test_the_email_says_exactly_what_the_page_says(repo):
    """Form may differ. Content may not differ by one character."""
    items = [_item(i) for i in range(3)]
    issue = _issue(items)

    web = render_issue(issue, items)
    mail = to_email(web)

    a, b = _Text(), _Text()
    a.feed(web)
    b.feed(mail)

    assert a.cards == b.cards == 3
    assert a.titles == b.titles
    assert a.text == b.text


def test_the_email_carries_no_stylesheet(repo):
    items = [_item(0)]
    mail = to_email(render_issue(_issue(items), items))
    assert "<style" not in mail
    assert "var(--" not in mail
    assert "@media" not in mail


def test_every_card_still_carries_its_styling_inline(repo):
    items = [_item(0)]
    mail = to_email(render_issue(_issue(items), items))
    card = re.search(r'<article class="uc-card"[^>]*>', mail).group(0)
    assert "style=" in card
    assert "display: flex" in card


def test_font_stacks_survive_the_style_attribute(repo):
    """A double quote inside a font stack closes the attribute early.

    The stylesheet writes `font-family: "Source Serif 4", Georgia`; pasted
    verbatim into a double-quoted attribute the rest becomes bogus markup.
    """
    items = [_item(0)]
    mail = to_email(render_issue(_issue(items), items))
    assert '"Source Serif 4"' not in mail
    assert "'Source Serif 4'" in mail
    # And the document still parses into the same number of cards.
    p = _Text()
    p.feed(mail)
    assert p.cards == 1


def test_media_queries_and_hover_rules_are_dropped():
    rules = parse_rules("""
      .a { color: red; }
      .a:hover { color: blue; }
      @media (max-width: 640px) { .a { color: green; } }
      .b::before { content: "x"; }
    """)
    assert rules == [(".a", "color: red;")]


def test_custom_properties_resolve_to_values():
    props = custom_properties(":root { --uc-ink: #16181d; --uc-bg: #fdfdfc; }")
    assert props["--uc-ink"] == "#16181d"
    assert resolve_vars("color: var(--uc-ink);", props) == "color: #16181d;"
    # An unknown name is left alone rather than blanked: a declaration that
    # silently loses its value is worse than one a client ignores.
    assert resolve_vars("color: var(--nope);", props) == "color: var(--nope);"


def test_an_unknown_selector_shape_raises_rather_than_passing_through():
    """A new selector must surface as a test failure, not as a subtly wrong email."""
    try:
        inline_css("<style>.a[data-x] { color: red; }</style><p class='a'>hi</p>")
    except UnsupportedSelector:
        return
    raise AssertionError("expected UnsupportedSelector")


def test_the_subject_line_rule(repo):
    items = [_item(i) for i in range(3)]
    subject = email_subject(_issue(items))
    assert subject.startswith("Urban Currents 2026-08-11 — 3 papers")
    assert "A full sentence here" in subject

    quiet = Issue(date=date(2026, 8, 12), items=[], quiet_day=True)
    assert email_subject(quiet).endswith("a quiet day")
