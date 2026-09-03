"""The archive row shows the day's headline, not one paper's title (1A, A).

`site.py` looked up `issue.headline.work_key`, loaded that Item and printed
`bibliography.title`. So every row in the archive — and every row in the home
strip, which shares the same template — showed **the title of one paper** while
the sentence written about the day sat unused in `issue.headline.line`. A
reader scanning the list was reading a bibliography rather than a record of
days, and the one place the headline sentence did appear (the home hero) made
the two look like different things about different days.

Three states, and the third is why this file is longer than the change:

  1. A day with a headline shows the headline.
  2. A day that published papers and had none clear the bar has no sentence.
     It falls back to the representative paper's title — recorded since 0Z-A
     even below the threshold — and **marks it**, because an unmarked title in
     that column is indistinguishable from a headline the day did not have.
  3. A day with neither shows nothing. Every such day in the archive is
     backfilled or predates 0Z-A. The alternative is promoting `items[0]`,
     which is a collection order, not a judgement about the day.

No network, no keys.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from pipeline import paths, store
from pipeline.models import Bibliography, Headline, Issue, Item, PrimaryLocation, ScanMeta
from pipeline.render.site import archive_rows, build_archive, build_home

TITLE = "Ridership recovery on four metro systems after a fare change"
LINE = "Fare changes moved ridership on four metro systems, and a cycling study"


def _item(key: str, title: str = TITLE) -> Item:
    it = Item(
        work_key=key,
        first_published=date(2026, 8, 18),
        bibliography=Bibliography(
            title=title,
            abstract="x",
            primary_location=PrimaryLocation(
                source_name="arXiv", landing_page_url=f"https://arxiv.org/abs/{key[-7:]}"
            ),
        ),
    )
    store.save_item(it)
    return it


def _issue(d: date, keys: list[str], *, line: str | None, work_key: str | None) -> Issue:
    issue = Issue(
        date=d,
        items=sorted(keys),
        headline=Headline(present=bool(line), work_key=work_key, line=line),
        scan_meta=ScanMeta(items_published=len(keys), candidates_scanned=100, journals=96),
    )
    store.save_issue(issue)
    return issue


@pytest.fixture
def three_days(repo):
    """One day of each headline state, in one archive."""
    _item("arxiv:2608.40001")
    _issue(date(2026, 8, 18), ["arxiv:2608.40001"], line=LINE, work_key="arxiv:2608.40001")
    _issue(date(2026, 8, 17), ["arxiv:2608.40001"], line=None, work_key="arxiv:2608.40001")
    _issue(date(2026, 8, 16), ["arxiv:2608.40001"], line=None, work_key=None)
    build_archive()
    build_home()
    return (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")


def _row(html: str, d: str) -> str:
    m = re.search(rf'<li class="uc-row[^"]*" data-date="{d}">(.*?)</li>', html, re.S)
    assert m, f"no row for {d}"
    return m.group(1)


def _lead(row: str) -> str:
    m = re.search(r'<span class="uc-row__lead"[^>]*>(.*?)</span>\s*(?:<span class="uc-row__note"|$)',
                  row, re.S)
    return (m.group(1) if m else "").strip()


# --------------------------------------------------------------------------
# 1. The row data carries the sentence at all
# --------------------------------------------------------------------------


def test_the_row_carries_the_headline_line(three_days):
    rows = {r["date"]: r for r in archive_rows()}
    assert rows["2026-08-18"]["lead_line"] == LINE
    # The title is still carried — it is the fallback — but it is no longer
    # what a day with a headline shows.
    assert rows["2026-08-18"]["lead_title"] == TITLE
    assert rows["2026-08-17"]["lead_line"] is None
    assert rows["2026-08-17"]["lead_title"] == TITLE
    assert rows["2026-08-16"]["lead_line"] is None
    assert rows["2026-08-16"]["lead_title"] == ""


# --------------------------------------------------------------------------
# 2. And the page prints it
# --------------------------------------------------------------------------


def test_a_day_with_a_headline_shows_the_headline_not_the_paper(three_days):
    row = _row(three_days, "2026-08-18")
    assert LINE in row, "the headline sentence is not on the row"
    assert TITLE not in _lead(row), (
        "the row is still showing one paper's title where the day's sentence belongs"
    )


def test_a_day_with_no_headline_falls_back_to_the_title_and_marks_it(three_days):
    row = _row(three_days, "2026-08-17")
    lead = _lead(row)
    assert TITLE in lead, "nothing to show, and there was a representative paper"
    assert "uc-row__lead--title" in lead, (
        "an unmarked title here reads as a headline the day did not have"
    )
    # The words carry it too. The class is the second signal, never the only
    # one — base.css refuses meaning carried by styling alone.
    assert "no headline" in row


def test_a_day_with_neither_shows_nothing_rather_than_inventing_a_lead(three_days):
    lead = _lead(_row(three_days, "2026-08-16"))
    assert TITLE not in lead
    assert lead == "", f"something was invented for a day with no representative: {lead!r}"


def test_the_tooltip_matches_what_is_displayed(three_days):
    """A `title` attribute that disagrees with the text is a second string."""
    for d, expected in (("2026-08-18", LINE), ("2026-08-17", TITLE)):
        row = _row(three_days, d)
        m = re.search(r'<span class="uc-row__lead" title="([^"]*)"', row)
        assert m and m.group(1) == expected, f"{d}: tooltip is {m and m.group(1)!r}"


def test_home_and_archive_agree(three_days):
    """One template, so the two cannot drift — asserted, not assumed."""
    home = (paths.ROOT / "site" / "index.html").read_text(encoding="utf-8")
    assert LINE in _lead(_row(home, "2026-08-18"))
    assert "uc-row__lead--title" in _lead(_row(home, "2026-08-17"))


# --------------------------------------------------------------------------
# 3. The mark is drawn the way the missing-day reason is, and no taller
# --------------------------------------------------------------------------


def test_the_fallback_is_styled_like_the_other_not_usual_content():
    """The installed stylesheet, not `paths.ROOT` — `repo` points that at a
    tmp dir and the template lives in the package."""
    import pipeline.render

    css = (
        Path(pipeline.render.__file__).parent / "templates" / "base.css.j2"
    ).read_text(encoding="utf-8")
    block = re.search(r"\.uc-row__lead--title \{([^}]*)\}", css)
    assert block, "the fallback has no style, so it is not marked at all"
    body = block.group(1)
    assert "italic" in body and "--uc-faint" in body
    # Nothing that changes how tall the row is. One line per row does not get
    # to depend on state (0Z-C).
    for forbidden in ("font-size", "line-height", "font-weight", "padding", "margin"):
        assert forbidden not in body, f"{forbidden} in the fallback can change row height"
