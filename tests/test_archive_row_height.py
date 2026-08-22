"""Every archive row is one line tall, whatever state it is in (0Z-C).

Fourteen rows on the home strip, seven of them two lines tall, one line and two
lines alternating down the table. The cause looked like one thing and was three,
and only the first was the one the eye lands on:

  1. `.uc-row__note` — a sentence under the row, on any row that was backfilled,
     recent, or unranked. Seven of fourteen.
  2. `.uc-row__lead` with `flex-basis: auto` — a flex line breaks on an item's
     *hypothetical* size, which for `auto` is the whole untruncated title. Long
     titles pushed the lead onto a line of its own, where it then shrank to the
     full row width and ellipsised, looking like a normal truncated lead sitting
     one line too low. **Twelve of fourteen**, more than the note.
  3. `.uc-row__count` with a fixed `5.5rem` basis — the chips inside it had
     nowhere to go but downward, and two chips stacked four lines deep.

All three were measured in a browser before any of it was believed; only the
first was in the brief. The structural half of this file fixes the three facts
that produce the single line, each next to the failure it prevents.

★ And then a second accident (0Z-D) showed why that is not enough on its own:
an issue page lost its whole navigation while every string-level test passed,
because the markup was present and merely in the wrong place. A test that reads
CSS cannot see a screen. So the last test here **opens the built page in
headless chromium and measures the rendered rows**, which is the only assertion
in this file that would have failed for the original bug.

No network, no keys. The browser test skips if chromium is not installed —
`uv run python -m playwright install chromium` puts it there.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from pipeline import paths, store
from pipeline.models import Bibliography, Headline, Issue, Item, PrimaryLocation, ScanMeta
from pipeline.outcome import NOT_PUBLISHED, Outcome, record
from pipeline.render.site import build_archive, build_home

# A long title is the point: it is what used to wrap the lead onto its own line.
LONG = (
    "From announcement to operation: the effect of a new light rail line on "
    "residential property values across four distinct project phases"
)


def _item(key: str, title: str) -> Item:
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


def _issue(d, keys, *, backfilled=False, headline=True):
    issue = Issue(
        date=d,
        items=sorted(keys),
        headline=Headline(
            present=bool(headline and keys),
            work_key=keys[0] if (headline and keys) else None,
            line="A line." if (headline and keys) else None,
        ),
        scan_meta=ScanMeta(items_published=len(keys), candidates_scanned=100, journals=96),
        backfilled=backfilled,
    )
    store.save_issue(issue)
    return issue


@pytest.fixture
def every_state(repo):
    """One day of each kind, all in one table, so no state can be forgotten."""
    _item("arxiv:2608.30001", LONG)
    _item("arxiv:2608.30002", "A short one")

    _issue(date(2026, 8, 18), ["arxiv:2608.30001"])                       # plain
    _issue(date(2026, 8, 17), ["arxiv:2608.30002"], backfilled=True)      # filled in
    _issue(date(2026, 8, 16), ["arxiv:2608.30001"], headline=False)       # no headline
    quiet = _issue(date(2026, 8, 15), ["arxiv:2608.30002"])               # quiet
    quiet.items = []
    store.save_issue(quiet)
    record(Outcome(date=date(2026, 8, 14), status=NOT_PUBLISHED,          # no issue
                   reasons=["collect.arxiv did not run"], published=0))
    build_archive()
    build_home()
    return (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")


def _rows(html: str) -> list[str]:
    return re.findall(r'<li class="uc-row[^"]*" data-date="[\d-]+">(.*?)</li>', html, re.S)


# --------------------------------------------------------------------------
# 1. The note is gone from every row that does not own its sentence
# --------------------------------------------------------------------------


def test_no_row_carries_a_note(every_state):
    """The withheld day is the only exception, and there is not one here."""
    rows = _rows(every_state)
    assert len(rows) == 5, "one row per kind, or this test is not testing them"

    for row in rows:
        assert "uc-row__note" not in row, f"a note is a second line: {row[:120]}"


def test_the_states_are_still_named_on_the_row(every_state):
    """Because the alternative is colour alone, which base.css refuses."""
    html = every_state
    for words in ("filled in later", "no headline", "a quiet day", "no issue"):
        assert words in html, f"{words!r} left the row and nothing replaced it"


def test_the_legend_says_each_thing_once(every_state):
    """Once under the table, not once per row — that was the whole bug."""
    html = every_state
    assert html.count("assembled from the archived candidates") == 1
    assert html.count("none of them scored high enough") == 1
    assert html.count("The sources did not answer") == 1
    assert html.count("nothing was worth publishing") == 1


def test_the_legend_explains_only_marks_that_are_on_the_page(repo):
    """A definition for a chip the reader cannot see is the same noise moved."""
    _item("arxiv:2608.30003", "An ordinary paper")
    _issue(date(2026, 8, 18), ["arxiv:2608.30003"])
    build_archive()
    html = (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")

    assert "uc-row" in html, "there is a table"
    assert 'class="uc-legend"' not in html, "and nothing on it needs explaining"


def test_a_withheld_day_keeps_its_note(repo):
    """The exception, stated as a test so it cannot be tidied away later.

    The reason a live run withheld a day belongs to that date. A legend says a
    thing once for every row that carries the mark; there is no such sentence
    for "collect.openalex finished FAILED" on one morning in August, and
    dropping it would undo 0Y's Y1-2 — the backfilled issue would once more
    silently replace the record of the run that refused to publish.
    """
    _item("arxiv:2608.30004", "A withheld-then-filled day's paper")
    _issue(date(2026, 8, 13), ["arxiv:2608.30004"], backfilled=True)
    record(Outcome(date=date(2026, 8, 13), status=NOT_PUBLISHED,
                   reasons=["collect.openalex finished FAILED"], published=0))
    build_archive()
    html = (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")

    (row,) = _rows(html)
    assert "uc-row__note" in row
    assert "collect.openalex finished FAILED" in row


# --------------------------------------------------------------------------
# 2. The three CSS declarations the single line rests on
# --------------------------------------------------------------------------


def _css() -> str:
    # The installed template, not `paths.ROOT` — the `repo` fixture points that
    # at a tmp dir, and the stylesheet these tests are about lives in the
    # package.
    import pipeline.render

    return (Path(pipeline.render.__file__).parent / "templates" / "base.css.j2").read_text(
        encoding="utf-8"
    )


def test_the_lead_has_a_zero_flex_basis():
    """`auto` here is the bug that the ellipsis hid — twelve rows of fourteen.

    A flex line breaks on flex-basis, not on the width the item settles at, so
    `1 1 auto` measured the whole title and wrapped. `1 1 0` never forces a
    break and still truncates, because `min-width: 0` lets it shrink below its
    content.
    """
    css = _css()
    block = re.search(r"\.uc-row__lead \{(.*?)\}", css, re.S).group(1)

    assert re.search(r"flex:\s*1\s+1\s+0\s*;", block), "a lead with an auto basis wraps"
    assert "min-width: 0" in block, "without this it cannot shrink, so it wraps anyway"
    assert "text-overflow: ellipsis" in block, "and the reader must see it was cut"


def test_the_count_column_may_grow_but_never_wraps():
    """Chips in a fixed 88px box stack downward. Two of them cost four lines."""
    block = re.search(r"\.uc-row__count \{(.*?)\}", _css(), re.S).group(1)

    assert "white-space: nowrap" in block
    assert "min-width: 5.5rem" in block, "the number still starts at one x"
    assert not re.search(r"flex:\s*0\s+0\s+5\.5rem", block), "a fixed basis is the bug"


def test_the_recent_underline_costs_no_height():
    """2px is not a wrapped line, but it is still height that depends on state."""
    css = _css()
    base = re.search(r"\.uc-row__date \{(.*?)\}", css, re.S).group(1)

    assert "border-bottom: 2px dotted transparent" in base, "every row reserves it"
    assert re.search(
        r"\.uc-row--recent \.uc-row__date \{ border-bottom-color:", css
    ), "and a recent day only colours it in"


def test_the_narrow_screen_rule_survives():
    """7b is deliberate: below 640px the lead drops to its own line. Keeping the
    wide table on one line must not be done by deleting the mobile design."""
    css = _css()
    # There is more than one 640px block in this stylesheet, so anchor on the
    # rule itself rather than on the first `@media` that matches.
    anchor = css.index(".uc-row__lead { flex: 1 1 100%; white-space: normal; }")
    narrow = css[anchor : anchor + 700]

    assert css.rindex("@media (max-width: 640px)", 0, anchor) > 0, "still inside 7b"
    # And the chips must be allowed to wrap again down here, or three of them
    # push a 375px row wider than the screen.
    assert re.search(r"\.uc-row__count \{[^}]*white-space: normal", narrow), (
        "nowrap is a wide-screen rule and must be undone for the stacked layout"
    )


# --------------------------------------------------------------------------
# 3. And the only assertion here that looks at a screen
# --------------------------------------------------------------------------


def _browser():
    """Headless chromium, or a skip that says which of the two is missing."""
    sync_playwright = pytest.importorskip(
        "playwright.sync_api", reason="playwright is not installed"
    ).sync_playwright
    pw = sync_playwright().start()
    try:
        return pw, pw.chromium.launch()
    except Exception as exc:  # the package is there, the browser binary is not
        pw.stop()
        pytest.skip(f"chromium is not installed: {exc}")


def test_the_rendered_rows_are_all_the_same_height(every_state):
    """The measurement, not the stylesheet.

    Every row on the wide layout must render to one height whatever state it
    carries — chips or none, a 131-character headline or a 62-character one.
    The first row is allowed to be exactly 1px shorter, and only that: it has
    `border-top: 0`, which is its **position** in the list and not its state.
    """
    pw, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 1400})
        page.goto((paths.ROOT / "site" / "archive.html").as_uri())
        rows = page.evaluate(
            """() => [...document.querySelectorAll('.uc-row')].map(li => {
                 const lead = li.querySelector('.uc-row__lead');
                 return {
                   date: li.dataset.date,
                   height: +li.getBoundingClientRect().height.toFixed(2),
                   borderTop: getComputedStyle(li).borderTopWidth,
                   chips: li.querySelectorAll('.uc-chip').length,
                   headline: lead ? lead.textContent.trim().length : 0,
                   cut: lead ? lead.scrollWidth > lead.clientWidth : false,
                 };
               })"""
        )
        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
    finally:
        browser.close()
        pw.stop()

    assert len(rows) == 5, "one row per kind"

    # Take the border off each row's box and one height is left. The first row
    # has none (`border-top: 0`), the rest have 1px; subtracting it is what
    # makes the comparison about state instead of position.
    def body_height(row):
        return round(row["height"] - float(row["borderTop"].rstrip("px")), 2)

    heights = {body_height(r) for r in rows}
    assert len(heights) == 1, (
        "rows differ in height: "
        + ", ".join(
            f"{r['date']} h={r['height']} chips={r['chips']} headline={r['headline']}"
            for r in rows
        )
    )

    # And the states really were different, or the assertion above is vacuous.
    assert len({r["chips"] for r in rows}) > 1, "every row carried the same chips"
    assert max(r["headline"] for r in rows) > min(r["headline"] for r in rows)
    # Truncation is the healthy state: the DOM keeps the whole string and the
    # ellipsis is CSS, so a long lead overflows its box on purpose.
    assert any(r["cut"] for r in rows), "nothing was truncated, so nothing was long"
    assert overflow == 0, "the page scrolls sideways"
