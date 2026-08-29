"""A backfilled day says so, and a withheld day still says that too (0Y).

`backfill_issues.py` writes `backfilled: true` and explains why in the same
breath — *"so no aggregate mixes the two kinds without saying so"*. The
aggregates kept that promise. The screen did not: nothing under `render/` read
the flag, so a day assembled weeks after the fact was drawn exactly like a day
the pipeline watched at 06:00.

The second half is narrower and easier to lose. A date can carry **two** facts:
a `not_published` run-log row, because the live run withheld the day, and a
backfilled issue written later. `archive_rows` only ever added a "no issue" row
for dates with *no* issue, so filling a withheld day silently deleted the
evidence that `REQUIRED_SOURCES` had worked. `pages.yml` names the principle
that breaks — "the archive is the record" — so the row now carries both, and
this file is what keeps it carrying them.

No network, no keys.
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from pipeline import paths, store
from pipeline.models import Bibliography, Headline, Issue, Item, PrimaryLocation, ScanMeta
from pipeline.outcome import NOT_PUBLISHED, PUBLISHED, Outcome, record
from pipeline.render.site import archive_rows, archive_stats, build_archive, item_index

LIVE = date(2026, 8, 18)
FILLED = date(2026, 8, 12)
BOTH = date(2026, 8, 20)


def _item(key: str, title: str) -> Item:
    it = Item(
        work_key=key,
        first_published=LIVE,
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


def _issue(d: date, keys: list[str], *, backfilled: bool = False) -> Issue:
    issue = Issue(
        date=d,
        items=sorted(keys),
        headline=Headline(present=True, work_key=keys[0], line=f"A line for {d}."),
        scan_meta=ScanMeta(items_published=len(keys), candidates_scanned=100, journals=96),
        backfilled=backfilled,
    )
    store.save_issue(issue)
    return issue


def _withheld(d: date, reason: str) -> None:
    record(Outcome(date=d, status=NOT_PUBLISHED, reasons=[reason], published=0))


@pytest.fixture
def archive(repo):
    """Three kinds of day, so the distinctions have to hold at once."""
    _item("arxiv:2608.10001", "A live day's paper")
    _item("arxiv:2608.10002", "A backfilled day's paper")
    _item("arxiv:2608.10003", "A withheld-then-backfilled day's paper")

    _issue(LIVE, ["arxiv:2608.10001"])
    _issue(FILLED, ["arxiv:2608.10002"], backfilled=True)
    _issue(BOTH, ["arxiv:2608.10003"], backfilled=True)
    _withheld(BOTH, "collect.openalex finished FAILED")
    record(Outcome(date=LIVE, status=PUBLISHED, published=1))
    return archive_rows()


def _row(rows, d: date) -> dict:
    match = [r for r in rows if r["date"] == str(d)]
    assert match, f"no row for {d}"
    return match[0]


# --------------------------------------------------------------------------
# The data
# --------------------------------------------------------------------------


def test_a_live_day_is_not_marked(archive):
    row = _row(archive, LIVE)
    assert row["backfilled"] is False
    assert row["withheld"] is None
    assert row["missing"] is False


def test_a_backfilled_day_is_marked(archive):
    row = _row(archive, FILLED)
    assert row["backfilled"] is True
    assert row["published"] == 1, "it published — that is not in question"
    assert row["missing"] is False, "a backfilled day is not a day nobody could see"


def test_a_withheld_day_that_was_later_filled_keeps_both_facts(archive):
    """Y1-2. Before this, the issue replaced the record and the record lost."""
    row = _row(archive, BOTH)

    assert row["backfilled"] is True
    assert row["published"] == 1
    assert row["withheld"] == "collect.openalex finished FAILED"

    # And it must not *also* appear as a separate "no issue" row: one date, one
    # row, two facts.
    assert len([r for r in archive if r["date"] == str(BOTH)]) == 1


def test_a_withheld_day_with_no_issue_is_still_a_missing_row(repo):
    """The pre-existing behaviour, unchanged: nothing filled it, so it is a gap."""
    _withheld(date(2026, 8, 15), "collect.arxiv did not run")
    rows = archive_rows()

    row = _row(rows, date(2026, 8, 15))
    assert row["missing"] is True
    assert row["backfilled"] is False
    assert row["reason"] == "collect.arxiv did not run"


# --------------------------------------------------------------------------
# The rendering
# --------------------------------------------------------------------------


def test_the_backfilled_row_is_drawn_differently(archive):
    build_archive()
    html = (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")

    live = _li(html, LIVE)
    filled = _li(html, FILLED)

    assert "uc-row--backfilled" in filled
    assert "uc-row--backfilled" not in live
    assert "filled in later" in filled
    assert "filled in later" not in live
    # Not colour alone: the row still says it in words. What moved in 0Z-C is
    # the *explanation*, out of the row and into the legend under the list —
    # the same sentence on every backfilled row was what made half the table
    # two lines tall. The chip stayed exactly because moving it would leave
    # the row drawn in colour alone.
    assert "one-day window" not in filled, "the sentence is in the legend now"
    assert "from the candidates archived for that date" in html, "and it is still said"


def test_the_rendered_row_states_both_facts(archive):
    """The assertion Y1-2 exists for, at the level a reader meets it."""
    build_archive()
    html = (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")

    both = _li(html, BOTH)

    assert "filled in later" in both, "the issue is there and says how it got there"
    assert "published nothing" in both, "the live verdict survives the fill"
    assert "collect.openalex finished FAILED" in both, "including why"
    # ★ The withheld reason is the one note 0Z-C could not move. A legend says
    # a thing once for every row that carries the mark, and this reason is one
    # date's own — no sentence written once under the table can hold it.
    assert 'class="uc-row__note"' in both


def test_a_backfilled_day_is_not_drawn_as_a_failure(archive):
    """A day assembled later is a published day, drawn as one.

    The phrase this docstring used to carry — *"it is part of the method, not
    an apology"* — was on the page until 0Z-F (S4), where it came off for the
    reason it was written: a line saying it will not apologise is one. The
    principle it stated is unchanged and is what the assertions below check.
    """
    build_archive()
    html = (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")
    filled = _li(html, FILLED)

    assert "uc-row--missing" not in filled
    assert "no issue" not in filled
    # The day's own link still works — it is a published issue like any other.
    assert f'{FILLED}.html' in filled


def test_the_other_three_states_are_untouched(repo):
    """published / quiet / missing, exactly as before."""
    _item("arxiv:2608.20001", "An ordinary paper")
    _issue(date(2026, 8, 18), ["arxiv:2608.20001"])
    _issue(date(2026, 8, 17), ["arxiv:2608.20001"])
    quiet = store.load_issue(date(2026, 8, 17))
    quiet.items = []
    quiet.quiet_day = True
    store.save_issue(quiet)
    _withheld(date(2026, 8, 16), "collect.arxiv did not run")

    rows = archive_rows()
    build_archive()
    html = (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")

    assert _row(rows, date(2026, 8, 18))["published"] == 1
    assert _row(rows, date(2026, 8, 17))["quiet"] is True
    assert _row(rows, date(2026, 8, 16))["missing"] is True
    assert "a quiet day" in html
    assert "no issue" in html
    # And none of them picked up the new note.
    assert "Assembled later" not in html


# --------------------------------------------------------------------------
# The aggregates
# --------------------------------------------------------------------------


def test_the_aggregate_says_how_many_were_filled_in(archive):
    """`backfill_issues.py`: "so no aggregate mixes the two kinds without
    saying so". The counting is the saying so."""
    stats = archive_stats(archive, item_index())

    assert stats["backfilled_days"] == 2
    # `published_days` still counts every day that published, because each of
    # them did. What changed is that the mixture is now stated.
    assert stats["published_days"] == 3
    assert stats["missing_days"] == 0


def _li(html: str, d: date) -> str:
    """The one `<li>` for a date, so an assertion cannot pass on another row."""
    m = re.search(rf'<li class="uc-row[^"]*" data-date="{d}">(.*?)</li>', html, re.S)
    assert m, f"no rendered row for {d}"
    return m.group(0)
