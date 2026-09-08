"""A day a required source went quiet says so, in all three places (1E, A).

2026-09-07 published three journal papers with `collect.arxiv` silent. The
verdict layer saw it — `REQUIRED_SOURCES` exists for exactly the claim
*"half our declared scope, missing, with nothing in the data saying so"* — and
`outcome.silent_sources` recorded it in the run log. Then every surface a
reader can see printed **"7 arXiv categories"**, the configured number, on a
day arXiv had contributed nothing.

The judgement that this does not stop publication is unchanged and is right:
*"Journal items collected on a day arXiv was unreachable are real papers, and
withholding them would trade a partial issue for none."* What was wrong was the
display.

So: the issue page, the archive row and the API. And the derivation, which is
the other half of the fix — the fact lives in `content/runs_log/` and no issue
file has ever carried it, so the pages read the log rather than gaining a
second copy of the number (D318's rule, and D322's).
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from pipeline.models import Headline, Issue, ScanMeta
from pipeline.outcome import (
    PUBLISHED,
    Outcome,
    record,
    silent_by_date,
    silent_for,
    silent_streak,
    source_label,
)

DAY = date(2026, 9, 7)
QUIET_DAY = date(2026, 9, 6)


def _issue(d: date, items: list[str]) -> Issue:
    return Issue(
        date=d,
        headline=Headline(present=True, line="a line about the day", work_key=items[0]),
        scan_meta=ScanMeta(
            arxiv_categories=7, journals=96, candidates_scanned=351,
            items_published=len(items),
        ),
        items=items,
    )


def _row(d: date, silent: list[str]) -> None:
    record(Outcome(
        date=d, status=PUBLISHED, candidates=211, published=3, silent_sources=silent,
    ))


@pytest.fixture()
def archive(repo):
    """Two days: one with arXiv silent, one ordinary. The second is the
    regression guard — every assertion below has a partner saying the mark does
    not appear where it should not."""
    from pipeline import store

    for d, silent in ((DAY, ["collect.arxiv"]), (QUIET_DAY, [])):
        store.save_issue(_issue(d, [f"doi:10.1016/x{d.day}"]))
        _row(d, silent)
    return repo


# --------------------------------------------------------------------------
# The derivation
# --------------------------------------------------------------------------


def test_the_fact_is_read_from_the_run_log_not_the_issue(archive):
    """No issue file has a `silent_sources` field and none is being added.

    A source going quiet is a fact about the **run**, not about the artefact,
    and the run log is where the run's facts live. Storing it a second time in
    the issue is the fallback D318 refused to leave behind: two homes for one
    number is the next bug, not a safety net.
    """
    from pipeline import store

    stored = json.loads(
        (store.paths.CONTENT / "issues" / f"{DAY}.json").read_text(encoding="utf-8")
    )
    assert "silent_sources" not in stored
    assert "silent_sources" not in stored["scan_meta"]

    assert silent_for(DAY) == ["arXiv"]
    assert silent_for(QUIET_DAY) == []


def test_both_derivations_agree(archive):
    """One page and a whole-site build ask differently and must answer alike."""
    table = silent_by_date()
    for d in (DAY, QUIET_DAY):
        assert silent_for(d, table) == silent_for(d), d


def test_a_stage_name_never_reaches_a_reader(archive):
    """`collect.arxiv` belongs in the log. One map, so the three surfaces
    cannot spell the same source three ways."""
    assert source_label("collect.arxiv") == "arXiv"
    assert source_label("collect.openalex") == "OpenAlex"
    assert all("collect." not in name for name in silent_for(DAY))


def test_a_date_with_no_row_claims_nothing(repo):
    """The five backfilled days have no run-log row, and silence on them is not
    merely unrecorded — it is **unmeasurable**: `silent_sources` needs a window
    of two days or more and a backfill reads one. An empty list here means "no
    mark", never "both sources answered"; those days already say `filled in
    later`, which is the honest thing to say about them."""
    assert silent_for(date(2026, 1, 1)) == []


def test_the_streak_stops_at_a_day_that_answered(archive):
    """09-06 has a row and it is not silent, so 09-05's silence is a separate
    episode rather than a continuation. The streak counts consecutive days."""
    assert silent_streak("collect.arxiv", DAY) == 1
    _row(date(2026, 9, 5), ["collect.arxiv"])
    assert silent_streak("collect.arxiv", DAY) == 1


def test_the_streak_stops_at_a_day_with_no_row(repo):
    """"We did not look" is not evidence of silence. A gap in the log ends the
    count rather than being read through — the distinction `outcome.py` exists
    for, applied to the thing that decides whether a person gets mailed."""
    _row(DAY, ["collect.arxiv"])
    _row(DAY - timedelta(days=1), ["collect.arxiv"])
    assert silent_streak("collect.arxiv", DAY) == 2

    # DAY-3 was silent too, but DAY-2 has no row at all. The streak is 2, not 3.
    _row(DAY - timedelta(days=3), ["collect.arxiv"])
    assert silent_streak("collect.arxiv", DAY) == 2


# --------------------------------------------------------------------------
# Surface 1: the issue page
# --------------------------------------------------------------------------


def _page(d: date) -> str:
    from pipeline import store
    from pipeline.render.preview import render_issue

    issue = store.load_issue(d)
    return render_issue(issue, [], [])


def test_the_scan_line_stops_claiming_categories_arxiv_did_not_read(archive):
    """The sentence at the top of the issue. It said "7 arXiv categories" on a
    day arXiv gave us nothing — the configured number printed as a reading."""
    html = _page(DAY)

    assert "arXiv categories" not in html
    assert "96 journals" in html
    assert "351 candidates" in html


def test_an_ordinary_day_is_untouched(archive):
    html = _page(QUIET_DAY)

    assert "7 arXiv categories" in html
    # The class names are in the inlined stylesheet on every page; what must be
    # absent is the markup that uses them.
    assert '<p class="uc-silent">' not in html
    assert ">no arXiv<" not in html


def test_the_page_says_what_it_does_not_mean(archive):
    """The mark alone would read as a retraction of the papers below it. The
    sentence beside it exists to say that the papers are real and the missing
    thing is the scope."""
    html = _page(DAY)

    assert 'class="uc-chip uc-chip--silent">no arXiv<' in html
    assert "The papers below are real" in html
    assert "other half of the scope" in html


def test_the_plain_text_edition_says_the_same(archive):
    from pipeline import store
    from pipeline.render.plaintext import render_text

    text = render_text(store.load_issue(DAY), [], [])
    assert "arXiv categories" not in text
    assert "no arXiv" in text
    assert "papers below are real" in text

    ordinary = render_text(store.load_issue(QUIET_DAY), [], [])
    assert "7 arXiv categories" in ordinary
    assert "no arXiv" not in ordinary


# --------------------------------------------------------------------------
# Surface 2: the archive row
# --------------------------------------------------------------------------


def test_the_archive_row_carries_the_mark_without_changing_its_kind(archive):
    """Additive, like `recent` and `withheld`. The day published three papers
    and is still a published day; a state that replaced the row's kind would
    say something the outcome model does not."""
    from pipeline.render.site import archive_rows

    rows = {r["date"]: r for r in archive_rows()}

    assert rows[str(DAY)]["silent"] == ["arXiv"]
    assert rows[str(DAY)]["published"] == 1
    assert rows[str(DAY)]["quiet"] is False
    assert rows[str(DAY)]["missing"] is False
    assert rows[str(QUIET_DAY)]["silent"] == []


def test_the_row_and_the_legend_both_appear(archive):
    from pipeline.render.site import archive_rows, build_archive

    rows = archive_rows()
    html = "\n".join(
        p.read_text(encoding="utf-8") for p in build_archive()
    )
    assert 'uc-chip--silent">no arXiv<' in html
    # The legend explains only marks that are on the page — a definition for
    # something the reader cannot see is noise in a different place.
    assert "One of the two sources we promise to read" in html
    assert any(r["silent"] for r in rows)


def test_the_histogram_note_says_it_too(archive):
    """Colour never carries meaning here, and the bar has no room for a chip,
    so the fact travels in the bar's own label."""
    from pipeline.render.site import archive_rows, spark_bars

    bars = {b["date"]: b for b in spark_bars(archive_rows())}
    assert ", no arXiv" in bars[str(DAY)]["label"]
    assert "no arXiv" not in bars[str(QUIET_DAY)]["label"]


# --------------------------------------------------------------------------
# Surface 3: the API
# --------------------------------------------------------------------------


def test_the_api_publishes_it_and_the_catalogue_carries_it(archive, tmp_path):
    from pipeline.render.api import build_api

    build_api(tmp_path / "api")

    day = json.loads((tmp_path / "api/issues" / f"{DAY}.json").read_text("utf-8"))
    assert day["silent_sources"] == ["arXiv"]

    ordinary = json.loads(
        (tmp_path / "api/issues" / f"{QUIET_DAY}.json").read_text("utf-8")
    )
    assert ordinary["silent_sources"] == []

    # In the catalogue too, so a consumer scanning `index.json` learns it
    # without fetching every day.
    index = json.loads((tmp_path / "api/index.json").read_text("utf-8"))
    catalogue = {r["date"]: r for r in index["issues"]}
    assert catalogue[str(DAY)]["silent_sources"] == ["arXiv"]


def test_arxiv_categories_keeps_its_meaning_and_gains_a_neighbour(archive, tmp_path):
    """The API's own promise is that fields are **added, never repurposed** —
    *"renaming in place is how a consumer ends up silently wrong"*. So
    `counts.arxiv_categories` still reports what the run was configured to
    query, which is true, and `silent_sources` next to it says whether the
    query returned anything. The page drops the clause because a page cannot
    carry a footnote per number; the API keeps it and labels it."""
    from pipeline.render.api import build_api
    from pipeline.render.site import API_FIELDS

    build_api(tmp_path / "api")
    day = json.loads((tmp_path / "api/issues" / f"{DAY}.json").read_text("utf-8"))

    assert day["counts"]["arxiv_categories"] == 7

    documented = dict(API_FIELDS)
    assert "silent_sources" in documented
    assert "does **not** say that arXiv answered" in documented[
        "counts.arxiv_categories"
    ]


# --------------------------------------------------------------------------
# And the three of them say one thing
# --------------------------------------------------------------------------


def test_the_three_surfaces_agree_across_the_whole_archive(archive, tmp_path):
    """Not "each is right" — that they cannot disagree. Every issue, every
    surface, one derivation behind all of them."""
    from pipeline.render.api import build_api
    from pipeline.render.site import archive_rows, build_issue_pages

    build_api(tmp_path / "api")
    pages = {p.stem: p.read_text(encoding="utf-8") for p in
             build_issue_pages(tmp_path / "issues")}
    rows = {r["date"]: r for r in archive_rows()}

    for d in (DAY, QUIET_DAY):
        key = str(d)
        expected = silent_for(d)
        api = json.loads((tmp_path / "api/issues" / f"{key}.json").read_text("utf-8"))

        assert api["silent_sources"] == expected, key
        assert rows[key]["silent"] == expected, key
        for name in expected:
            assert f">no {name}<" in pages[key], (key, name)
        if not expected:
            assert '<p class="uc-silent">' not in pages[key], key


def test_rendering_does_not_rewrite_a_single_issue(archive):
    """The 1B guarantee, restated for this fact: the whole point of deriving is
    that `content/` does not move."""
    from pipeline import store
    from pipeline.render.site import build_issue_pages

    path = store.paths.CONTENT / "issues" / f"{DAY}.json"
    before = path.read_bytes()
    build_issue_pages()
    assert path.read_bytes() == before
