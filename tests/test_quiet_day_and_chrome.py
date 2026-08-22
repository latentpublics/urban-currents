"""What "a quiet day" means, and the chrome that says it (phase 0Z).

`Issue.quiet_day` on disk does not mean the day was quiet. `stage_issue` writes
`quiet_day=headline_item is None`, and `pick_headline` returns None whenever
nothing clears the headline threshold — so the stored flag is a fact about
*ranking*. Two published issues prove the cost: 2026-08-09 carries it over
eleven papers and 2026-08-21 over nine, and every renderer printed "A quiet day
in urban data science." across all of them.

U5 found this one layer up and fixed the run log, then wrote that `quiet_day`
"is what the renderer keys on, and that one is true". That was the half nobody
checked. So these tests hold two lines:

  * **quiet is derived from the item count**, never read from storage, in all
    four render paths — archive row, issue page, plaintext, preview subject;
  * **"published papers, no headline" is its own state** and says so in words.

The rest covers the chrome those states are drawn in: one nav list for three
renderers, the histogram speaking the same colour language as the list, and the
front page's deliberate lag.

No network, no keys.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pytest

from pipeline import paths, store
from pipeline.models import Bibliography, Headline, Issue, Item, ScanMeta
from pipeline.render import site as site_mod
from pipeline.render.plaintext import render_text
from pipeline.render.preview import email_subject, render_issue
from pipeline.render.site import archive_rows, build_archive, build_home, spark_bars

TODAY = date(2026, 8, 22)


def _item(key: str, title: str, headline_score: float = 0.1) -> Item:
    it = Item(
        work_key=key,
        first_published=date(2026, 8, 9),
        bibliography=Bibliography(title=title, abstract="x"),
    )
    it.scores.headline = headline_score
    store.save_item(it)
    return it


def _issue(d: date, n: int, *, headline: bool, stored_quiet: bool, backfilled: bool = False) -> Issue:
    keys = []
    for i in range(n):
        key = f"arxiv:{d.strftime('%m%d')}.{i:04d}"
        _item(key, f"Paper {i} for {d}")
        keys.append(key)
    issue = Issue(
        date=d,
        items=sorted(keys),
        headline=Headline(
            present=headline,
            work_key=keys[0] if keys else None,
            line="A line that leads the day." if headline else None,
        ),
        # The lie, exactly as it sits in the two real files.
        quiet_day=stored_quiet,
        backfilled=backfilled,
        scan_meta=ScanMeta(items_published=n, candidates_scanned=90, journals=96),
    )
    store.save_issue(issue)
    return issue


# --------------------------------------------------------------------------
# Z1 — the definition
# --------------------------------------------------------------------------


def test_a_day_with_papers_is_not_quiet_however_the_file_reads(repo):
    """2026-08-21 in miniature: nine papers, `quiet_day: true` on disk."""
    issue = _issue(date(2026, 8, 21), 9, headline=False, stored_quiet=True)

    assert issue.quiet_day is True, "the stored flag is left exactly as it was"
    assert issue.is_quiet is False, "and it is not what anything reads"
    assert issue.has_headline is False


def test_a_day_with_nothing_is_quiet(repo):
    issue = _issue(date(2026, 8, 16), 0, headline=False, stored_quiet=True)

    assert issue.is_quiet is True
    assert issue.has_headline is False


def test_a_headline_needs_a_line_not_just_a_flag(repo):
    """`present` without a line is not a headline anyone can read."""
    issue = _issue(date(2026, 8, 18), 3, headline=True, stored_quiet=False)
    assert issue.has_headline is True

    issue.headline.line = None
    assert issue.has_headline is False


@pytest.mark.parametrize("stored", [True, False])
def test_the_stored_flag_never_changes_the_answer(repo, stored):
    """The point of deriving: old files cannot colour the screen."""
    issue = _issue(date(2026, 8, 17), 5, headline=False, stored_quiet=stored)
    assert issue.is_quiet is False


# --------------------------------------------------------------------------
# Z1 — all four render paths
# --------------------------------------------------------------------------


def test_no_render_path_calls_a_nine_paper_day_quiet(repo):
    """The four places the sentence could reach a reader."""
    d = date(2026, 8, 21)
    issue = _issue(d, 9, headline=False, stored_quiet=True)
    items = [store.load_item(k) for k in issue.items]

    # 1. the issue page / email body. The stylesheet is inlined and its own
    # comments discuss quiet days, so the assertion is about what a reader
    # sees, not about the bytes.
    page = _visible(render_issue(issue, items))
    assert "a quiet day" not in page.lower()
    assert "nothing cleared the headline bar" in page.lower()

    # 2. the plaintext mail
    text = render_text(issue, items)
    assert "quiet day" not in text.lower()
    assert "nothing cleared the headline" in text.lower()

    # 3. the mail subject
    subject = email_subject(issue)
    assert "quiet day" not in subject.lower()
    assert "9 papers" in subject

    # 4. the archive row
    build_archive()
    html = (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")
    row = _li(html, d)
    assert "a quiet day" not in row
    assert "no headline" in row


def test_a_genuinely_quiet_day_still_says_so(repo):
    """Guard against fixing the false positive by deleting the sentence."""
    d = date(2026, 8, 16)
    issue = _issue(d, 0, headline=False, stored_quiet=True)

    assert "a quiet day" in email_subject(issue)
    assert "quiet day" in render_text(issue, []).lower()

    build_archive()
    html = (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")
    assert "a quiet day" in _li(html, d)


def test_the_row_distinguishes_quiet_from_unranked(repo):
    _issue(date(2026, 8, 21), 9, headline=False, stored_quiet=True)
    _issue(date(2026, 8, 16), 0, headline=False, stored_quiet=True)
    _issue(date(2026, 8, 18), 4, headline=True, stored_quiet=False)

    rows = {r["date"]: r for r in archive_rows()}

    assert (rows["2026-08-21"]["quiet"], rows["2026-08-21"]["unranked"]) == (False, True)
    assert (rows["2026-08-16"]["quiet"], rows["2026-08-16"]["unranked"]) == (True, False)
    assert (rows["2026-08-18"]["quiet"], rows["2026-08-18"]["unranked"]) == (False, False)


def test_the_publication_rate_follows_the_item_count(repo):
    """`report.py` counts these, and the number lands on the About page."""
    issues = [
        _issue(date(2026, 8, 21), 9, headline=False, stored_quiet=True),
        _issue(date(2026, 8, 16), 0, headline=False, stored_quiet=True),
        _issue(date(2026, 8, 18), 4, headline=True, stored_quiet=False),
    ]

    assert sum(1 for i in issues if i.is_quiet) == 1, "one day published nothing"
    assert sum(1 for i in issues if i.has_headline) == 1
    assert sum(1 for i in issues if not i.is_quiet and not i.has_headline) == 1


def test_the_best_item_is_kept_even_when_it_misses_the_bar(repo):
    """Z1-4: `pick_headline` computed the best item and threw it away."""
    from pipeline.score.headline import best_candidate, pick_headline

    items = [_item("arxiv:1", "Low", 0.10), _item("arxiv:2", "High", 0.40)]

    assert pick_headline(items, threshold=0.9) is None, "nothing clears the bar"
    assert best_candidate(items).work_key == "arxiv:2", "and the best is still known"
    assert best_candidate([]) is None


# --------------------------------------------------------------------------
# Z4 — one nav, three renderers
# --------------------------------------------------------------------------


def test_nav_is_the_same_everywhere(repo):
    """The drift this replaces was real: `uc-nav__lang` was in the templates
    and missing from the injected copy."""
    _issue(date(2026, 8, 18), 3, headline=True, stored_quiet=False)
    build_home()
    build_archive()
    site_mod.build_issue_pages()

    navs = {
        name: _nav(_read(name))
        for name in ("index.html", "archive.html", "issues/2026-08-18.html")
    }
    labels = {name: [label for _, label in links] for name, links in navs.items()}

    assert labels["index.html"] == ["Urban Currents", "Latest", "Archive"]
    assert len({tuple(v) for v in labels.values()}) == 1, labels
    for name, html in ((n, _read(n)) for n in navs):
        assert "uc-nav__lang" in _nav_html(html), name


def test_nav_links_are_relative_and_latest_matches_the_front_page(repo):
    """A root-absolute href would resolve against the organisation domain and
    404 under a sub-path deploy; and two "latest" answers is the next bug."""
    _issue(date(2026, 8, 18), 3, headline=True, stored_quiet=False)
    build_home()
    site_mod.build_issue_pages()

    for name in ("index.html", "issues/2026-08-18.html"):
        for href, _ in _nav(_read(name)):
            assert not href.startswith("/"), f"{name}: {href}"

    home_latest = dict((label, href) for href, label in _nav(_read("index.html")))["Latest"]
    issue_latest = dict(
        (label, href) for href, label in _nav(_read("issues/2026-08-18.html"))
    )["Latest"]
    assert home_latest.endswith(issue_latest.split("/")[-1])


# --------------------------------------------------------------------------
# Z3 — the front page lags on purpose, and the lag is one config line
# --------------------------------------------------------------------------


def test_latest_skips_the_days_still_filling_in(repo, monkeypatch):
    issues = [
        _issue(TODAY - timedelta(days=1), 3, headline=True, stored_quiet=False),
        _issue(TODAY - timedelta(days=5), 4, headline=True, stored_quiet=False),
    ]
    monkeypatch.setattr(site_mod, "recent_cutoff", lambda today=None: TODAY - timedelta(days=2))

    assert site_mod.latest_issue(issues).date == TODAY - timedelta(days=5)


def test_one_config_line_turns_the_lag_off(repo, monkeypatch):
    issues = [
        _issue(TODAY - timedelta(days=1), 3, headline=True, stored_quiet=False),
        _issue(TODAY - timedelta(days=5), 4, headline=True, stored_quiet=False),
    ]
    monkeypatch.setattr(site_mod, "recent_cutoff", lambda today=None: TODAY - timedelta(days=2))
    monkeypatch.setattr(
        site_mod, "cfg",
        lambda k, d=None: False if k == "site.latest_skips_recent" else d,
    )

    assert site_mod.latest_issue(issues).date == TODAY - timedelta(days=1)


def test_a_young_archive_still_has_a_front_page(repo, monkeypatch):
    """Every issue is recent — falling through to nothing would blank the home
    page, so the newest wins instead."""
    issues = [_issue(TODAY - timedelta(days=1), 3, headline=True, stored_quiet=False)]
    monkeypatch.setattr(site_mod, "recent_cutoff", lambda today=None: TODAY - timedelta(days=2))

    assert site_mod.latest_issue(issues).date == TODAY - timedelta(days=1)


# --------------------------------------------------------------------------
# Z8 — the histogram speaks the list's language
# --------------------------------------------------------------------------


def test_the_histogram_draws_every_state_the_list_draws(repo):
    rows = [
        {"date": "2026-08-20", "published": 5, "quiet": False, "missing": False,
         "backfilled": False, "recent": False},
        {"date": "2026-08-19", "published": 0, "quiet": True, "missing": False,
         "backfilled": False, "recent": False},
        {"date": "2026-08-18", "published": 0, "quiet": False, "missing": True,
         "backfilled": False, "recent": False, "reason": "collect.arxiv did not run"},
        {"date": "2026-08-17", "published": 4, "quiet": False, "missing": False,
         "backfilled": True, "recent": False},
    ]

    kinds = {b["date"]: b["kind"] for b in spark_bars(rows)}

    assert kinds["2026-08-20"] == "published"
    assert kinds["2026-08-19"] == "quiet"
    assert kinds["2026-08-18"] == "missing"
    # This one had no bar of its own before 0Z: a backfilled day was drawn
    # exactly like a live published day, forty pixels from a list row that
    # marked it with a dashed bar and a chip.
    assert kinds["2026-08-17"] == "backfilled"


def test_recency_marks_the_bar_without_replacing_its_state(repo):
    rows = [{"date": "2026-08-21", "published": 5, "quiet": False, "missing": False,
             "backfilled": False, "recent": True}]

    bar = spark_bars(rows)[0]

    assert bar["kind"] == "published", "recency is additive, not a fifth state"
    assert bar["recent"] is True
    assert "more may follow" in bar["label"]


def test_the_featured_bar_is_not_called_today(repo):
    """With `latest_skips_recent` the front page's issue is days old; a bar
    labelled `today` would have the page assert a date it is not showing."""
    rows = [{"date": "2026-08-19", "published": 5, "quiet": False, "missing": False,
             "backfilled": False, "recent": False}]

    assert spark_bars(rows, today="2026-08-19")[0]["kind"] == "featured"


def test_every_state_the_strip_draws_is_named_in_the_legend(repo):
    _issue(date(2026, 8, 18), 3, headline=True, stored_quiet=False)
    build_home()
    html = (paths.ROOT / "site" / "index.html").read_text(encoding="utf-8")

    legend = re.search(r'uc-spark__legend">(.*?)</span>', html, re.S).group(1).lower()
    for word in ("quiet", "not seen", "filled in later", "still filling in"):
        assert word in legend, word


def test_one_state_has_one_colour(repo):
    """The mapping used to be hardcoded twice, ninety lines apart, and had
    drifted: quiet was #5f646d in the list and #e3e5ea in the histogram."""
    css = _stylesheet()

    for token in ("--uc-state-published", "--uc-state-quiet", "--uc-state-missing",
                  "--uc-state-backfilled"):
        assert f"{token}:" in css, f"{token} is not defined"

    for rule, token in (
        (r"\.uc-row--quiet \.uc-row__fill \{[^}]*\}", "--uc-state-quiet"),
        (r"\.uc-spark__bar--quiet \{[^}]*\}", "--uc-state-quiet"),
        (r"\.uc-spark__bar--published \{[^}]*\}", "--uc-state-published"),
    ):
        m = re.search(rule, css)
        assert m and token in m.group(0), rule


# --------------------------------------------------------------------------
# Z9 / Z10 — what was taken away
# --------------------------------------------------------------------------


def test_hovering_a_row_no_longer_reflows_it(repo):
    css = _stylesheet()
    assert ".uc-row__lead:hover" not in css


def test_the_full_title_is_still_in_the_dom(repo):
    """Removing the hover must not become removing the string: a string cut in
    Python is cut for a screen reader too."""
    long_title = "A very long paper title that the archive row will visually truncate " * 2
    _item("arxiv:9999.0001", long_title.strip(), 0.9)
    issue = Issue(
        date=date(2026, 8, 18),
        items=["arxiv:9999.0001"],
        headline=Headline(present=True, work_key="arxiv:9999.0001", line="A line."),
        scan_meta=ScanMeta(items_published=1, candidates_scanned=10, journals=96),
    )
    store.save_issue(issue)

    build_archive()
    html = (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")

    assert long_title.strip() in html


def test_rows_no_longer_carry_a_code_data_column(repo):
    _issue(date(2026, 8, 18), 3, headline=True, stored_quiet=False)
    build_archive()
    html = (paths.ROOT / "site" / "archive.html").read_text(encoding="utf-8")

    assert '<span class="uc-row__artifacts">' not in html
    # The derivation stays: the latest issue's stat rail still reports it.
    rows = archive_rows()
    assert "code" in rows[0] and "data" in rows[0]


# --------------------------------------------------------------------------


def _stylesheet() -> str:
    """`base.css.j2` from the repository. `paths.ROOT` is the throwaway root."""
    from pathlib import Path as _Path

    return (
        _Path(__file__).resolve().parent.parent
        / "pipeline" / "render" / "templates" / "base.css.j2"
    ).read_text(encoding="utf-8")


def _visible(html: str) -> str:
    """The page without its inlined stylesheet."""
    return re.sub(r"<style>.*?</style>", "", html, flags=re.S)


def _read(name: str) -> str:
    return (paths.ROOT / "site" / name).read_text(encoding="utf-8")


def _nav_html(html: str) -> str:
    m = re.search(r"<nav class=\"uc-nav\">(.*?)</nav>", html, re.S)
    assert m, "no nav"
    return m.group(1)


def _nav(html: str) -> list[tuple[str, str]]:
    """(href, label) for the shared links — the prev/next arrows are per-page."""
    return [
        (href, label)
        for href, label in re.findall(r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', _nav_html(html))
        if "&larr;" not in label and "&rarr;" not in label
    ]


def _li(html: str, d: date) -> str:
    m = re.search(rf'<li class="uc-row[^"]*" data-date="{d}">(.*?)</li>', html, re.S)
    assert m, f"no rendered row for {d}"
    return m.group(0)
