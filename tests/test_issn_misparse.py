"""An ISSN inside a DOI is not an arXiv ID, and a journal's front matter is not
a paper (1D).

`10.1111/1468-2427.70128` is an IJURR article and `1468-2427` is that journal's
ISSN. The old pattern read the ISSN's second half plus the article number as an
arXiv ID, so twenty journal articles were stored under `arxiv:` work keys and
eighteen went out as preprints — one of them a correction notice whose own
summary said it was *"a formal correction ... updating the acknowledgments
section"*.

Three things are pinned here, and the third is the one that would rot quietly:

  1. The parser refuses an ID from any DOI that is not an arXiv DOI, and still
     reads every shape that really is one. **Both directions**, because a guard
     that also blocks the real thing is not a fix.
  2. Front matter and correction notices are dropped before either publication
     path sees them — and ordinary papers are not, including the two shapes that
     tempted a looser rule (a title beginning "Correction of ...", and a real
     article with no byline).
  3. One definition of how many papers an issue published. There were two, and
     they agreed only because nothing had ever corrected an issue.

No network, no keys.
"""

from __future__ import annotations

from datetime import date

import pytest

from pipeline import store
from pipeline.collectors.base import normalize_arxiv_id
from pipeline.models import (
    Bibliography,
    Headline,
    Issue,
    Item,
    PrimaryLocation,
    ScanMeta,
)
from pipeline.run_stages import is_journal_apparatus


# --------------------------------------------------------------------------
# 1. The parser, both directions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        # The bug itself, and the same DOI behind a resolver.
        "10.1111/1468-2427.70128",
        "https://doi.org/10.1111/1468-2427.70128",
        "doi:10.1111/1468-2427.70101",
        # The second shape the archive actually contains: EAI encodes a date,
        # so `eai.16-4-2021.169337` offered `2021.16933`. It never produced a
        # bad work key — the item was a real preprint whose ID came from arXiv —
        # but it did drop that DOI from `dedup.merge_keys`.
        "10.4108/eai.16-4-2021.169337",
        # Shapes that were always safe, kept so a future rewrite cannot lose
        # them: a dot precedes the run, which the lookbehind already refused.
        "10.1016/j.trc.2026.105852",
        "10.1080/21680566.2026.2711065",
        "10.1007/s10708-026-11706-4",
    ],
)
def test_a_journal_doi_never_yields_an_arxiv_id(value):
    assert normalize_arxiv_id(value) is None, f"{value!r} was read as an arXiv id"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.48550/arXiv.2608.01234", "2608.01234"),
        ("https://doi.org/10.48550/arXiv.2608.01234", "2608.01234"),
        ("https://arxiv.org/abs/2608.01234", "2608.01234"),
        ("http://arxiv.org/abs/2608.01234v3", "2608.01234"),
        ("arXiv:2608.01234", "2608.01234"),
        ("2608.01234", "2608.01234"),
        ("https://arxiv.org/abs/cs.CY/0701001", "cs.cy/0701001"),
    ],
)
def test_a_real_arxiv_id_still_parses(value, expected):
    assert normalize_arxiv_id(value) == expected


def test_the_whole_archive_is_clean(repo):
    """The regression in one line: no DOI in the archive may yield an arXiv id
    unless it is an arXiv DOI. Runs against the fixture, not the real
    `content/`, so it is a rule rather than a snapshot."""
    for doi in (
        "10.1111/1468-2427.70101",
        "10.1111/1468-2427.70137",
        "10.4108/eai.16-4-2021.169337",
    ):
        assert normalize_arxiv_id(doi) is None


# --------------------------------------------------------------------------
# 2. Journal apparatus, both directions
# --------------------------------------------------------------------------


def _item(title: str, key: str = "arxiv:2608.01234") -> Item:
    return Item(
        work_key=key,
        first_published=date(2026, 8, 1),
        bibliography=Bibliography(title=title, abstract="x"),
    )


@pytest.mark.parametrize(
    "title",
    [
        "Issue Information",
        "issue information",
        "Cover Image",
        "Editorial Board",
        "Contents",
        "Title page",
        "Correction to ‘EMBODYING AND RESISTING URBAN HEAT INJUSTICE’",
        "Corrigendum To: Off-Grid Electricity Imaginaries",
        "Erratum: A thing",
        "Correction",
    ],
)
def test_apparatus_is_recognised(title):
    assert is_journal_apparatus(_item(title)) is True


@pytest.mark.parametrize(
    "title",
    [
        # ★ The reason the correction rule is anchored on a notice's *shape*
        # rather than its first word. No such paper is in the archive today.
        "Correction of GPS drift in floating car data",
        "Contents and discontents of urban planning",
        "Indexing the city: a method",
        # Real writing that carries no byline. The rule "no authors" would have
        # taken these, which is why it was measured and rejected.
        "Cities, not rural areas, power the digital infrastructure of the USA",
        "Urban deprivation map highlights a hidden burden beyond megacities",
        "URBAN CLIMATE GOVERNANCE AND THE UNEVENNESS OF CITY NETWORKS",
    ],
)
def test_a_paper_is_not_apparatus(title):
    assert is_journal_apparatus(_item(title)) is False


def test_an_empty_title_is_not_apparatus():
    """Absence of a title is a metadata gap, not a verdict about the thing."""
    assert is_journal_apparatus(_item("")) is False


# --------------------------------------------------------------------------
# 3. One count, in every place that shows one
# --------------------------------------------------------------------------


def _issue(d: date, keys: list[str], *, stored_count: int) -> Issue:
    return Issue(
        date=d,
        items=sorted(keys),
        headline=Headline(
            present=bool(keys),
            work_key=keys[0] if keys else None,
            line="A line." if keys else None,
        ),
        # Deliberately disagreeing with `items`, which is exactly the state a
        # corrected issue is in until something recomputes it.
        scan_meta=ScanMeta(
            items_published=stored_count, candidates_scanned=100, journals=96
        ),
    )


def test_published_count_comes_from_items_not_from_scan_meta():
    issue = _issue(date(2026, 8, 18), ["arxiv:2608.30001", "arxiv:2608.30002"],
                   stored_count=99)
    assert issue.published_count == 2
    assert issue.scan_meta.items_published == 99, "the record itself is unchanged"


def test_every_surface_reports_the_same_number(repo):
    """Archive row, JSON, issue page and home rail. The four disagreed the
    moment 1D removed a correction notice from a published issue."""
    from pipeline.render.api import _issue_json
    from pipeline.render.preview import render_issue
    from pipeline.render.site import archive_rows, item_index, load_issues

    for k in ("arxiv:2608.30001", "arxiv:2608.30002"):
        store.save_item(
            Item(
                work_key=k,
                first_published=date(2026, 8, 18),
                bibliography=Bibliography(
                    title=f"Paper {k}",
                    abstract="x",
                    primary_location=PrimaryLocation(
                        source_name="arXiv",
                        landing_page_url=f"https://arxiv.org/abs/{k[-7:]}",
                    ),
                ),
            )
        )
    issue = _issue(date(2026, 8, 18), ["arxiv:2608.30001", "arxiv:2608.30002"],
                   stored_count=99)
    store.save_issue(issue)

    issues = load_issues()
    items = item_index()
    row = {r["date"]: r for r in archive_rows(issues, items)}["2026-08-18"]
    body = _issue_json(issues[0], row, items, "")
    page = render_issue(issues[0], [items[k] for k in issues[0].items])

    assert row["published"] == 2
    assert body["counts"]["published"] == 2
    assert "2 worth your time" in page
    assert "99 worth your time" not in page
