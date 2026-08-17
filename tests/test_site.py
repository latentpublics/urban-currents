"""Home, archive and the design-review file (phase 0j, W4).

The pages are the schema test: they ask `content/` for things a daily issue
never had to answer. What is pinned here is that the derived answers stay
derived, that the aggregate excludes what belongs to no issue, and that no
number in the chrome is written into a template.
"""

from __future__ import annotations

import re
from datetime import date

from pipeline import store
from pipeline.models import Bibliography, Headline, Issue, Item, ScanMeta
from pipeline.render.site import archive_rows, archive_stats, build_archive, build_home


def _item(key: str, badges: list[str], title: str = "A Paper") -> Item:
    it = Item(
        work_key=key,
        first_published=date(2026, 8, 11),
        bibliography=Bibliography(title=title),
    )
    it.badges = badges
    return it


def _seed(repo, published: list[Item], orphans: list[Item] = ()) -> Issue:
    for it in list(published) + list(orphans):
        store.save_item(it, today=date(2026, 8, 11))
    issue = Issue(
        date=date(2026, 8, 11),
        items=sorted(it.work_key for it in published),
        headline=Headline(
            present=True,
            work_key=published[0].work_key if published else None,
            line="A line.",
        ),
        scan_meta=ScanMeta(items_published=len(published), unreadable_count=0),
    )
    store.save_issue(issue)
    return issue


def test_code_and_data_counts_are_derived_not_stored(repo):
    """The counts a row needs are not on the Issue, and are not added to it.

    An issue is immutable once published, so a field added today is empty for
    every issue already written — the archive would show counts for new days and
    blanks for old ones, which reads as a data gap rather than a schema change.
    """
    _seed(repo, [_item("arxiv:2608.00001", ["code", "data"]), _item("arxiv:2608.00002", ["data"])])

    rows = archive_rows()
    assert rows[0]["code"] == 1
    assert rows[0]["data"] == 2

    issue = store.load_issue(date(2026, 8, 11))
    assert not hasattr(issue.scan_meta, "code_count")
    assert "code" not in issue.scan_meta.model_dump()


def test_items_belonging_to_no_issue_are_counted_apart(repo):
    """12 such items exist in the real archive (D127). They must not inflate anything."""
    _seed(
        repo,
        [_item("arxiv:2608.00001", ["data"])],
        orphans=[_item("arxiv:2608.09999", ["code"])],
    )

    rows = archive_rows()
    stats = archive_stats(rows, {it.work_key: it for it in store.iter_items()})

    assert rows[0]["published"] == 1
    assert rows[0]["code"] == 0          # the orphan's badge is not in the day
    assert stats["orphan_items"] == 1    # but it is reported
    assert stats["items_total"] == 2


def test_a_quiet_day_keeps_its_row(repo):
    _seed(repo, [_item("arxiv:2608.00001", [])])
    quiet = Issue(date=date(2026, 8, 12), items=[], quiet_day=True)
    store.save_issue(quiet)

    rows = {r["date"]: r for r in archive_rows()}
    assert "2026-08-12" in rows
    assert rows["2026-08-12"]["quiet"] is True


def test_the_publish_range_comes_from_the_archive(repo):
    for n, day in ((3, 11), (7, 12), (5, 13)):
        items = [_item(f"arxiv:2608.{day}{i:03d}", []) for i in range(n)]
        for it in items:
            store.save_item(it, today=date(2026, 8, day))
        store.save_issue(Issue(
            date=date(2026, 8, day),
            items=sorted(it.work_key for it in items),
            scan_meta=ScanMeta(items_published=n),
        ))

    rows = archive_rows()
    stats = archive_stats(rows, {it.work_key: it for it in store.iter_items()})
    assert stats["published_days"] == 3
    assert stats["range_low"] <= 5 <= stats["range_high"]


def test_no_number_in_the_chrome_is_written_into_the_template(repo):
    """The mockup's own copy says 159 journals and 15-24 papers. Both are wrong.

    A page whose description of itself disagrees with its own data is the thing
    this service exists not to be, so the template holds no figures at all.
    """
    from pathlib import Path

    import pipeline.render.site as site_mod

    home = Path(site_mod.TEMPLATE_DIR / "home.html.j2").read_text(encoding="utf-8")
    body = re.sub(r"\{#.*?#\}", "", home, flags=re.S)      # drop comments
    body = re.sub(r"\{\{.*?\}\}", "", body, flags=re.S)     # drop expressions
    for forbidden in ("159", "15–24", "15-24", "24 papers"):
        assert forbidden not in body, f"{forbidden!r} is hardcoded in the template"


def test_the_pages_make_no_external_request(repo):
    _seed(repo, [_item("arxiv:2608.00001", ["data"])])
    for path in (build_home(), build_archive()):
        html = path.read_text(encoding="utf-8")
        assert "<link" not in html
        assert "<script" not in html
        assert "url(http" not in html
        assert "@font-face" not in html
        assert html.count("<h1") == 1


def test_a_quiet_day_says_so_in_words_not_only_in_colour(repo):
    _seed(repo, [_item("arxiv:2608.00001", [])])
    store.save_issue(Issue(date=date(2026, 8, 12), items=[], quiet_day=True))
    html = build_archive().read_text(encoding="utf-8")
    assert "a quiet day" in html
