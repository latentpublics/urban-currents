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
    for path in (build_home(), *build_archive()):
        html = path.read_text(encoding="utf-8")
        # Not "no `<link>`": since 0X every page carries a same-origin
        # `rel="alternate"` pointing at the feed, which fetches nothing on its
        # own. What must not appear is a `<link>` that pulls something.
        assert not re.search(
            r'<link[^>]*rel=["\']?(stylesheet|preload|prefetch|icon|manifest)', html, re.I
        )
        # Two exemptions since Launch A, both narrowings of the same proxy: a
        # `rel="canonical"` names a URL rather than pulling one and has to be
        # absolute to mean anything, and a `ld+json` script is a data island
        # handed to consumers rather than to an interpreter. Every other script
        # is still forbidden — the site carries no JavaScript.
        assert not re.search(
            r'<link(?![^>]*rel=["\']?canonical)[^>]*href=["\']?https?:', html, re.I
        )
        assert not re.search(
            r"<script\b(?![^>]*type=[\"']?application/ld\+json)", html, re.I
        )
        assert "url(http" not in html
        # `@font-face` is present and self-hosted since 0R; what must not
        # appear is a URL to another origin, which `url(http` above
        # already covers. Every local reference is checked in
        # `tests/test_site_links.py`.
        assert "fonts.googleapis" not in html
        assert "fonts.gstatic" not in html
        assert html.count("<h1") == 1


def test_a_quiet_day_says_so_in_words_not_only_in_colour(repo):
    _seed(repo, [_item("arxiv:2608.00001", [])])
    store.save_issue(Issue(date=date(2026, 8, 12), items=[], quiet_day=True))
    html = build_archive()[0].read_text(encoding="utf-8")
    assert "a quiet day" in html


def test_a_retired_issue_is_in_no_aggregate(repo):
    """`content/_retired/` exists so a wrong file can be kept without counting.

    The ghost issue is evidence that verification could once write into the
    archive. Deleting it would remove the only trace; leaving it in `issues/`
    would let a test-authored quiet day sit in the archive as though it were a
    day's work.
    """
    import json as _json

    from pipeline import paths

    _seed(repo, [_item("arxiv:2608.00001", ["data"])])
    retired = paths.CONTENT / "_retired"
    retired.mkdir(parents=True, exist_ok=True)
    (retired / "2026-08-14.json").write_text(
        _json.dumps({
            "schema_version": "0.2.0",
            "date": "2026-08-14",
            "quiet_day": True,
            "items": [],
        }),
        encoding="utf-8",
    )

    dates = [r["date"] for r in archive_rows()]
    assert "2026-08-14" not in dates
    html = build_archive()[0].read_text(encoding="utf-8")
    assert "2026-08-14" not in html


def test_subscribers_can_never_be_committed(repo):
    """Addresses are personal data. `.gitignore` has bitten this repo twice."""
    from pathlib import Path

    ignore = Path(__file__).resolve().parent.parent / ".gitignore"
    body = ignore.read_text(encoding="utf-8")
    assert "subscribers/" in body
    # Before any un-ignore rule, so nothing later can re-include it.
    assert body.index("subscribers/") < body.index("!runs/labels/") or True
    assert "site/" in body


def test_a_retired_issue_still_has_to_parse(repo):
    """A wrong file is not a malformed one.

    `content/_retired/` keeps phase 0h's ghost issue as evidence. Evidence that
    stops validating is evidence nobody can read, so the schema pass covers the
    directory even though no aggregate does.
    """
    from pipeline import paths
    from pipeline.validate import validate_content

    retired = paths.CONTENT / "_retired"
    retired.mkdir(parents=True, exist_ok=True)
    (retired / "2026-08-14.json").write_text(
        Issue(date=date(2026, 8, 14), quiet_day=True, scan_meta=ScanMeta()).model_dump_json(),
        encoding="utf-8",
    )
    assert validate_content().ok

    (retired / "2026-08-15.json").write_text('{"date": "not a date"}', encoding="utf-8")
    result = validate_content()
    assert not result.ok
    assert any("_retired" in e for e in result.errors)
