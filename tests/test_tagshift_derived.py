"""`tag shift` is derived at render time, for every issue (1B).

The fix in 1A corrected a real defect — the baseline counted collected papers
while the numerator counted published ones — but it only reaches days the
pipeline runs after it. Every issue already published carries the old number in
its file, and **an issue is immutable once published** (D127). D312 refused to
rewrite the archive for a smaller reason than this one.

So the value is derived instead, the way `site.py` already derives the per-issue
code and data counts and deliberately does not store them. What these tests pin
is the part that could rot quietly:

  1. The renderer must never read `issue.synthesis.deviations` again. A stored
     value and a derived one both feeding the page is the next bug, not a
     fallback — so an issue whose stored value is a lie must still render the
     truth, and an issue with no stored value at all must still render.
  2. The page and the API must come from one call. They said different things
     about nothing before (the API was silent); they must not start saying
     different things about something.
  3. `NO_BASELINE` and a measured zero are different claims and stay different.
  4. Nothing on disk changes.

No network, no keys.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta

import pytest

from pipeline import paths, store, synthesis
from pipeline.models import (
    Bibliography,
    EntityRef,
    Headline,
    Issue,
    Item,
    PrimaryLocation,
    ScanMeta,
    Synthesis,
    SynthesisDeviation,
)

TAG = "Urban Heat Island Mitigation"
OTHER = "Traffic and Road Safety"


def _item(key: str, tags: list[str]) -> Item:
    it = Item(
        work_key=key,
        first_published=date(2026, 8, 1),
        bibliography=Bibliography(
            title=f"Paper {key}",
            abstract="x",
            primary_location=PrimaryLocation(
                source_name="arXiv", landing_page_url=f"https://arxiv.org/abs/{key[-7:]}"
            ),
        ),
    )
    it.entities.topics = [
        EntityRef(id=f"openalex:T{1000 + i}", label=t) for i, t in enumerate(tags)
    ]
    store.save_item(it)
    return it


def _issue(d: date, keys: list[str], *, stored: list[SynthesisDeviation] | None = None,
           status: str = "OK") -> Issue:
    issue = Issue(
        date=d,
        items=sorted(keys),
        headline=Headline(present=bool(keys), work_key=keys[0] if keys else None,
                          line="A line." if keys else None),
        scan_meta=ScanMeta(items_published=len(keys), candidates_scanned=10, journals=96),
        synthesis=Synthesis(deviations=stored or [], deviation_status=status),
    )
    store.save_issue(issue)
    return issue


@pytest.fixture
def archive(repo):
    """Twelve quiet days, then one day where a tag spikes.

    The stored value on the spike day is deliberately **wrong** — empty, which
    is what every issue published before 2026-09-03 has — so that a renderer
    still reading it would show nothing and fail these tests.
    """
    day0 = date(2026, 8, 1)
    # Background: one paper a day carrying OTHER, so TAG's baseline is near zero
    # and OTHER's is 1.0/day.
    for i in range(12):
        k = f"arxiv:2608.1{i:04d}"
        _item(k, [OTHER])
        _issue(day0 + timedelta(days=i), [k])
    # The spike: three papers carrying TAG.
    spike = day0 + timedelta(days=12)
    keys = []
    for j in range(3):
        k = f"arxiv:2608.2{j:04d}"
        _item(k, [TAG])
        keys.append(k)
    _issue(spike, keys)
    return spike


# --------------------------------------------------------------------------
# 1. The derivation itself
# --------------------------------------------------------------------------


def test_the_archive_wide_derivation_finds_the_spike(archive):
    from pipeline.render.site import item_index, load_issues

    out = synthesis.deviations_over_archive(load_issues(), item_index())
    found = out[archive]["found"]
    assert [d["label"] for d in found] == [TAG]
    assert found[0]["today"] == 3
    assert found[0]["baseline_per_day"] == 0.0


def test_it_agrees_with_the_single_day_function(archive):
    """Two callers, one rule. `_verdict` exists so this cannot drift."""
    from pipeline.render.site import item_index, load_issues

    issues = load_issues()
    items = item_index()
    bulk = synthesis.deviations_over_archive(issues, items)
    for issue in issues:
        day = [items[k] for k in issue.items if k in items]
        assert synthesis.deviations(issue.date, day) == bulk[issue.date], issue.date


def test_a_short_archive_is_no_baseline_not_a_zero(archive):
    """The first days have nothing behind them, and say so rather than
    reporting every tag as a spike against zero."""
    from pipeline.render.site import item_index, load_issues

    out = synthesis.deviations_over_archive(load_issues(), item_index())
    early = sorted(out)[0]
    assert out[early]["status"] == "NO_BASELINE"
    assert out[early]["found"] == []
    assert out[archive]["status"] == "OK"


# --------------------------------------------------------------------------
# 2. The renderer reads the derivation, never the file
# --------------------------------------------------------------------------


def test_the_page_ignores_a_stored_value_that_disagrees(archive):
    """The whole point. A published issue's stored number is not consulted."""
    issue = store.load_issue(archive)
    issue.synthesis.deviations = [
        SynthesisDeviation(label="A LIE", today=99, baseline_per_day=0.1, window_days=30)
    ]
    store.save_issue(issue)

    from pipeline.render.preview import render_issue

    items = [store.load_item(k) for k in issue.items]
    html = render_issue(issue, items)
    assert "A LIE" not in html, "the renderer is still reading the stored value"
    assert TAG in html


def test_a_stored_no_baseline_does_not_silence_a_derivable_day(archive):
    """Four real backfilled days are in exactly this state: recorded
    `NO_BASELINE` because the archive was short when they were assembled, and
    comparable now. The row's `measurable` flag has to follow the derivation
    too, or the row goes silent while its entries exist."""
    issue = store.load_issue(archive)
    issue.synthesis.deviation_status = "NO_BASELINE"
    issue.synthesis.deviations = []
    store.save_issue(issue)

    from pipeline.render.preview import build_synthesis

    items = [store.load_item(k) for k in issue.items]
    rows = {r["label"]: r for r in build_synthesis(issue, items)["rows"]}
    assert rows["tag shift"]["measurable"] is True
    assert [d["label"] for d in rows["tag shift"]["entries"]] == [TAG]


# --------------------------------------------------------------------------
# 3. The page and the JSON say the same thing
# --------------------------------------------------------------------------


def test_the_api_serves_the_same_value_as_the_page(archive):
    from pipeline.render.api import build_api
    from pipeline.render.site import build_issue_pages

    build_issue_pages()
    build_api()

    body = json.loads(
        (paths.ROOT / "site" / "api" / "issues" / f"{archive}.json").read_text(encoding="utf-8")
    )
    shift = body["synthesis"]["tag_shift"]
    assert shift["status"] == "OK"
    assert [d["label"] for d in shift["found"]] == [TAG]

    page = (paths.ROOT / "site" / "issues" / f"{archive}.html").read_text(encoding="utf-8")
    assert TAG in page
    for d in shift["found"]:
        assert str(d["today"]) in page


# --------------------------------------------------------------------------
# 4. And nothing on disk moved
# --------------------------------------------------------------------------


def test_rendering_does_not_rewrite_a_single_issue(archive):
    """D127, asserted rather than trusted. Deriving is only worth doing if it
    leaves the archive alone, and a renderer that saved would be the exact
    thing D312 refused."""
    before = {
        p.name: p.read_bytes()
        for p in sorted((paths.CONTENT / "issues").glob("*.json"))
    }
    items_before = {
        p.name: p.read_bytes()
        for p in sorted((paths.CONTENT / "items").glob("*.json"))
    }

    from pipeline.render.api import build_api
    from pipeline.render.site import build_issue_pages

    build_issue_pages()
    build_api()

    after = {
        p.name: p.read_bytes()
        for p in sorted((paths.CONTENT / "issues").glob("*.json"))
    }
    items_after = {
        p.name: p.read_bytes()
        for p in sorted((paths.CONTENT / "items").glob("*.json"))
    }
    assert before == after, "a render rewrote an issue"
    assert items_before == items_after, "a render rewrote an item"


# --------------------------------------------------------------------------
# 5. The cost, because "derive it every build" is only true while it is cheap
# --------------------------------------------------------------------------


def test_the_derivation_walks_each_item_once(archive):
    """A guard on shape, not a stopwatch — a timing assertion would be flaky in
    CI. The failure this prevents is real: calling `deviations()` once per page
    would re-read every item file once per issue, which is how 0.02% of a build
    becomes most of it."""
    from pipeline.render.site import item_index, load_issues

    issues = load_issues()
    items = item_index()
    seen: Counter = Counter()

    class CountingDict(dict):
        def get(self, k, default=None):  # noqa: D102
            seen[k] += 1
            return super().get(k, default)

    synthesis.deviations_over_archive(issues, CountingDict(items))
    assert seen, "the derivation never looked at an item"
    assert max(seen.values()) == 1, (
        f"an item was read {max(seen.values())} times; the per-day index is not being reused"
    )
