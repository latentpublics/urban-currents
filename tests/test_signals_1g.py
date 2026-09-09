"""The four signals 1C measured and 1G implemented.

1C surveyed every population the pipeline could count over and recommended one
thing per signal. This file holds the recommendations as assertions, because
each of them is a number that will drift:

  G1  `on the same shoulders` — revived on **every paper the day named** at a
      threshold of four. Over the papers we published it found nothing on any
      of fourteen days, which is why 1A took it off the page; the rule was never
      the problem, the population was.
  G2  `shares references` — the old `coupling`, same population, literal name.
      🔴 Two rows about citation now stand next to each other, and the risk
      1G exists to manage is a reader adding them together. Each says what it
      counted over; the section says the difference once in words.
  G3  the preprint → journal transition — **not a new signal**. The material was
      already in the archive and `status_changes` was empty in all eighty
      issues, because the transition happens in `enrich` and only `stage_issue`
      was writing anything down.
  G4  `first appearance` — institutions only, and the window on the screen is
      the window the archive actually has.

No network, no keys.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from pipeline import paths, store, synthesis
from pipeline.models import (
    Author,
    Bibliography,
    Headline,
    Institution,
    Issue,
    Item,
    PrimaryLocation,
    PublicationStatus,
    ScanMeta,
    Synthesis,
    SynthesisAnchor,
)

CANON = "openalex:W1987233492"
CANON_TITLE = "Travel demand and the 3Ds: Density, diversity, and design"
DAY = date(2026, 9, 8)


# --------------------------------------------------------------------------
# Fixture material
# --------------------------------------------------------------------------


def _item(key: str, *, title: str = "A paper", institutions: list[str] = ()) -> Item:
    it = Item(
        work_key=key,
        first_published=DAY,
        bibliography=Bibliography(
            title=title,
            authors=[
                Author(
                    name="A Researcher",
                    institutions=[Institution(name=n) for n in institutions],
                )
            ],
            publication_date=DAY,
            primary_location=PrimaryLocation(source_name="A Journal", type="journal"),
        ),
    )
    store.save_item(it)
    return it


def _canon_file(*ids: str) -> None:
    (paths.CONTENT / "canon").mkdir(parents=True, exist_ok=True)
    (paths.CONTENT / "canon" / "candidates.json").write_text(
        json.dumps({
            "candidates": [
                {
                    "openalex_id": i,
                    "class": "foundation",
                    "title": CANON_TITLE if i == CANON else f"Another foundation {i}",
                    "authors": ["Robert Cervero", "Kara M. Kockelman"],
                    "publication_date": "1997-09-01",
                }
                for i in ids
            ]
        }),
        encoding="utf-8",
    )


def _reference_base(rows: dict[str, list[str]]) -> None:
    paths.GRAPH.mkdir(parents=True, exist_ok=True)
    (paths.GRAPH / "references.jsonl").write_text(
        "\n".join(
            json.dumps({"work_key": k, "referenced_works": v}) for k, v in rows.items()
        )
        + "\n",
        encoding="utf-8",
    )


def _issue(d: date, items: list[str], unreadable: list[str] = ()) -> Issue:
    issue = Issue(
        date=d,
        headline=Headline(present=True, line="A line about the day.",
                          work_key=items[0] if items else None),
        scan_meta=ScanMeta(items_published=len(items), journals=96,
                           candidates_scanned=100,
                           unreadable_count=len(list(unreadable))),
        items=sorted(items),
        unreadable=sorted(unreadable),
        synthesis=Synthesis(deviation_status="OK"),
    )
    store.save_issue(issue)
    return issue


@pytest.fixture()
def a_day_standing_on_one_work(repo):
    """Two published papers and three we could not read, four citing the same
    foundation work. The split is the whole point of G1: on the published two
    the rule finds nothing, and on all five it finds one."""
    _canon_file(CANON)
    published = [_item("doi:10.1/a"), _item("doi:10.1/b")]
    unread = [_item(f"doi:10.1/u{i}") for i in range(3)]
    _reference_base({
        "doi:10.1/a": [CANON, "openalex:W1"],
        "doi:10.1/b": ["openalex:W2"],
        "doi:10.1/u0": [CANON],
        "doi:10.1/u1": [CANON],
        "doi:10.1/u2": [CANON],
    })
    issue = _issue(DAY, [i.work_key for i in published],
                   [i.work_key for i in unread])
    from pipeline.render import preview as preview_mod

    preview_mod._REFERENCE_KEYS = None  # module-level cache, per-test root
    return issue


def _table(issue: Issue) -> dict:
    index = {it.work_key: it for it in store.iter_items()}
    return synthesis.shoulders_over_archive([issue], index)[issue.date]


# --------------------------------------------------------------------------
# G1 — the population, not the rule
# --------------------------------------------------------------------------


def test_it_counts_every_paper_the_day_named(a_day_standing_on_one_work):
    """Four of the five cite it; two of them are papers we published.

    1C measured the same rule on the two populations over fourteen days:
    **0 entries on 0 days** over the published papers, 1.1 a day on 9 of 14 over
    every paper named. The row was taken off the page in 1A for being empty;
    this is the reading that says it was empty for a reason a threshold cannot
    reach.
    """
    row = _table(a_day_standing_on_one_work)

    assert row["measurable"] is True
    assert [e["title"] for e in row["entries"]] == [CANON_TITLE]
    assert row["entries"][0]["count"] == 4
    assert row["named"] == 5, "two published and three we could not read"
    assert row["with_references"] == 5


def test_the_same_rule_over_the_published_papers_alone_finds_nothing(
    a_day_standing_on_one_work,
):
    """The control. Without this the test above only says the code runs."""
    issue = a_day_standing_on_one_work
    published_only = issue.model_copy(update={"unreadable": []})

    assert _table(published_only)["entries"] == []


def test_three_is_not_enough(a_day_standing_on_one_work):
    """Four is the threshold 1C's sweep landed on, and it is a number to
    re-measure rather than a constant to trust — so it is asserted from both
    sides."""
    issue = a_day_standing_on_one_work
    index = {it.work_key: it for it in store.iter_items()}

    at_five = synthesis.shoulders_over_archive([issue], index, min_citing=5)
    assert at_five[DAY]["entries"] == []
    at_three = synthesis.shoulders_over_archive([issue], index, min_citing=3)
    assert len(at_three[DAY]["entries"]) == 1


def test_too_few_bibliographies_is_absence_not_zero(repo):
    """The distinction this repository keeps making. A day whose papers carry no
    reference list cannot be asked whether four of them cited the same work, and
    "no foundation work was shared" is not the honest thing to print."""
    _canon_file(CANON)
    _reference_base({})
    issue = _issue(DAY, [_item("doi:10.1/x").work_key])

    row = _table(issue)
    assert row["measurable"] is False
    assert row["entries"] == []


def test_the_page_derives_it_and_does_not_read_the_issue(a_day_standing_on_one_work):
    """D318, and the shape 1B settled for `tag shift`.

    `issue.synthesis.anchors` holds what the morning's run computed — at a
    threshold of two, over the papers it published — and an issue is immutable
    once published (D127). So the file keeps its number and the page shows the
    archive's answer. Here the stored value is deliberately a work that does not
    exist, and it must not reach the screen.
    """
    from pipeline.render.preview import render_issue

    issue = a_day_standing_on_one_work
    issue.synthesis.anchors = [
        SynthesisAnchor(openalex_id="openalex:W999", title="A stored anchor nobody measured",
                        citing_today=2)
    ]
    store.save_issue(issue)

    html = render_issue(store.load_issue(DAY),
                        [store.load_item(k) for k in issue.items])
    assert "A stored anchor nobody measured" not in html
    assert CANON_TITLE in html


# --------------------------------------------------------------------------
# G2 — two rows about citation, and the reader is not asked to add them up
# --------------------------------------------------------------------------


def _html(issue: Issue) -> str:
    from pipeline.render.preview import render_issue

    return render_issue(issue, [store.load_item(k) for k in issue.items])


def test_the_identifier_stays_and_only_the_caption_moves(a_day_standing_on_one_work):
    """1C's recommendation, and the reason for it: `coupling` is the
    configuration key `citation.min_shared_references`, twenty-seven call sites
    and the `data-row=` hook the tests read, and "bibliographic coupling" is the
    correct term for what it computes. A caption is not a reason to rename a
    measurement."""
    html = _html(a_day_standing_on_one_work)

    assert 'data-row="coupling"' in html
    assert "<dt class=\"uc-synthesis__label\">shares references</dt>" in html
    assert "<dt class=\"uc-synthesis__label\">coupling</dt>" not in html

    assert 'data-row="canon"' in html
    assert "<dt class=\"uc-synthesis__label\">on the same shoulders</dt>" in html


def test_each_row_says_what_it_counted_over(a_day_standing_on_one_work):
    """🔴 The risk this batch was told to manage. Two rows about citation, side
    by side, over different sets of papers — and nothing in the arithmetic tells
    a reader they are not the same five papers counted twice."""
    html = _html(a_day_standing_on_one_work)

    assert "counted over the 2 papers published above" in html
    # The apostrophe is `&#39;` by the time it is markup, so the assertion is
    # split around it rather than written in the shape the template has.
    assert "counted over the 5 of this day" in html
    assert "5 named papers whose reference lists we have" in html


def test_the_difference_is_also_said_once_in_words(a_day_standing_on_one_work):
    """A denominator is a fact; the sentence is what makes it a distinction."""
    html = _html(a_day_standing_on_one_work)

    assert "Two counts, not one counted twice" in html
    assert "every paper this day named" in html


def test_the_metaphor_is_the_only_one_and_it_is_explained(a_day_standing_on_one_work):
    """The site has no other figure of speech, so this row's caption carries a
    risk the others do not: `on the same shoulders` can be read as "about the
    same thing". The words beside it deny exactly that, and they are the reason
    the caption is allowed."""
    html = _html(a_day_standing_on_one_work)

    assert "standing on the same older work is not the same as being about the same thing" in html


def test_the_sentence_is_absent_when_only_one_kind_of_row_is(repo):
    """A sentence explaining the difference between two rows, one of which is
    not on the page, is the noise the archive's legend rule exists to keep
    off."""
    _canon_file(CANON)
    _reference_base({})
    issue = _issue(DAY, [_item("doi:10.1/x").work_key])
    from pipeline.render import preview as preview_mod

    preview_mod._REFERENCE_KEYS = None

    html = _html(issue)
    assert "Two counts, not one counted twice" not in html


# --------------------------------------------------------------------------
# G4 — the window is the one we have
# --------------------------------------------------------------------------


def _archive_of(days: int, *, institutions_on_last: list[str]) -> list[Issue]:
    """`days` consecutive issues ending on DAY, each with one paper."""
    issues = []
    for i in range(days):
        d = DAY - timedelta(days=days - 1 - i)
        insts = institutions_on_last if d == DAY else ["An Old University"]
        item = _item(f"doi:10.1/{d}", institutions=insts)
        item.first_published = d
        store.save_item(item)
        issues.append(_issue(d, [item.work_key]))
    return issues


def _firsts(issues: list[Issue]) -> dict:
    index = {it.work_key: it for it in store.iter_items()}
    return synthesis.first_seen_institutions_over_archive(issues, index)


def test_the_window_is_what_the_archive_has_not_what_we_asked_for(repo):
    """🔴 The screen must not claim a lookback the archive does not hold.

    1C measured N=90 and N=180 returning identical values on an 84-day archive —
    both truncate at the same floor — so a template printing "180 days" would
    have been stating a number nobody had.
    """
    issues = _archive_of(70, institutions_on_last=["A New Institute"])
    row = _firsts(issues)[DAY]

    assert row["measurable"] is True
    assert row["window_days"] == 69, "the archive behind this issue, not the ceiling"
    assert [e["name"] for e in row["entries"]] == ["A New Institute"]


def test_the_ceiling_still_applies_once_the_archive_passes_it(repo):
    """And the other side of it: the window stops growing at 180."""
    issues = _archive_of(200, institutions_on_last=["A New Institute"])
    assert _firsts(issues)[DAY]["window_days"] == synthesis.FIRST_SEEN_WINDOW_DAYS


def test_a_young_archive_gets_no_row_at_all(repo):
    """At two weeks of archive, 22% of a day's tags have never been seen — a
    measurement of how young we are wearing the costume of a finding. 1C put the
    floor at sixty days, where the rate falls to 11%."""
    issues = _archive_of(30, institutions_on_last=["A New Institute"])
    row = _firsts(issues)[DAY]

    assert row["measurable"] is False
    assert row["entries"] == []


def test_the_window_is_measured_backwards_from_the_issues_own_date(repo):
    """🔴 D318's rule applied to a window: an issue published in July must not
    change what it says because it is now September. Every issue here has a
    different amount of archive behind it, and each reports its own."""
    issues = _archive_of(80, institutions_on_last=["A New Institute"])
    table = _firsts(issues)

    assert table[DAY]["window_days"] == 79
    assert table[DAY - timedelta(days=10)]["window_days"] == 69
    # And the ones with less than sixty days behind them stay silent even
    # though the archive as a whole is now long enough.
    assert table[DAY - timedelta(days=70)]["measurable"] is False


def test_the_row_prints_the_window_it_used(repo):
    """The number in the sentence and the number in the derivation are the same
    number, asserted together rather than each on its own."""
    _archive_of(70, institutions_on_last=["A New Institute"])
    from pipeline.render import preview as preview_mod

    preview_mod._REFERENCE_KEYS = None
    html = _html(store.load_issue(DAY))

    assert "in the 69 days before this issue" in html
    assert "180 days" not in html


# --------------------------------------------------------------------------
# G3 — the transition that was happening and nothing was writing down
# --------------------------------------------------------------------------


def _preprint(key: str = "arxiv:2608.01234") -> Item:
    it = Item(
        work_key=key,
        first_published=date(2026, 8, 1),
        ids=__import__("pipeline.models", fromlist=["Ids"]).Ids(arxiv=key.split(":", 1)[1]),
        bibliography=Bibliography(
            title="A preprint that later came out",
            authors=[Author(name="A Researcher")],
            publication_date=date(2026, 8, 1),
            primary_location=PrimaryLocation(source_name="arXiv", type="repository"),
        ),
        publication_status=PublicationStatus(state="preprint"),
    )
    store.save_item(it)
    return it


JOURNAL_WORK = {
    "id": "https://openalex.org/W4392000001",
    "type": "article",
    "primary_location": {
        "source": {
            "id": "https://openalex.org/S1234",
            "display_name": "Cities",
            "type": "journal",
        }
    },
}


def _enrich_once(d: date, item: Item, work: dict | None = JOURNAL_WORK):
    """The real enrich path with the network replaced, and nothing else."""
    from pipeline.collectors.openalex import OpenAlexCollector, EnrichQueue
    from pipeline.metrics import Run

    run = Run.for_date(d)
    collector = OpenAlexCollector(run)
    collector.find_work_for_arxiv = lambda *a, **k: work  # type: ignore[method-assign]
    EnrichQueue({item.work_key: 0}).save()
    collector.enrich([], max_lookups=10)
    return run


def test_the_transition_is_recorded_where_it_actually_happens(repo):
    """🔴 The finding behind G3. `status_changes` was empty in all eighty issues
    while the archive filled with preprints that had come out in journals, and
    the reason was not the matching rule — it was that the state changes in
    `enrich`, which writes the Item straight to `content/items/` and hands
    nothing to `stage_issue`.
    """
    from pipeline.stages import read_status_changes

    item = _preprint()
    _issue(date(2026, 8, 1), [item.work_key])
    run = _enrich_once(DAY, item)

    assert store.load_item(item.work_key).publication_status.state == "published"
    (change,) = read_status_changes(run)
    assert change.work_key == item.work_key
    assert (change.from_, change.to, change.journal) == ("preprint", "published", "Cities")


def test_a_paper_no_issue_ever_carried_is_not_announced(repo):
    """Precision over recall, and a reader who can follow the line. The section
    lists a work_key; a work_key nothing in the archive published is a change
    announced about a paper the reader cannot find."""
    from pipeline.stages import read_status_changes

    item = _preprint()  # never in an issue
    run = _enrich_once(DAY, item)

    assert store.load_item(item.work_key).publication_status.state == "published"
    assert read_status_changes(run) == []


def test_only_the_one_direction(repo):
    """A preprint staying a preprint is not a transition, and a `published`
    record turning back into a preprint is a data fault rather than news."""
    from pipeline.stages import read_status_changes

    item = _preprint()
    _issue(date(2026, 8, 1), [item.work_key])
    run = _enrich_once(DAY, item, work={"id": "https://openalex.org/W1", "type": "preprint"})

    assert store.load_item(item.work_key).publication_status.state == "preprint"
    assert read_status_changes(run) == []


def test_running_the_day_twice_does_not_double_the_line(repo):
    """Idempotence is a promise about `content/`, and this file feeds it."""
    from pipeline.stages import read_status_changes, record_status_change
    from pipeline.metrics import Run
    from pipeline.models import StatusChange

    run = Run.for_date(DAY)
    for _ in range(3):
        record_status_change(run, StatusChange(
            work_key="arxiv:2608.01234", **{"from": "preprint"}, to="published",
            journal="Cities",
        ))

    assert len(read_status_changes(run)) == 1


def test_the_issue_carries_what_the_run_recorded(repo):
    """The other half: `enrich` writes it down and `stage_issue` publishes it.

    Without this the transition is a line in a run directory, and run
    directories are gitignored and die with the runner (D261).
    """
    from pipeline import run_stages
    from pipeline.metrics import Run
    from pipeline.models import StatusChange
    from pipeline.stages import record_status_change, write_stage

    item = _preprint()
    _issue(date(2026, 8, 1), [item.work_key])

    run = Run.for_date(DAY)
    record_status_change(run, StatusChange(
        work_key=item.work_key, **{"from": "preprint"}, to="published",
        journal="Cities",
    ))
    write_stage(run, "issue", [])
    issue = run_stages.stage_issue(run, DAY, use_llm=False)

    assert [c.work_key for c in issue.status_changes] == [item.work_key]
    assert store.load_issue(DAY).status_changes[0].journal == "Cities"
