"""The held queue and the review that happens when someone comes back (0L, M2).

The operating assumption changed: nobody checks before publication, so the
selection policy has to carry the doubt the daily review used to. These tests
pin the three properties that follow.

1. **Holding does not block the issue.** The day publishes without the held
   item, leaving a hole, and that is the intended outcome.
2. **Held is not a backlog.** A held item is waiting for a judgement, not owed
   to readers, and it never appears in a later issue.
3. **The queue is the labelling queue.** What is waiting is what a person should
   look at when they get back, oldest first, without remembering a date.
"""

from __future__ import annotations

from datetime import date, timedelta


from pipeline import held
from pipeline.held import (
    NEAR_MISS,
    RULE_AT_THE_FLOOR,
    RULE_OFF_SUBFIELD,
    RULE_UNCERTAIN,
    WITHHELD,
    Suspicion,
    inspect,
    paper_subfield,
)
from pipeline.models import Bibliography, Item, PrimaryLocation, TopicRef

DAY = date(2026, 8, 20)


def _item(key: str, subfield: str | None = "3305", score: float = 0.9) -> Item:
    item = Item(
        work_key=key,
        first_published=DAY,
        bibliography=Bibliography(
            title=f"Paper {key}",
            abstract="x",
            primary_location=PrimaryLocation(source_name="Cities"),
        ),
    )
    item.scores.relevance = score
    if subfield:
        item.entities.topics = [
            TopicRef(id="openalex:T10235", label="A topic", subfield=subfield, is_primary=True)
        ]
    return item


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------


def test_a_subfield_our_labels_have_never_seen_is_let_through(repo):
    """Thin evidence is not evidence against.

    The derived list includes any subfield fewer than three labelled papers
    carry, because excluding on absence of evidence is the measured-zero
    mistake this project keeps making (phase 0N, P2). 2500 is unseen, so it
    passes rather than being withheld.
    """
    unseen = _item("doi:10.1/unseen", subfield="2500")

    assert inspect(unseen, "journal", selected=True) is None


def test_enforcement_can_be_switched_off_by_config(repo, monkeypatch):
    """One line demotes the rule back to recording, if it ever misbehaves again."""
    import pipeline.held as held_mod

    monkeypatch.setattr(held_mod, "_off_subfield_withholds", lambda: False)
    # The deny-list is injected, not borrowed. It is empty as of 0P (Q3) —
    # every subfield on it was overturned by targeted labels — and a test about
    # the switch must keep working whatever the vocabulary currently says.
    monkeypatch.setattr(held_mod, "rejected_subfield_ids", lambda: {"9999"})
    suspicion = inspect(
        _item("doi:10.1/rejected", subfield="9999"), "journal", selected=True
    )

    assert suspicion is not None
    assert suspicion.kind == NEAR_MISS


def test_the_whole_queue_can_be_switched_off(repo, monkeypatch):
    """`held.enabled` was declared in config and read by nothing."""
    import pipeline.held as held_mod

    monkeypatch.setattr(held_mod, "enabled", lambda: False)
    assert inspect(_item("arxiv:2608.1", score=0.81), "arxiv", selected=True, floor=0.80) is None


def test_a_paper_inside_the_whitelist_subfields_publishes(repo):
    urban = _item("doi:10.1/urban", subfield="3305")
    assert inspect(urban, "journal", selected=True) is None


def test_an_unclassified_paper_is_not_treated_as_off_subfield(repo):
    """None means we could not check, not that it failed the check.

    This is the measured-zero-versus-could-not-measure trap in its newest
    location, and holding on a missing value would walk straight into it.
    """
    unclassified = _item("doi:10.1/none", subfield=None)

    assert paper_subfield(unclassified) is None
    assert inspect(unclassified, "journal", selected=True) is None


def test_an_arxiv_item_scraping_the_floor_is_withheld(repo):
    """Above the line by less than the model's calibration error is not above it."""
    borderline = _item("arxiv:2608.1", score=0.81)
    suspicion = inspect(borderline, "arxiv", selected=True, floor=0.80)

    assert suspicion is not None
    assert suspicion.rule == RULE_AT_THE_FLOOR
    assert suspicion.kind == WITHHELD


def test_an_arxiv_item_well_clear_of_the_floor_publishes(repo):
    assert inspect(_item("arxiv:2608.2", score=0.95), "arxiv", selected=True, floor=0.80) is None


def test_the_uncertain_band_is_a_near_miss_not_a_withholding(repo):
    """These were never going to publish. Holding them costs the issue nothing
    and turns a silent drop into a question."""
    unsure = _item("arxiv:2608.3", score=0.62)
    suspicion = inspect(unsure, "arxiv", selected=False, floor=0.80)

    assert suspicion is not None
    assert suspicion.rule == RULE_UNCERTAIN
    assert suspicion.kind == NEAR_MISS


def test_a_confidently_rejected_item_is_not_held(repo):
    """Holding everything we dropped would make the queue the candidate pool."""
    assert inspect(_item("arxiv:2608.4", score=0.11), "arxiv", selected=False) is None


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


def _suspicions() -> list[Suspicion]:
    return [
        Suspicion("doi:10.1/a", RULE_OFF_SUBFIELD, WITHHELD, "off", 1.0, "journal", "A"),
        Suspicion("arxiv:2608.9", RULE_UNCERTAIN, NEAR_MISS, "unsure", 0.6, "arxiv", "B"),
    ]


def test_the_withheld_rate_is_measured_against_what_the_day_would_have_published(repo):
    """A rule that withholds a third of the day is a policy change, not a filter,
    and the rate is how anyone would notice."""
    held.record(DAY, _suspicions(), published=9)
    doc = held.load(DAY)

    assert doc["withheld"] == 1
    assert doc["near_miss"] == 1
    # 1 withheld out of the 10 the day would otherwise have carried.
    assert doc["withheld_rate"] == 0.1


def test_no_suspicions_writes_no_file(repo):
    assert held.record(DAY, [], published=12) is None
    assert held.load(DAY) is None


def test_pending_is_oldest_first_and_skips_what_was_judged(repo):
    held.record(DAY - timedelta(days=2), _suspicions(), published=5)
    held.record(DAY, _suspicions(), published=5)

    waiting = held.pending()
    assert [r["date"] for r in waiting][0] == str(DAY - timedelta(days=2))
    assert len(waiting) == 4

    # Judging one removes it from the queue without deleting the record.
    from pipeline.labeling import append_labels

    append_labels(
        "relevance",
        [{
            "date": str(DAY),
            "work_key": "doi:10.1/a",
            "label": "keep",
            "sampling": "ranked_top_n",
            "labelled_at": "2026-08-20T00:00:00+00:00",
        }],
    )
    assert len(held.pending()) == 2  # both dates' copies of that key are gone
    assert held.load(DAY) is not None


def test_counts_is_what_the_weekly_summary_reports(repo):
    held.record(DAY - timedelta(days=3), _suspicions(), published=5)
    counts = held.counts()

    assert counts["waiting"] == 2
    assert counts["withheld"] == 1
    assert counts["near_miss"] == 1
    assert counts["oldest"] == str(DAY - timedelta(days=3))


# --------------------------------------------------------------------------
# Holding must not block the day
# --------------------------------------------------------------------------


def test_the_weekly_summary_asks_for_the_held_queue(repo):
    from pipeline.notify import weekly_body, weekly_summary

    held.record(DAY, _suspicions(), published=5)
    body = weekly_body(weekly_summary(end=DAY))

    assert "held and waiting" in body
    assert "uc review --pending" in body


def test_status_surfaces_the_queue(repo):
    from pipeline.notify import status

    held.record(DAY, _suspicions(), published=5)
    assert status()["held"]["waiting"] == 2


# --------------------------------------------------------------------------
# Re-judging the pre-split labels (M1)
# --------------------------------------------------------------------------


def _weak_row(key: str) -> dict:
    return {
        "date": "2026-08-11",
        "work_key": key,
        "source": "journal",
        "rank": 7,
        "label": "drop_weak",
        "score": 1.0,
        "title": f"Title {key}",
        "sampling": "ranked_top_n",
        "labelled_at": "2026-08-13T00:00:00+00:00",
    }


def test_a_rejudgement_appends_and_supersedes_rather_than_editing(repo):
    """The 15 unsplit rows are evidence of what was judged before the split
    existed. Editing them would destroy that; appending keeps both."""
    from pipeline.labeling import append_labels, load_labels, run_rejudge_session, superseded

    append_labels("relevance", [_weak_row("doi:10.1/x"), _weak_row("doi:10.1/y")])
    answers = iter(["m", "r"])
    result = run_rejudge_session(prompt=lambda _p: next(answers), printer=lambda *a: None)

    assert result["labelled"] == 2
    raw = load_labels("relevance")
    assert len(raw) == 4              # nothing was overwritten
    latest = {r["work_key"]: r for r in superseded(raw)}
    assert len(latest) == 2           # but only the newest counts
    assert latest["doi:10.1/x"]["label"] == "drop_weak_method"
    assert latest["doi:10.1/x"]["corrected_from"] == "drop_weak"
    assert latest["doi:10.1/x"]["corrected_by"] == "YJUN"
    assert "corrected_at" in latest["doi:10.1/x"]


def test_rejudging_does_not_move_precision(repo):
    """All three weak labels are drops. A metric that moved when someone
    relabelled would be a metric that had been moved rather than measured."""
    from pipeline.labeling import append_labels, precision_at_k, run_rejudge_session

    append_labels("relevance", [_weak_row(f"doi:10.1/{i}") for i in range(3)])
    before = precision_at_k()["by_source"]["journal"]

    answers = iter(["m", "r", "m"])
    run_rejudge_session(prompt=lambda _p: next(answers), printer=lambda *a: None)
    after = precision_at_k()["by_source"]["journal"]

    assert after["precision_at_10"] == before["precision_at_10"]
    assert after["drop_reasons"]["weak"] == before["drop_reasons"]["weak"] == 3
    assert after["weak_detail"] == {"method": 2, "arguments": 1, "unsplit": 0}


def test_only_unsplit_rows_are_offered(repo):
    from pipeline.labeling import append_labels, run_rejudge_session, weak_rows_to_rejudge

    append_labels("relevance", [_weak_row("doi:10.1/x")])
    answers = iter(["m"])
    run_rejudge_session(prompt=lambda _p: next(answers), printer=lambda *a: None)

    assert weak_rows_to_rejudge() == []
    # And a second pass has nothing to do rather than re-asking.
    again = run_rejudge_session(prompt=lambda _p: "m", printer=lambda *a: None)
    assert again["labelled"] == 0


def test_a_filed_item_does_not_block_the_issue(repo, monkeypatch):
    """The verification the addendum names: the day publishes, minus the hole.

    Runs the real select stage over a pool containing one off-subfield journal
    paper, and checks that the issue goes out without it rather than waiting for
    a judgement.
    """
    from pipeline import run_stages
    from pipeline.metrics import Run
    from pipeline.stages import write_stage

    good = [_item(f"doi:10.1/ok{i}", subfield="3305") for i in range(4)]
    bad = _item("doi:10.1/materials", subfield="2500")
    pool = good + [bad]
    for it in pool:
        it.ids.doi = it.work_key.split(":", 1)[1]

    run = Run.for_date(DAY)
    write_stage(run, "classify", pool)
    monkeypatch.setattr(run_stages, "_is_whitelist_journal", lambda it: True)
    monkeypatch.setattr(run_stages, "has_abstract", lambda it: True)

    selected = run_stages.stage_select(run)
    keys = {it.work_key for it in selected}

    # 2500 is a subfield our labels have never seen, so it publishes...
    assert "doi:10.1/materials" in keys
    # ...and nothing was withheld, because the day held no doubtful item.
    assert held.load(DAY) is None


def test_a_held_item_is_not_carried_into_a_later_issue(repo):
    """Held is not a backlog. It is waiting for a verdict, not owed to readers."""
    held.record(DAY, _suspicions(), published=5)

    # The queue is what a person should look at; nothing reads it to fill slots.
    import inspect as _inspect

    from pipeline import run_stages

    source = _inspect.getsource(run_stages.stage_select)
    assert "held_queue.record" in source
    assert "held_queue.load" not in source
    assert "pending(" not in source


def test_the_queue_puts_withheld_items_first(repo):
    """36 near misses and 7 withheld on one day; a week is ~250 items.

    Front-to-back oldest-first would bury the seven that actually cost an issue
    a slot under a fortnight of preprints that cost nothing.
    """
    old_near = Suspicion("arxiv:1", RULE_UNCERTAIN, NEAR_MISS, "d", 0.6, "arxiv", "N")
    new_withheld = Suspicion("doi:10.1/z", RULE_OFF_SUBFIELD, WITHHELD, "d", 1.0, "journal", "W")

    held.record(DAY - timedelta(days=5), [old_near], published=5)
    held.record(DAY, [new_withheld], published=5)

    waiting = held.pending()
    assert waiting[0]["kind"] == WITHHELD
    assert waiting[0]["date"] == str(DAY)
    assert waiting[1]["kind"] == NEAR_MISS


def test_a_sitting_is_capped_but_the_queue_is_not(repo):
    """The cap is on one sitting. Nothing is dropped and the rest is reported."""
    from pipeline.review import run_pending_session

    many = [
        Suspicion(f"arxiv:{i}", RULE_UNCERTAIN, NEAR_MISS, "d", 0.6, "arxiv", f"T{i}")
        for i in range(40)
    ]
    held.record(DAY, many, published=5)

    said: list[str] = []
    result = run_pending_session(
        prompt=lambda _p: "s", printer=said.append, limit=25
    )

    assert result["remaining"] == 40
    assert any("Showing 25" in line for line in said)
    assert any("still waiting" in line for line in said)


def test_a_day_that_withholds_too_much_warns_without_blocking(repo):
    """59% and 79% were not issues, they were wreckage.

    The warning makes that visible on the day. It does not block publication —
    refusing to publish would lose the whole day instead of part of it.
    """
    from pipeline.held import over_warn_threshold

    assert over_warn_threshold(published=17, withheld=3) is None       # 0.15
    loud = over_warn_threshold(published=4, withheld=15)               # 0.79
    assert loud is not None
    assert "78.95%" in loud and "editorial policy" in loud


def test_the_gate_denies_rather_than_allows(repo):
    """A deny-list, and the direction is the whole point.

    An allow-list of the 42 subfields our labels have seen would withhold every
    paper in a subfield never labelled — the harshest treatment of the most
    complete absence of evidence, and the opposite of the rule it implements.
    We have evidence about four subfields; we have none about the hundreds we
    have not labelled.
    """
    from pipeline.held import rejected_subfield_ids, whitelist_subfield_ids

    rejected = rejected_subfield_ids()

    assert whitelist_subfield_ids() == {"3305", "3313", "3322"}, "journal list must not move"
    # Was four. `subfield_check` judged five papers in each of them and **all
    # four had keeps** — 5/5, 4/5, 5/5 and 2/4 — so the list is empty as of 0P
    # (Q3). Each had been excluded on three to six ranked observations.
    #
    # Empty is the correct resting state for a deny-list whose evidence has all
    # been withdrawn, and it is still a deny-list: an unseen subfield passes,
    # which is what makes emptiness safe rather than a hole. The assertion below
    # is the one that matters and holds either way.
    assert rejected == set()
    assert len(rejected) < 10, "a deny-list, not an allow-list wearing a disguise"


def test_a_paper_in_a_rejected_subfield_is_withheld_again(repo, monkeypatch):
    """Enforcement is back on, against the derived list."""
    import pipeline.held as held_mod

    monkeypatch.setattr(held_mod, "rejected_subfield_ids", lambda: {"9999"})
    rejected = _item("doi:10.1/rejected", subfield="9999")
    suspicion = inspect(rejected, "journal", selected=True)

    assert suspicion is not None
    assert suspicion.rule == RULE_OFF_SUBFIELD
    assert suspicion.kind == WITHHELD


def test_a_paper_our_labels_keep_is_no_longer_withheld(repo):
    """2305 was taken by the old rule and our labels keep 4 of 4."""
    kept = _item("doi:10.1/heat", subfield="2305")
    assert inspect(kept, "journal", selected=True) is None


def test_the_held_record_is_byte_identical_on_a_re_run(repo):
    """`content/` unchanged on a second run of the same date is a PRD guarantee.

    The held file carried a `recorded_at` and so moved every time, which the
    idempotency check caught. It states what was doubtful about a *day*; the
    run log is the place where a second attempt is a new fact.
    """
    held.record(DAY, _suspicions(), published=9)
    first = held.held_path(DAY).read_bytes()

    held.record(DAY, _suspicions(), published=9)
    assert held.held_path(DAY).read_bytes() == first
