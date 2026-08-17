"""Signals, headline scoring, quiet-day logic, and the preview render contract."""

from __future__ import annotations

from datetime import date, timedelta
from html.parser import HTMLParser

from pipeline.calibrate import _quantile, calibrate_threshold, daily_distribution
from pipeline.models import (
    Bibliography,
    Cluster,
    EntityRef,
    Headline,
    Issue,
    Item,
    ScanMeta,
    StatusChange,
    SummaryEn,
)
from pipeline.render.preview import build_card, render_issue
from pipeline.score.headline import (
    artifact_completeness,
    headline_line,
    novelty,
    pick_headline,
    score_item,
    source_multiplicity,
)
from pipeline.signals import apply_badges, apply_rule_signals


def _item(work_key="arxiv:2608.01234", abstract="", title="A Paper") -> Item:
    return Item(
        work_key=work_key,
        first_published=date(2026, 8, 11),
        bibliography=Bibliography(title=title, abstract=abstract),
    )


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def test_code_url_in_abstract_sets_a_high_confidence_signal(repo):
    item = _item(abstract="We release code at https://github.com/example/streetcount.")
    apply_rule_signals(item)
    apply_badges(item)
    assert item.signals.code_available.value is True
    assert item.signals.code_available.url == "https://github.com/example/streetcount"
    assert item.signals.code_available.confidence == "high"
    assert "code" in item.badges


def test_explicit_non_release_beats_a_phrase_match(repo):
    item = _item(abstract="Neither the data nor the code is publicly released.")
    apply_rule_signals(item)
    apply_badges(item)
    assert item.signals.code_available.value is False
    assert item.signals.data_available.value is False
    assert "code" not in item.badges
    assert "data" not in item.badges


def test_sample_size_and_temporal_coverage_are_detected_with_detail(repo):
    item = _item(abstract="We use 3.4M images across 12 cities from 2019-2023.")
    apply_rule_signals(item)
    assert item.signals.sample_size_reported.value is True
    assert "images" in item.signals.sample_size_reported.detail
    assert item.signals.temporal_coverage_reported.value is True
    assert "2019-2023" in item.signals.temporal_coverage_reported.detail


def test_absent_numbers_are_reported_as_absent(repo):
    item = _item(abstract="We discuss the concept of urban form qualitatively.")
    apply_rule_signals(item)
    assert item.signals.sample_size_reported.value is False
    assert item.signals.temporal_coverage_reported.value is False


def test_preprint_and_published_badges_are_mutually_exclusive(repo):
    item = _item()
    apply_rule_signals(item)
    apply_badges(item)
    assert "preprint" in item.badges and "published" not in item.badges
    item.publication_status.state = "published"
    apply_badges(item)
    assert "published" in item.badges and "preprint" not in item.badges


# --------------------------------------------------------------------------
# Headline score
# --------------------------------------------------------------------------


def test_source_multiplicity_saturates(repo):
    a = _item()
    assert source_multiplicity(a) == 0.0
    a.cluster = Cluster(members=["a", "b"])
    assert 0 < source_multiplicity(a) <= 1.0
    a.cluster = Cluster(members=["a", "b", "c", "d"])
    assert source_multiplicity(a) == 1.0


def test_artifact_completeness_caps_at_one(repo):
    item = _item()
    item.badges = ["code", "data", "published"]
    assert artifact_completeness(item) == 1.0


def test_novelty_is_the_share_of_unseen_overlay_tags(repo):
    item = _item()
    item.entities.methods = [EntityRef(id="method:gnn", label="gnn")]
    item.entities.data = [EntityRef(id="data:street-view", label="sv")]
    assert novelty(item, seen_entity_ids=set()) == 1.0
    assert novelty(item, seen_entity_ids={"method:gnn"}) == 0.5
    assert novelty(item, seen_entity_ids={"method:gnn", "data:street-view"}) == 0.0
    # No overlay tags at all means no novelty claim, not a free 1.0.
    assert novelty(_item(), seen_entity_ids=set()) == 0.0


def test_headline_score_is_the_weighted_sum(repo):
    item = _item()
    item.scores.relevance = 1.0
    item.badges = ["code", "data", "published"]
    item.cluster = Cluster(members=["a", "b", "c"])
    item.entities.methods = [EntityRef(id="method:gnn", label="gnn")]
    score_item(item, seen_entity_ids=set())
    # 0.4*1 + 0.2*1 + 0.2*1 + 0.2*1
    assert item.scores.headline == 1.0


def test_quiet_day_when_nothing_clears_the_threshold(repo):
    low = _item("arxiv:2608.00001")
    low.scores.headline = 0.1
    assert pick_headline([low], threshold=0.5) is None
    high = _item("arxiv:2608.00002")
    high.scores.headline = 0.9
    assert pick_headline([low, high], threshold=0.5) is high


def test_headline_line_comes_from_the_summary_not_from_thin_air(repo):
    item = _item(title="Fallback Title")
    assert headline_line(item) == "Fallback Title"
    item.summary.en = SummaryEn(what="First sentence here. Second one.", why="Because.")
    assert headline_line(item) == "First sentence here."


# --------------------------------------------------------------------------
# Calibration maths
# --------------------------------------------------------------------------


def test_quantile_interpolates():
    assert _quantile([0.0, 1.0], 0.5) == 0.5
    assert _quantile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert _quantile([1.0, 2.0, 3.0], 1.0) == 3.0


def test_daily_distribution_counts_days_with_zero(repo):
    rows = [
        {"date": "2026-08-01", "relevance": 0.9},
        {"date": "2026-08-01", "relevance": 0.9},
        {"date": "2026-08-02", "relevance": 0.1},
    ]
    dist = daily_distribution(rows, threshold=0.5)
    assert dist["days_observed"] == 2
    assert dist["per_day"] == {"2026-08-01": 2, "2026-08-02": 0}
    assert dist["total_selected"] == 2


def test_calibrate_reports_no_data_rather_than_a_number(repo):
    assert calibrate_threshold()["status"] == "NO_DATA"


def test_calibration_hits_the_target_headline_rate(repo):
    """The threshold is chosen from the distribution of *daily top* scores."""
    from pipeline.calibrate import backfill_dir, _scores_path
    import json

    backfill_dir().mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(100):
        rows.append({
            "work_key": f"arxiv:2608.{i:05d}",
            "date": f"2026-05-{(i % 28) + 1:02d}",
            "relevance": 0.9,
            "headline": i / 100,
            "components": {},
            "categories": [],
            "source": "arxiv",
            "selected": True,
        })
    with _scores_path().open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    result = calibrate_threshold(0.30, 0.50)
    assert result["status"] == "OK"
    assert 0.30 <= result["headline_rate"] <= 0.50
    assert result["in_band"] is True


# --------------------------------------------------------------------------
# Preview render
# --------------------------------------------------------------------------


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cards = 0
        self.classes: list[str] = []
        self.external: list[str] = []
        self.entity_ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        cls = d.get("class") or ""
        if cls:
            self.classes.append(cls)
        if tag == "article" and "uc-card" in cls:
            self.cards += 1
        if "data-entity-id" in d:
            self.entity_ids.append(d["data-entity-id"])
        for attr in ("src", "href"):
            v = d.get(attr) or ""
            if tag in ("img", "script", "iframe", "link") and v.startswith(
                ("http", "//")
            ):
                self.external.append(v)


def _issue_with(items: list[Item], quiet: bool = False) -> Issue:
    return Issue(
        date=date(2026, 8, 11),
        headline=Headline(
            present=not quiet,
            work_key=items[0].work_key if items and not quiet else None,
            line="A twelve-city model puts a number on pedestrian volume." if not quiet else None,
        ),
        quiet_day=quiet,
        scan_meta=ScanMeta(candidates_scanned=311, candidates_after_gate=167,
                           items_published=len(items), arxiv_categories=7, journals=96),
        items=[i.work_key for i in items],
        run_id="run_2026-08-11",
    )


def test_card_carries_the_two_layers_and_canonical_tag_ids(repo):
    item = _item(title="Street-View Imagery", abstract="x")
    item.summary.en = SummaryEn(what="What happened.", why="Why it matters.")
    item.entities.methods = [EntityRef(id="method:gnn", label="graph neural network")]
    item.badges = ["code", "preprint"]

    card = build_card(item)
    assert card["what"] == "What happened."
    assert card["why"] == "Why it matters."
    assert card["facets"][0]["tags"][0]["id"] == "method:gnn"

    html = render_issue(_issue_with([item]), [item])
    p = _Collector()
    p.feed(html)
    assert p.cards == 1
    assert p.entity_ids == ["method:gnn"]
    assert p.external == []


def test_quiet_day_renders_the_quiet_line_and_still_shows_cards(repo):
    """A quiet day is not an empty day (PRD §5.6)."""
    item = _item()
    item.summary.en = SummaryEn(what="What.", why="Why.")
    html = render_issue(_issue_with([item], quiet=True), [item])
    assert "a quiet day in urban data science" in html
    assert "uc-headline--quiet" in html
    p = _Collector()
    p.feed(html)
    assert p.cards == 1


def test_item_without_a_summary_still_renders(repo):
    """A summarize failure must not produce a broken card."""
    item = _item()
    html = render_issue(_issue_with([item]), [item])
    assert "Summary pending review." in html


def test_status_changes_are_rendered(repo):
    item = _item()
    issue = _issue_with([item])
    issue.status_changes = [
        StatusChange(work_key="arxiv:2604.09876", **{"from": "preprint"},
                     to="published", journal="Cities")
    ]
    html = render_issue(issue, [item])
    assert "uc-status-changes" in html
    assert "preprint" in html and "Cities" in html


def test_class_names_are_the_phase1_contract(repo):
    """Phase 1's Astro components inherit these names; renaming is cross-phase."""
    item = _item()
    item.summary.en = SummaryEn(what="What.", why="Why.")
    item.entities.methods = [EntityRef(id="method:gnn", label="gnn")]
    item.badges = ["code", "preprint"]
    html = render_issue(_issue_with([item]), [item])
    for expected in (
        "uc-issue", "uc-issue__masthead", "uc-scanmeta", "uc-headline",
        "uc-headline__line", "uc-cards", "uc-card", "uc-card__title",
        "uc-card__byline", "uc-card__what", "uc-card__why", "uc-badges",
        "uc-badge", "uc-facets", "uc-facet__tags", "uc-tag", "uc-card__links",
    ):
        assert expected in html, f"missing class {expected}"


# --------------------------------------------------------------------------
# Backfill aging — the archive the novelty term is measured against
# --------------------------------------------------------------------------


def _tagged(key: str, day: int, tags: list[str], relevance: float = 0.9) -> Item:
    it = Item(
        work_key=key,
        first_published=date(2026, 5, day),
        bibliography=Bibliography(title="A Paper"),
    )
    it.scores.relevance = relevance
    it.entities.methods = [EntityRef(id=t, label=t.split(":", 1)[1]) for t in tags]
    return it


def test_backfill_novelty_ages_day_by_day(repo):
    """A tag published in May is not still novel in August.

    Scoring the whole range against one frozen archive was pinning novelty near
    1.0 for a few items every day, which put 78 of 90 daily top scores on the
    same value and left no threshold able to reach the 30-50% band.
    """
    from pipeline.calibrate import score_days

    first = _tagged("arxiv:2605.00001", 1, ["method:clustering"])
    later = _tagged("arxiv:2605.00002", 2, ["method:clustering"])

    # Deliberately out of order: the walk sorts by date, it does not trust input.
    rows = {r["work_key"]: r for r in score_days([later, first], set(), 0.35)}

    assert rows["arxiv:2605.00001"]["components"]["novelty"] == 1.0
    assert rows["arxiv:2605.00002"]["components"]["novelty"] == 0.0


def test_only_published_items_age_the_archive(repo):
    """24 of ~190 candidates a day reach content/. A tag on an item that never
    published was never seen, so it is still novel when it reappears."""
    from pipeline.calibrate import score_days

    # 30 arXiv candidates and no journal articles. The arXiv path publishes at
    # most 12 (V1-1), so t12..t29 miss out — and the point of the test is that
    # missing out is what keeps a tag novel.
    day_one = [
        _tagged(f"arxiv:2605.1{i:04d}", 1, [f"method:t{i}"], relevance=0.9 - i / 1000)
        for i in range(30)
    ]
    published_tag = _tagged("arxiv:2605.20000", 2, ["method:t0"])
    dropped_tag = _tagged("arxiv:2605.20001", 2, ["method:t29"])

    rows = {
        r["work_key"]: r
        for r in score_days(day_one + [published_tag, dropped_tag], set(), 0.35)
    }

    assert sum(1 for r in rows.values() if r["date"] == "2026-05-01" and r["published"]) == 12
    assert rows["arxiv:2605.20000"]["components"]["novelty"] == 0.0
    assert rows["arxiv:2605.20001"]["components"]["novelty"] == 1.0


def test_an_undated_item_neither_publishes_nor_ages_the_archive(repo):
    from pipeline.calibrate import score_days

    undated = Item(
        work_key="arxiv:2605.30000",
        bibliography=Bibliography(title="No date"),
    )
    undated.scores.relevance = 0.9
    undated.entities.methods = [EntityRef(id="method:clustering", label="clustering")]

    rows = {r["work_key"]: r for r in score_days([undated], set(), 0.35)}
    assert rows["arxiv:2605.30000"]["published"] is False
    assert rows["arxiv:2605.30000"]["date"] == ""


def test_the_threshold_is_calibrated_on_what_would_publish(repo):
    """The day's headline is the top card of its issue, not the top of a pool
    the reader never sees."""
    import json

    from pipeline.calibrate import _scores_path, backfill_dir

    backfill_dir().mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(100):
        day = f"2026-05-{(i % 28) + 1:02d}"
        rows.append({
            "work_key": f"arxiv:2608.{i:05d}", "date": day, "relevance": 0.9,
            "headline": i / 100, "components": {}, "categories": [],
            "source": "arxiv", "selected": True, "published": True,
        })
        # A higher-scoring candidate on every day that never reaches the issue.
        rows.append({
            "work_key": f"arxiv:2609.{i:05d}", "date": day, "relevance": 0.9,
            "headline": 0.99, "components": {}, "categories": [],
            "source": "arxiv", "selected": True, "published": False,
        })
    with _scores_path().open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    result = calibrate_threshold(0.30, 0.50)
    assert result["population"] == "published"
    assert result["n_selected"] == 100
    # Had the unpublished 0.99s counted, every daily top would be 0.99 and no
    # threshold could split the days at all.
    assert result["headline_threshold"] < 0.99
    assert result["in_band"] is True


def test_a_tied_daily_top_does_not_defeat_the_calibration(repo):
    """57 of 90 days share one top score, because every day publishes a
    whitelist journal article and they all score exactly 0.44. A quantile lands
    inside that tie and `>=` then admits all of them — a 100% headline rate when
    one increment higher gives 37%. The rate is enumerated, not estimated."""
    import json

    from pipeline.calibrate import _scores_path, backfill_dir

    backfill_dir().mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(90):
        day = date(2026, 5, 14) + timedelta(days=i)
        # 57 days peak at the tie value; 33 rise above it.
        top = 0.44 if i < 57 else 0.44 + (i - 56) / 1000
        rows.append({
            "work_key": f"arxiv:2608.{i:05d}", "date": str(day), "relevance": 0.9,
            "headline": top, "components": {"novelty": 0.0}, "categories": [],
            "source": "journal", "selected": True, "published": True,
        })
    with _scores_path().open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    result = calibrate_threshold(0.30, 0.50)
    assert result["quantile_method"]["rate"] == 1.0, "the tie must still be visible"
    assert result["in_band"] is True
    assert result["headline_threshold"] > 0.44
    assert 0.30 <= result["headline_rate"] <= 0.50


def test_calibration_records_that_novelty_dies(repo):
    """A closed overlay vocabulary saturates. The report should be able to say
    so with a number rather than leaving it in a comment."""
    import json

    from pipeline.calibrate import _scores_path, backfill_dir

    backfill_dir().mkdir(parents=True, exist_ok=True)
    with _scores_path().open("w", encoding="utf-8") as fh:
        for i in range(30):
            row = {
                "work_key": f"arxiv:2608.{i:05d}",
                "date": f"2026-0{5 if i < 15 else 8}-{(i % 15) + 1:02d}",
                "relevance": 0.9, "headline": 0.44 + i / 1000,
                "components": {"novelty": 1.0 if i < 15 else 0.0},
                "categories": [], "source": "journal",
                "selected": True, "published": True,
            }
            fh.write(json.dumps(row) + "\n")

    decay = calibrate_threshold(0.30, 0.50)["novelty_decay"]
    assert decay["2026-05"]["mean"] == 1.0
    assert decay["2026-08"]["mean"] == 0.0


def test_a_threshold_on_a_degenerate_score_is_marked_provisional(repo):
    """Landing in the band is not the same as the threshold meaning something.
    Every weighted component is one value on the journal path, so the number is
    reported with the reason it cannot be trusted rather than presented clean."""
    import json

    from pipeline.calibrate import _scores_path, backfill_dir

    backfill_dir().mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(90):
        day = date(2026, 5, 14) + timedelta(days=i)
        rows.append({
            "work_key": f"doi:10.0000/{i:05d}", "date": str(day), "relevance": 1.0,
            "headline": 0.44 if i < 57 else 0.44 + (i - 56) / 1000,
            "components": {"relevance": 1.0, "novelty": 0.0,
                           "artifact_completeness": 0.2, "source_multiplicity": 0.0},
            "categories": [], "source": "journal", "selected": True, "published": True,
        })
    with _scores_path().open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    result = calibrate_threshold(0.30, 0.50)
    assert result["in_band"] is True, "the band is reachable — that is the trap"
    assert result["provisional"] is True
    assert any("journal path" in r for r in result["reasons"])

    audit = result["component_audit"]
    assert audit["relevance"]["by_source"]["journal"]["distinct_values"] == 1
    assert audit["relevance"]["by_source"]["journal"]["modal_share"] == 1.0


def test_a_healthy_distribution_is_not_marked_provisional(repo):
    """The flag has to be able to clear itself, or it is decoration."""
    import json

    from pipeline.calibrate import _scores_path, backfill_dir

    backfill_dir().mkdir(parents=True, exist_ok=True)
    with _scores_path().open("w", encoding="utf-8") as fh:
        for i in range(90):
            day = date(2026, 5, 14) + timedelta(days=i)
            fh.write(json.dumps({
                "work_key": f"arxiv:2608.{i:05d}", "date": str(day),
                "relevance": 0.5 + i / 400, "headline": 0.2 + i / 200,
                "components": {"relevance": 0.5 + i / 400, "novelty": i / 90,
                               "artifact_completeness": (i % 5) / 5,
                               "source_multiplicity": (i % 3) / 3},
                "categories": [], "source": "arxiv", "selected": True,
                "published": True,
            }) + "\n")

    result = calibrate_threshold(0.30, 0.50)
    assert result["provisional"] is False
    assert result["reasons"] == []


def test_the_daily_volume_is_reported_per_entry_path(repo):
    """A whitelist article clears by membership, so pooling the paths turns
    "is there enough signal" into "how many whitelist articles appeared". The
    pooled median moved 28 -> 72 when journals joined the backfill."""
    from pipeline.calibrate import daily_distribution

    rows = []
    for i in range(10):
        day = f"2026-05-{i + 1:02d}"
        rows += [{"date": day, "relevance": 1.0, "source": "journal"}] * 20
        rows += [{"date": day, "relevance": 0.9, "source": "arxiv"}] * 6

    dd = daily_distribution(rows, 0.35)
    assert dd["median_per_day"] == 26
    assert dd["by_source"]["journal"]["median_per_day"] == 20
    assert dd["by_source"]["arxiv"]["median_per_day"] == 6


def test_arxiv_candidate_volume_is_reported_at_every_plausible_floor(repo):
    """Precision says how good the items above a floor are; this says whether
    there are enough of them. A floor yielding 3 a day for 12 slots is not a
    precision decision, it is a decision to stop filling the path."""
    import json

    from pipeline.calibrate import _scores_path, arxiv_candidates_by_floor, backfill_dir

    backfill_dir().mkdir(parents=True, exist_ok=True)
    with _scores_path().open("w", encoding="utf-8") as fh:
        for day in range(10):
            # 20 candidates a day: relevance 0.30, 0.33, … 0.87.
            for i in range(20):
                fh.write(json.dumps({
                    "work_key": f"arxiv:26{day:02d}.{i:05d}",
                    "date": f"2026-05-{day + 1:02d}",
                    "relevance": 0.30 + i * 0.03, "headline": 0.4,
                    "components": {}, "categories": [], "source": "arxiv",
                    "selected": True, "published": i < 12,
                }) + "\n")

    result = arxiv_candidates_by_floor()
    assert result["status"] == "OK"
    assert result["days"] == 10
    by_floor = {r["floor"]: r for r in result["floors"]}

    # 18 of 20 clear 0.35; only 2 clear 0.85, so the high floor starves the path.
    assert by_floor[0.35]["median_per_day"] == 18
    assert by_floor[0.35]["days_short_of_slots"] == 0
    assert by_floor[0.90]["days_short_of_slots"] == 10


def test_candidate_volume_reports_absence_rather_than_zero(repo):
    from pipeline.calibrate import arxiv_candidates_by_floor

    assert arxiv_candidates_by_floor()["status"] == "NO_DATA"
