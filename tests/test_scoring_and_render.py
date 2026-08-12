"""Signals, headline scoring, quiet-day logic, and the preview render contract."""

from __future__ import annotations

from datetime import date
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
