"""Mockup alignment: card order, meta rail, canon block, label boundary (0R).

T5, T6, T7 and T8 together. Each is small; what they share is that every one of
them is a place where a display choice could quietly become a claim.
"""

from __future__ import annotations

import json

import pytest

from pipeline.models import (
    Author,
    Bibliography,
    Institution,
    Issue,
    Item,
    Summary,
    SummaryEn,
)


def _item(key, state="preprint", headline=0.5, title="A paper", institution=None):
    authors = [Author(name="A. Author", institutions=(
        [Institution(name=institution)] if institution else []
    ))]
    it = Item(work_key=key, bibliography=Bibliography(title=title, authors=authors))
    it.publication_status.state = state
    it.scores.headline = headline
    it.summary = Summary(en=SummaryEn(what="It did a thing.", why="It matters."))
    return it


# --------------------------------------------------------------------------
# T5 — published articles first, preprints after
# --------------------------------------------------------------------------


def test_published_articles_sort_ahead_of_preprints(repo):
    from pipeline.render.preview import card_order

    items = [
        _item("arxiv:2608.00001", "preprint", headline=0.9),
        _item("doi:10.1/a", "published", headline=0.2),
        _item("doi:10.1/b", "published", headline=0.7),
        _item("arxiv:2608.00002", "preprint", headline=0.8),
    ]
    order = [it.work_key for it in sorted(items, key=lambda i: card_order(i, None))]

    assert order == ["doi:10.1/b", "doi:10.1/a", "arxiv:2608.00001", "arxiv:2608.00002"]


def test_the_headline_item_leads_whatever_it_is(repo):
    """A preprint that is the headline still leads. The headline is the issue's
    own choice and the group ordering does not overrule it."""
    from pipeline.render.preview import card_order

    items = [_item("doi:10.1/a", "published", headline=0.9),
             _item("arxiv:2608.00001", "preprint", headline=0.1)]
    order = [it.work_key for it in
             sorted(items, key=lambda i: card_order(i, "arxiv:2608.00001"))]

    assert order[0] == "arxiv:2608.00001"


def test_importance_still_decides_inside_each_group(repo):
    from pipeline.render.preview import card_order

    items = [_item("doi:10.1/low", "published", headline=0.1),
             _item("doi:10.1/high", "published", headline=0.9)]
    order = [it.work_key for it in sorted(items, key=lambda i: card_order(i, None))]

    assert order == ["doi:10.1/high", "doi:10.1/low"]


def test_the_order_reads_the_fact_and_not_the_badge(repo):
    """`badges` is derived for display. Sorting on it would make the order
    depend on the rendering of the thing being rendered."""
    it = _item("doi:10.1/a", "published")
    it.badges = ["preprint"]  # deliberately inconsistent

    from pipeline.render.preview import card_order

    assert card_order(it, None)[1] == 0, "publication_status decides, not badges"


# --------------------------------------------------------------------------
# T6 — the meta rail
# --------------------------------------------------------------------------


def test_the_affiliation_is_its_own_line_when_there_is_one(repo):
    from pipeline.render.preview import build_card

    card = build_card(_item("doi:10.1/a", institution="Lund University"))

    assert card["affiliation"] == "Lund University"


def test_no_affiliation_means_no_line_rather_than_an_empty_one(repo):
    """An empty line reads as a value we lost. Most arXiv items have no
    institution in OpenAlex at all, which is a fact about OpenAlex."""
    from pipeline.render.preview import build_card

    assert build_card(_item("arxiv:2608.00001"))["affiliation"] == ""


def test_keywords_are_italic_and_faint(repo):
    """Mockup 6a's style. The mockup's own #9296a0 measures 2.91:1 and fails
    AA, so the ink is --uc-faint — the substitution 0j made everywhere."""
    from pipeline.render.site import TEMPLATE_DIR

    css = (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8")
    block = css[css.index(".uc-card__facets"):css.index(".uc-card__facets") + 160]

    assert "font-style: italic" in block
    assert "--uc-faint" in block
    assert "#9296a0" not in block


# --------------------------------------------------------------------------
# T7 — Still cited
# --------------------------------------------------------------------------


def test_still_cited_is_on(repo):
    from pipeline.config import cfg

    assert cfg("render.still_cited", False) is True


def test_a_day_that_cites_no_foundational_work_renders_no_block(repo):
    """0j's rule: a section with nothing to say does not appear. It does not
    appear empty and it does not reach for something."""
    from pipeline.render.preview import build_still_cited

    issue = Issue(date="2026-08-05", run_id="r")

    assert build_still_cited(issue, []) is None


# --------------------------------------------------------------------------
# T8 — two labelling standards, never averaged
# --------------------------------------------------------------------------


def test_the_boundary_is_recorded_in_config_not_on_the_rows(repo):
    """Existing labels are not edited. A label written under the old bar is a
    true record of the old bar."""
    from pipeline.labeling import standard_boundary

    assert standard_boundary() == "2026-08-19T00:00:00+00:00"


def test_labels_are_split_by_when_the_judgement_was_made(repo):
    from pipeline.labeling import split_by_standard

    rows = [
        {"work_key": "a", "label": "keep", "labelled_at": "2026-08-17T10:00:00+00:00"},
        {"work_key": "b", "label": "keep", "labelled_at": "2026-08-20T10:00:00+00:00"},
    ]
    groups = split_by_standard(rows)

    assert [r["work_key"] for r in groups["before"]] == ["a"]
    assert [r["work_key"] for r in groups["after"]] == ["b"]


def test_a_correction_does_not_move_a_row_across_the_boundary(repo):
    """D204 renamed a category and stamped 15 rows with 2026-08-19. Keying on
    `corrected_at` put every one of them on the far side as "0 keeps under the
    new standard" — a number that would have been read as the stricter bar
    working. **A correction is not a new judgement.**"""
    from pipeline.labeling import split_by_standard

    rows = [{
        "work_key": "a",
        "label": "drop_weak_arguments",
        "labelled_at": "2026-08-17T10:00:00+00:00",
        "corrected_from": "drop_weak",
        "corrected_at": "2026-08-19T16:00:00+00:00",
    }]
    groups = split_by_standard(rows)

    assert len(groups["before"]) == 1
    assert groups["after"] == []


def test_the_two_standards_are_never_added_together(repo):
    """There is no pooled figure, on purpose. A caller who wants one across two
    bars has to say so and own what it means."""
    from pipeline.labeling import keep_rate_by_standard

    out = keep_rate_by_standard("relevance")

    assert "pooled" not in out
    assert "not comparable" in out["note"]
    assert set(out) >= {"boundary", "before", "after", "after_is_readable"}


def test_a_thin_new_standard_says_it_is_not_measurable_yet(repo):
    """Fewer than 30 labels under the new bar and the answer is "we cannot tell
    yet", not a keep rate presented as a finding."""
    from pipeline.labeling import keep_rate_by_standard

    out = keep_rate_by_standard("relevance")

    if out["after"]["n"] < 30:
        assert out["after_is_readable"] is False
        assert "not yet measurable" in out["after"]["caveat"]
