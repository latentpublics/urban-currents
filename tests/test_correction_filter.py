"""A correction notice is not a paper (0P, Q3).

Two of them reached published issues, one titled — in full — `Correction`.
"""

from __future__ import annotations

import pytest

from pipeline.filters.correction import demotion, is_correction, signals

# Titles taken verbatim from the archive.
REAL_NOTICES = [
    'Corrigendum to "Understanding employees\' residential choices in connection to work"',
    "Correction",
    "Correction to ' EMBODYING AND RESISTING URBAN HEAT INJUSTICE: Migrant Vulnerability'",
    'Erratum to "Quantifying the Spatial-Governance Substitution Effect: A GIS-Integrated"',
    "Correction: Fellah, S.; Mabrouki, C. Dry Port-Seaport System: A Systematic Review",
]

# Also verbatim, and all three were caught by the first draft of this rule.
REAL_PAPERS = [
    "Learning to Distort: Weakly-Supervised Image Quality Transfer for Prostate Diffusion",
    "Efficient Visual Pointing for Embodied AI: Agent-Driven Data Synthesis, Cross-Domain Correction",
    "PARA-PV: Physics-Aware Retrieval-Augmented PV Prediction Based on Frozen Foundation Models",
    "A Systematic Literature Review of Urban Noise Modeling",
    "Error correction codes for urban sensor networks",
]


@pytest.mark.parametrize("title", REAL_NOTICES)
def test_a_notice_is_recognised(title):
    assert is_correction(title) is True
    assert demotion(title) == 0.0


@pytest.mark.parametrize("title", REAL_PAPERS)
def test_a_paper_about_correcting_something_is_not_a_notice(title):
    """The word is not the rule; its position is."""
    assert is_correction(title) is False
    assert demotion(title) == 1.0


def test_openalex_confirms_but_is_not_required():
    """Days already on disk have no `type` on Bibliography, so the title has to
    carry the rule on its own."""
    assert is_correction("Urban heat and mortality", openalex_type="erratum") is True
    assert signals("Correction to 'X'")["openalex_type"] is False
    assert signals("Correction to 'X'")["notice_phrase"] is True


def test_a_review_type_is_not_a_correction():
    """`review` means a systematic literature review here, which is a genre the
    digest covers — the same trap the book-review filter documents."""
    assert is_correction("A scientometric review of UAVs", openalex_type="review") is False


def test_it_demotes_rather_than_drops():
    """The item stays a candidate; it sinks to the bottom of the ranking."""
    assert demotion("Correction") == 0.0
    assert is_correction("Correction") is True


def test_the_two_filters_compose():
    """A book review and a corrigendum are different genres; neither rule needs
    to know about the other."""
    from pipeline.filters.book_review import demotion as book_demotion

    ordinary = "Cycling infrastructure and mode choice in Seoul"
    assert book_demotion(ordinary) * demotion(ordinary) == 1.0
    assert book_demotion(ordinary) * demotion("Corrigendum to 'X'") == 0.0
