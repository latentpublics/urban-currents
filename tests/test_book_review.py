"""Book-review detection and demotion (phase 0i, V1-2).

Three of the nine labelled `drop_not_our_kind` journal items are book reviews.
They are the one part of "not our kind" a machine can see, and the tests here
pin both halves of that claim: what counts as a review, and what must not.
"""

from __future__ import annotations

from pipeline.filters.book_review import demotion, is_book_review, signals

# Verbatim from `runs/labels/relevance.jsonl` — the three YJUN dropped.
LABELLED_REVIEWS = [
    "<i>Printing Nueva York: Spanish-language print culture, media change, and "
    "democracy in the late nineteenth century</i> , by Kelley Kreitz",
    "<i>Unruly comparison: Queerness, Hong Kong, and the Sinophone</i> , by Alvin Wong",
    "<i>Exiles in New York City: Warehousing the marginalized on Ward’s Island</i> , "
    "by Philip T. Yanos",
]

# Verbatim from the same corpus, and every one of them is our kind.
NOT_REVIEWS = [
    "A Systematic Literature Review of Urban Noise Modeling",
    "A scientometric review of unmanned aerial vehicles (UAVs) and transportation science",
    "Beyond carbon: a systematic review of building decarbonization co-benefits",
    "Mapping thematic evolution in urban design for older people: A hybrid scientometric review",
    "Multilevel structural equation modelling of walkability in a mid-sized city",
    "Temporally Weighted Land Surface Temperature",
]


def test_the_three_labelled_book_reviews_are_detected():
    for title in LABELLED_REVIEWS:
        assert is_book_review(title), title


def test_a_systematic_review_is_not_a_book_review():
    """The false positive that nearly shipped.

    Keying on OpenAlex's `review` type caught 52 works, nearly all of them
    systematic literature reviews — a genre this digest covers. Demoting them
    would have removed survey papers from the journal path with no label
    supporting it.
    """
    for title in NOT_REVIEWS:
        assert not is_book_review(title, openalex_type="review"), title


def test_openalex_book_review_type_is_enough_on_its_own():
    assert is_book_review("Spatial weapons for the working class", openalex_type="book-review")


def test_italics_alone_are_not_enough():
    # Species names, ship names and cited work titles all arrive in italics.
    assert not is_book_review("Modelling <i>Aedes aegypti</i> spread in dense settlements")
    assert signals("Modelling <i>Aedes aegypti</i> spread")["markup"] is True


def test_a_lowercase_by_phrase_is_not_an_attribution():
    assert not is_book_review("Urban heat, by the numbers")


def test_the_editors_form_is_caught():
    assert is_book_review(
        "<i>Research handbook on urban sociology</i> , edited by Miguel A. Martínez",
        openalex_type="book-review",
    )


def test_review_forum_and_review_prefix_are_caught():
    assert is_book_review(
        "Book review forum: <i>Artificial Intelligence and the City</i> Cugurullo"
    )
    assert is_book_review(
        "Review: Decolonizing Planning, by Bjørn Sletto, Tanja Winkler, and Efadul Huq"
    )


def test_demotion_is_a_multiplier_not_a_filter():
    """Demoted, not dropped — a thin journal day still publishes what was printed."""
    assert demotion("An ordinary paper about cities") == 1.0
    assert demotion(LABELLED_REVIEWS[0]) == 0.0


def test_journal_ranking_puts_a_review_below_an_ordinary_paper(repo):
    from tests.test_selection_paths import _whitelist_source_id, journal_item

    from pipeline.run_stages import journal_rank_score

    wl = _whitelist_source_id()
    ordinary = journal_item(1, wl)
    review = journal_item(2, wl)
    review.bibliography.title = LABELLED_REVIEWS[1]
    for it in (ordinary, review):
        it.scores.components.artifact_completeness = 1.0
        it.scores.components.novelty = 1.0
        it.scores.components.source_multiplicity = 1.0

    assert journal_rank_score(review) < journal_rank_score(ordinary)
