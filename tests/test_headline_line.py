"""The headline is a title now, and it still may not overstate (0R, T4).

YJUN asked for a headline that catches attention. That request sits one step
from everything this service refuses to do — quiet days are declared rather than
padded, unmeasured connections are not drawn — and the headline is the most
visible line on the page, so it is where overstatement would cost most.

**The goal is compression, not excitement.** These tests are the mechanical half
of that: the prompt asks, and `check()` guarantees.
"""

from __future__ import annotations

import pytest

from pipeline.models import Bibliography, Item, Summary, SummaryEn
from pipeline.summarize.headline import (
    MAX_WORDS,
    check,
    fallback,
    write_headline,
)


def _item(what="A dataset of 1284 rail transit stations across three US cities.",
          why="It matters for planners.",
          title="A Rail Transit Station Knowledge Graph"):
    it = Item(work_key="arxiv:2608.01234", bibliography=Bibliography(title=title))
    it.summary = Summary(en=SummaryEn(what=what, why=why))
    return it


GOOD = [
    "A rail transit knowledge graph, published as Linked Data",
    "Montreal's REM light rail and housing prices, across project phases",
    "Barge-tow configurations, recovered from AIS trajectories",
]


@pytest.mark.parametrize("line", GOOD)
def test_a_plain_title_passes(line):
    assert check(line, _item()) is None


@pytest.mark.parametrize("line,reason", [
    ("A breakthrough in urban knowledge graphs", "banned"),
    ("An unprecedented dataset of transit stations", "banned"),
    ("This transforms how cities plan transit", "banned"),
    ("A graph that finally solves transit planning", "banned"),
    ("The first dataset of its kind for transit", "novelty"),
])
def test_hype_is_refused(line, reason):
    """Superlatives, novelty claims and hype verbs, whatever the prompt did."""
    problem = check(line, _item())
    assert problem is not None
    assert reason in problem


@pytest.mark.parametrize("line", [
    "Can knowledge graphs fix transit planning?",
    "Why transit graphs matter for planners",
    "How cities measure heat exposure",
    "What if transit were a graph",
])
def test_the_question_register_is_refused(line):
    """Both with the mark and without it: dropping the `?` does not stop a
    question being a question."""
    assert check(line, _item()) is not None


def test_second_person_is_refused():
    assert check("A transit graph for your city", _item()) is not None


def test_a_number_the_material_does_not_state_is_refused():
    """The check that matters most. A reader can hold the headline against the
    summary we showed them, and a figure that is in neither is unanswerable."""
    assert check("A graph of 1284 stations", _item()) is None
    assert check("A graph of 9999 stations", _item()) is not None


def test_a_number_the_model_worked_out_is_also_refused():
    """This one is real. For a summary reading "Singapore and three US cities"
    the model wrote "across four cities" — which is **correct**, and still a
    number the source never states. A count the reader cannot check against the
    material we gave them is as unusable as an invented one."""
    it = _item(what="Data spanning Singapore and three US cities.")

    assert check("Waiting times, across three US cities", it) is None
    assert check("Waiting times, across four cities", it) is not None


def test_ordinary_words_that_look_like_claims_are_allowed():
    """`first-mile access` and `first-order effects` are this field's subject
    matter. Banning the token outright would reject good lines."""
    it = _item(what="First-mile access measured near stations.")

    assert check("First-mile access, measured near stations", it) is None


def test_length_is_rejected_and_never_trimmed():
    """Cutting a line to fit produces a truncated claim, which is a **different
    claim**. The prompt is asked to keep to the limit and the answer is refused
    if it did not."""
    long = " ".join(["word"] * (MAX_WORDS + 3))
    problem = check(long, _item())

    assert problem is not None
    assert str(MAX_WORDS) in problem


def test_terminal_punctuation_and_quotes_are_refused():
    assert check("A rail transit knowledge graph.", _item()) is not None
    assert check('"A rail transit knowledge graph"', _item()) is not None


# --------------------------------------------------------------------------
# The fallback is a real path, not an error
# --------------------------------------------------------------------------


def test_without_an_llm_the_old_behaviour_returns(repo):
    """An issue must never fail to publish because a headline could not be
    written. The pre-0R extractive sentence is a perfectly honest lead."""
    line, basis = write_headline(_item(), use_llm=False)

    assert basis == "fallback:llm_disabled"
    assert line == fallback(_item())
    assert line.startswith("A dataset of 1284")


def test_a_refused_answer_is_retried_then_falls_back(repo):
    """One retry, told what it broke — then the extractive line."""
    from pipeline.llm import LLMClient, LLMResponse

    said: list[str] = []

    def caller(system, user):
        said.append(user)
        return LLMResponse(text="A breakthrough in transit graphs")

    client = LLMClient(task="headline", caller=caller, cache_enabled=False)
    line, basis = write_headline(_item(), client=client)

    assert len(said) == 2, "asked once, then retried once"
    assert "rejected" in said[1]
    assert basis.startswith("fallback:")
    assert "banned" in basis


def test_a_good_retry_is_recorded_as_a_retry(repo):
    from pipeline.llm import LLMClient, LLMResponse

    answers = iter([
        "A breakthrough in transit graphs",
        "A rail transit knowledge graph, published as Linked Data",
    ])

    client = LLMClient(
        task="headline", caller=lambda s, u: LLMResponse(text=next(answers)),
        cache_enabled=False,
    )
    line, basis = write_headline(_item(), client=client)

    # The shape is part of the basis since 0Z-B: `lead` is the single-paper
    # form, `day` the one written over the day's papers.
    assert basis == "llm:lead:retry"
    assert line == "A rail transit knowledge graph, published as Linked Data"


def test_the_basis_travels_with_the_line(repo):
    """A title written by the model and a sentence quoted from a summary read
    differently, and which one you are looking at should not require guessing."""
    from pipeline.llm import LLMClient, LLMResponse

    client = LLMClient(
        task="headline",
        caller=lambda s, u: LLMResponse(text="A rail transit knowledge graph"),
        cache_enabled=False,
    )
    line, basis = write_headline(_item(), client=client)

    assert basis == "llm:lead"
    assert line == "A rail transit knowledge graph"


def test_the_headline_task_has_its_own_prompt_version(repo):
    """Not part of `summarize`: bumping that prompt would re-summarise 2,224
    items. This is one call per issue."""
    from pipeline.llm import LLMClient

    head = LLMClient(task="headline")
    summ = LLMClient(task="summarize")

    assert head.prompt_version != summ.prompt_version
    assert head.prompt_version.startswith("headline@")


# --------------------------------------------------------------------------
# Several papers at once (phase 0Z-B)
# --------------------------------------------------------------------------
#
# The failure that only exists at this width is not an invented number — that
# was already caught — but an invented **theme**. "A day of urban mobility
# research" sounds like a fact: three of nine papers being mobility papers is
# true, and that it amounts to a trend is something nobody measured. These
# tests are the mechanical half of refusing it.


def _day():
    """Three papers with nothing honest in common."""
    a = _item(
        title="Street-Network Entropy Across 40 Cities",
        what="We compute street-network entropy for 40 cities.",
        why="It gives planners a comparable shape measure.",
    )
    b = _item(
        title="Bus Delay from AVL Traces",
        what="We estimate bus delay from 2.1M AVL pings in Seoul.",
        why="Delay has been measured only at timing points until now.",
    )
    c = _item(
        title="Flood Exposure of Informal Settlements",
        what="We map flood exposure for informal settlements in Accra.",
        why="Exposure maps have excluded unmapped settlements.",
    )
    for i, it in enumerate((a, b, c)):
        it.work_key = f"arxiv:2608.1000{i}"
    return a, [a, b, c]


def test_a_line_naming_several_subjects_passes():
    """The point of the batch: a true line about more than one paper."""
    lead, day = _day()
    line = "Street-network entropy, bus delay and flood exposure, modelled separately"

    assert check(line, lead, day) is None


def test_an_invented_common_theme_is_refused():
    """None of the three papers says anything about a theme, a day or a field
    moving. The line reads as reportage and is not."""
    lead, day = _day()

    for line in (
        "A wave of urban mobility research",
        "Several models of city infrastructure risk",
        "Machine learning dominates urban research today",
        "Most of today's papers concern street networks",
    ):
        problem = check(line, lead, day)
        assert problem is not None, line
        assert "quantity" in problem or "number" in problem, (line, problem)


def test_a_claimed_relation_between_papers_is_refused():
    """We did not compare them, so the line cannot."""
    lead, day = _day()

    for line in (
        "Entropy and flood exposure, converging on the same street grid",
        "Bus delay and entropy, taken together as one measure",
        "Flood mapping that contradicts the entropy result",
        "Three findings that complement one another",
    ):
        problem = check(line, lead, day)
        assert problem is not None, line


def test_counting_the_papers_is_refused():
    """B4's boundary, decided: still refused.

    The count is a fact about **our selection**, not about the field — the
    model sees at most `MATERIAL_ITEMS` of a day that may hold twenty-four, so
    "three studies" invites a reader to conclude something we never counted.
    It is also a quantity word wearing digits, and those are banned above.
    """
    lead, day = _day()

    for line in ("Three studies of street networks and flooding",
                 "3 models of urban infrastructure risk"):
        problem = check(line, lead, day)
        assert problem is not None, line


def test_a_number_from_any_paper_shown_is_quotable():
    """The check runs against the **union** of the material, not the lead."""
    lead, day = _day()

    # 2.1M is in the second paper, not the first.
    assert check("Bus delay from 2.1M AVL pings, in Seoul", lead, day) is None
    # 40 is in the first.
    assert check("Street-network entropy across 40 cities", lead, day) is None
    # 99 is in none of them.
    assert check("Street-network entropy across 99 cities", lead, day) is not None


def test_a_number_in_another_paper_is_not_quotable_for_a_single_paper_line():
    """The union is the union of what was *shown*. A single-paper call shows
    one paper, so the same line is refused there."""
    lead, _ = _day()

    assert check("Bus delay from 2.1M AVL pings, in Seoul", lead) is not None


def test_the_word_limit_holds_for_the_day_form_too():
    """B3: 12 words either way. A longer allowance buys a list, and a list is
    where an invented connection hides."""
    lead, day = _day()
    long_line = " ".join(["word"] * (MAX_WORDS + 1))

    problem = check(long_line, lead, day)
    assert problem is not None and "limit" in problem


def test_the_threshold_now_picks_the_shape_not_whether_there_is_a_line():
    """B0. Every 'no headline' day in the archive had every item on exactly
    0.44 while the threshold sat at 0.444 — a 0.004 margin over a plateau
    holding a third of the archive was deciding whether the most visible line
    on the page existed."""
    from pipeline.score.headline import headline_form, pick_headline

    lead, day = _day()
    for it in day:
        it.scores.headline = 0.44

    assert pick_headline(day, threshold=0.444) is None, "nothing clears the bar"
    assert headline_form(day, threshold=0.444) == "day"

    day[1].scores.headline = 0.60
    assert headline_form(day, threshold=0.444) == "lead"


def test_the_day_form_is_recorded_in_the_basis():
    """Which shape wrote the line is not something to guess at afterwards."""
    from pipeline.llm import LLMClient, LLMResponse

    lead, day = _day()
    client = LLMClient(
        task="headline",
        caller=lambda s, u: LLMResponse(text="Street-network entropy and bus delay, modelled separately"),
        cache_enabled=False,
    )

    line, basis = write_headline(lead, client=client, others=day)

    assert basis == "llm:day"
    assert line == "Street-network entropy and bus delay, modelled separately"


def test_the_day_form_falls_back_to_one_paper_rather_than_publishing_a_theme():
    """Narrowing is not a failure. A refused line must not become a published
    one, and the fallback is a single paper's own sentence."""
    from pipeline.llm import LLMClient, LLMResponse

    lead, day = _day()
    client = LLMClient(
        task="headline",
        caller=lambda s, u: LLMResponse(text="A wave of urban mobility research"),
        cache_enabled=False,
    )

    line, basis = write_headline(lead, client=client, others=day)

    assert basis.startswith("fallback:")
    assert "quantity" in basis
    assert line == fallback(lead)


def test_the_material_is_capped():
    """A twenty-four paper day handed over whole is an invitation to
    generalise."""
    from pipeline.summarize.headline import MATERIAL_ITEMS, _material

    lead, day = _day()
    many = day + [_item(title=f"Paper {i}") for i in range(20)]
    for i, it in enumerate(many):
        it.work_key = f"arxiv:2608.2{i:04d}"
    many[0] = lead

    text = _material(lead, many)

    assert text.count("title:") == MATERIAL_ITEMS
    assert "PAPER 1:" in text
