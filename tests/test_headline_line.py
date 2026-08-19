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

    assert basis == "llm:retry"
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

    assert basis == "llm"
    assert line == "A rail transit knowledge graph"


def test_the_headline_task_has_its_own_prompt_version(repo):
    """Not part of `summarize`: bumping that prompt would re-summarise 2,224
    items. This is one call per issue."""
    from pipeline.llm import LLMClient

    head = LLMClient(task="headline")
    summ = LLMClient(task="summarize")

    assert head.prompt_version != summ.prompt_version
    assert head.prompt_version.startswith("headline@")
