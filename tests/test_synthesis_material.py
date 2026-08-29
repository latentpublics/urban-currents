"""The paragraph is about what arrived, not about who cited whom (0S, U2).

V3 (0i) removed the *slot* for a theme rather than asking the model not to
invent one: "we did not write in the prompt that it must not name a topic, we
removed the place in the output where a topic could go." That was right when
there was nothing to read a topic from.

There is now. Every published item carries a `what` and a `why` grounded in an
abstract, and the overlay tags are only those that matched a controlled
vocabulary. **So the group's name is not invented — it is the tag.** Naming a
tag states a fact; naming a theme proposes one, and these tests are the line
between the two.
"""

from __future__ import annotations

from datetime import date


from pipeline.models import (
    Bibliography,
    Entities,
    EntityRef,
    Item,
    Summary,
    SummaryEn,
)
from pipeline.synthesis import (
    GROUP_MIN_PAPERS,
    build_facts,
    highlights,
    render_facts,
    tag_groups,
    write_paragraph,
)


def _item(key, tags=(), what="It measured a thing.", headline=0.5, title=None):
    it = Item(
        work_key=key,
        bibliography=Bibliography(title=title or f"Paper {key}"),
    )
    it.entities = Entities(
        methods=[EntityRef(id=f"method:{t}", label=t) for t in tags]
    )
    it.summary = Summary(en=SummaryEn(what=what, why="It matters."))
    it.scores.headline = headline
    return it


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def test_a_tag_three_papers_share_is_a_group(repo):
    items = [_item(f"arxiv:2608.0000{i}", tags=["clustering"]) for i in range(3)]

    groups = tag_groups(items)

    assert len(groups) == 1
    assert groups[0]["tag"] == "clustering"
    assert groups[0]["papers"] == 3


def test_a_pair_is_not_a_group(repo):
    """Two is not a condition. Over the 63-issue archive **every single day**
    had some tag shared by two papers — with fifteen items and dozens of tags,
    a pair happens by arithmetic. A gate that never closes is not a gate."""
    items = [_item(f"arxiv:2608.0000{i}", tags=["clustering"]) for i in range(2)]

    assert tag_groups(items) == []
    assert GROUP_MIN_PAPERS == 3


def test_a_paper_carrying_a_tag_twice_counts_once(repo):
    """The question is how many papers share it, not how many times it was
    written down."""
    it = _item("arxiv:2608.00001", tags=["clustering"])
    it.entities.data = [EntityRef(id="data:clustering", label="clustering")]

    assert tag_groups([it, it, it])[0]["papers"] == 3


def test_groups_carry_their_papers(repo):
    """The paragraph is asked to name the papers in a group, so it may only be
    given papers it is allowed to name."""
    items = [
        _item("arxiv:2608.00001", tags=["clustering"], what="It clustered trips."),
        _item("arxiv:2608.00002", tags=["clustering"], what="It clustered streets."),
        _item("arxiv:2608.00003", tags=["clustering"], what="It clustered parcels."),
    ]

    group = tag_groups(items)[0]

    assert len(group["titles"]) == 3
    assert "It clustered trips." in group["whats"]


def test_groups_are_ordered_biggest_first_and_stably(repo):
    items = [_item(f"arxiv:2608.0000{i}", tags=["a"]) for i in range(4)]
    items += [_item(f"arxiv:2608.0001{i}", tags=["b"]) for i in range(3)]

    assert [g["tag"] for g in tag_groups(items)] == ["a", "b"]


# --------------------------------------------------------------------------
# Highlights
# --------------------------------------------------------------------------


def test_the_best_ungrouped_papers_are_offered(repo):
    """Without this a day's single strongest paper could go unmentioned because
    nothing else resembled it."""
    grouped = [_item(f"arxiv:2608.0000{i}", tags=["clustering"]) for i in range(3)]
    loose = _item("arxiv:2608.00099", tags=["rare"], headline=0.9, what="It stood alone.")

    items = grouped + [loose]
    hl = highlights(items, tag_groups(items))

    assert [h["work_key"] for h in hl] == ["arxiv:2608.00099"]


def test_a_paper_already_in_a_group_is_not_repeated_as_a_highlight(repo):
    items = [_item(f"arxiv:2608.0000{i}", tags=["clustering"], headline=0.9)
             for i in range(3)]

    assert highlights(items, tag_groups(items)) == []


def test_a_paper_with_no_summary_is_not_offered(repo):
    """The highlight is a compression of `what`. With no `what` there is
    nothing to compress and the model would be left to invent."""
    loose = _item("arxiv:2608.00099", tags=["rare"], what="")

    assert highlights([loose], []) == []


# --------------------------------------------------------------------------
# Silence — the condition moved, the principle did not
# --------------------------------------------------------------------------


def test_no_group_means_no_grouping_sentence(repo):
    """Rewritten in 0Z-F (S1), and the thing it guards did not move.

    It used to assert that a day with no group gets no paragraph at all, on the
    argument that the only paragraph available would be "today's papers are not
    much alike" — the filler sentence mockup 6a wrote and this project refused.
    The argument was right about that sentence and wrong that it was the only
    one available: a paragraph can name what each paper did without saying
    anything about the set. So the assertion moves from **no paragraph** to
    **no grouping claim**, which is what was ever at stake.

    `tests/test_ungrouped_paragraph.py` holds the rest of that story.
    """
    items = [_item(f"arxiv:2608.0000{i}", tags=[f"tag{i}"]) for i in range(5)]
    facts = build_facts(date(2026, 8, 5), items)
    block = render_facts(facts)

    assert facts["tag_groups"] == []
    assert "carry the tag" not in block
    assert "of today's papers" not in block
    # Still refused, loudly, if the model claims one anyway.
    from pipeline.llm import LLMClient, LLMResponse

    bad = LLMClient(
        task="synthesis",
        caller=lambda s, u: LLMResponse(text="Two of today's papers agree."),
    )
    assert write_paragraph(facts, client=bad)["omitted"] is True


def test_a_grouped_day_is_not_blocked_by_thin_citations(repo):
    """The old bar required a measured citation link. That was the right bar
    for a paragraph made of citation links, and it is not the question any
    more — a day whose papers plainly share a subject has something to say
    about what arrived even if nobody cited anybody."""
    items = [_item(f"arxiv:2608.0000{i}", tags=["clustering"]) for i in range(3)]
    facts = build_facts(date(2026, 8, 5), items)

    assert facts["anchors"] == []
    assert facts["clusters"] == []
    assert len(facts["tag_groups"]) == 1
    # It reaches the LLM step rather than being refused on material.
    assert write_paragraph(facts)["reason"] != "no measured link"


# --------------------------------------------------------------------------
# What the model is allowed to see
# --------------------------------------------------------------------------


def test_the_facts_block_is_the_day_not_the_citation_graph(repo):
    """The citation facts are not deleted and not demoted — they keep every
    label row. They stopped being **the prose**."""
    items = [_item(f"arxiv:2608.0000{i}", tags=["clustering"]) for i in range(3)]
    facts = build_facts(date(2026, 8, 5), items)
    block = render_facts(facts)

    assert 'carry the tag "clustering"' in block
    assert "shares" not in block, "no shared-reference sentences in the material"
    assert "cited today by" not in block

    # And the rows still hold them.
    assert "anchors" in facts and "clusters" in facts and "deviations" in facts


def test_the_prompt_forbids_naming_a_concept_above_the_tags(repo):
    """The single rule this whole change turns on. `graph-structured urban
    data` is not a tag, so it is not available however well it fits."""
    from pipeline.synthesis import PROMPT_PATH

    # Whitespace-normalised: the prompt is wrapped for reading, so every one
    # of these phrases spans a line break in the file.
    prompt = " ".join(PROMPT_PATH.read_text(encoding="utf-8").split())

    assert "Do not invent a concept above the tags" in prompt
    assert "Naming a tag is stating a fact" in prompt
    assert "is not a tag, so it is not available" in prompt


def test_the_prompt_keeps_every_old_prohibition(repo):
    """The material changed; the rules did not."""
    from pipeline.synthesis import PROMPT_PATH

    prompt = " ".join(PROMPT_PATH.read_text(encoding="utf-8").split()).lower()

    for rule in ("no trends", "no evaluation", "nothing about what is absent",
                 "do not restate how many papers", "use only the facts given"):
        assert rule in prompt, rule
    assert "nothing to say" in prompt, "the refusal path stays open"


def test_the_prompt_version_moved_with_the_material(repo):
    """A cached response to the old prompt answers a question no longer being
    asked. 0R shipped a retry whose key did not cover the prompt and the fix
    read back the stale answer; this is the same failure at file scale."""
    from pipeline.llm import LLMClient

    assert LLMClient(task="synthesis").prompt_version == "synthesis/daily@0.4.0"
