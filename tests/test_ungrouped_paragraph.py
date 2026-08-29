"""A day with no measured group still has something to say (0Z-F, S1).

From 2026-08-21 the synthesis paragraph stopped appearing and nothing had
broken. `write_paragraph` wrote nothing whenever no controlled-vocabulary tag
was shared by three of the day's papers, and on those days none was. The gate
was protecting something real — 0i refused to let a model answer "what is today
about" with nothing to read — but it was answering a narrower question than it
was gating: **no group is no answer about grouping, not no answer at all.**

The 2026-08-18 paragraph is the proof, because it does both jobs in one breath:

    Four of today's papers carry the tag "Urban Transport and Accessibility" …
    Outside this group, another paper introduces … while another utilizes …

The first clause is a measured claim and needs a group. The second needs
nothing but the papers, and it is what a day with no group can still have.

So the paragraph now has two paths, and everything here is about keeping the
second one from quietly becoming the first. Three defences, in the order they
were added and in decreasing order of how much they are trusted:

1. **The facts block has nothing to group from.** No group lines, and — after
   this batch — no per-paper tag lists either. See the leak below.
2. **The prompt says so**, in the two shapes the model actually produced.
3. **A narrow regex refuses two claim shapes** and records why. It is the last
   defence, not the first, and it is not a general detector of invention.

The leak in 1 is the part worth reading. The first generation for 2026-08-03
opened *"Two of today's papers carry the tag 'Transportation Planning and
Optimization'"* — a group that does not exist, because two is below the
threshold and no such line was ever in the facts block. The model had not
invented the tag. It read the same string in two `(its tags: …)` clauses and
counted them itself. Asking the prompt not to do that is the arrangement 0i
rejected in the sentence this project keeps coming back to: *"we did not write
in the prompt that it must not name a topic, we removed the place in the output
where a topic could go."*
"""

from __future__ import annotations

from datetime import date

import pytest

from pipeline.llm import LLMClient, LLMResponse
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
    HIGHLIGHT_LIMIT,
    PARAGRAPH_MIN_ITEMS,
    PROMPT_PATH,
    build_facts,
    render_facts,
    write_paragraph,
)

DAY = date(2026, 8, 28)


def _item(key, tags=(), what="It measured a thing in a city.", headline=0.5, title=None):
    it = Item(work_key=key, bibliography=Bibliography(title=title or f"Paper {key}"))
    it.entities = Entities(
        methods=[EntityRef(id=f"method:{t}", label=t) for t in tags]
    )
    it.summary = Summary(en=SummaryEn(what=what, why="It matters."))
    it.scores.headline = headline
    return it


def _ungrouped(n=6):
    """`n` papers, every one carrying a tag nobody else carries."""
    return [_item(f"arxiv:2608.0000{i}", tags=[f"tag{i}"]) for i in range(n)]


def _grouped(n=6):
    """`n` papers, three of which share one tag."""
    items = [_item(f"arxiv:2608.0000{i}", tags=["clustering"]) for i in range(3)]
    items += [_item(f"arxiv:2608.1000{i}", tags=[f"other{i}"]) for i in range(n - 3)]
    return items


def _model(text):
    return LLMClient(task="synthesis", caller=lambda system, user: LLMResponse(text=text))


# --------------------------------------------------------------------------
# The floor
# --------------------------------------------------------------------------


def test_the_floor_is_derived_from_what_the_paragraph_can_name():
    """One more than `HIGHLIGHT_LIMIT`, and tied to it rather than typed.

    A paragraph that names every paper in the issue is the issue, not a summary
    of it. The two constants have to move together or the rule stops meaning
    what its comment says.
    """
    assert PARAGRAPH_MIN_ITEMS == HIGHLIGHT_LIMIT + 1


def test_a_day_below_the_floor_stays_silent_and_says_why(repo):
    """2026-08-23 published one paper. Its "summary paragraph" is that paper."""
    items = _ungrouped(2)
    facts = build_facts(DAY, items)

    out = write_paragraph(facts, client=_model("Something."))

    assert out["omitted"] is True
    assert out["text"] is None
    assert "2 paper(s) is too few" in out["reason"]
    # And the older half of the reason survives, because both are true and the
    # field is the only reason this batch could be diagnosed at all.
    assert f"{GROUP_MIN_PAPERS} or more" in out["reason"]


def test_a_day_at_the_floor_is_written(repo):
    items = _ungrouped(PARAGRAPH_MIN_ITEMS)
    facts = build_facts(DAY, items)

    out = write_paragraph(facts, client=_model("One paper did a thing."))

    assert out["omitted"] is False
    assert out["groups"] == 0


def test_a_day_with_nothing_to_compress_stays_silent(repo):
    """Papers without summaries are not offered as highlights, so a day can
    clear the floor and still have nothing for the paragraph to hold."""
    items = [_item(f"arxiv:2608.0000{i}", tags=[f"tag{i}"], what="") for i in range(6)]
    facts = build_facts(DAY, items)

    out = write_paragraph(facts, client=_model("Something."))

    assert out["omitted"] is True
    assert "fewer than two" in out["reason"]


# --------------------------------------------------------------------------
# What the model is shown, and what it is not
# --------------------------------------------------------------------------


def test_an_ungrouped_facts_block_has_no_group_and_no_tags(repo):
    """The structural defence, and the one this batch had to strengthen.

    No group line, so no count to copy. No tag list, so no two papers can be
    seen to share a string and counted into a group that was never measured.
    """
    block = render_facts(build_facts(DAY, _ungrouped(6)))

    assert "carry the tag" not in block
    assert "of today's papers" not in block
    assert "groups above" not in block
    assert "its tags" not in block, "the 2026-08-03 leak"
    assert "tag0" not in block and "tag5" not in block
    # And it is not empty — the papers themselves are all there.
    assert block.count("\n") + 1 == HIGHLIGHT_LIMIT


def test_a_grouped_facts_block_keeps_both(repo):
    """The grouped path is untouched. The tags are useful there precisely
    because there is a group for a paper to be outside of."""
    block = render_facts(build_facts(DAY, _grouped(6)))

    assert f'{GROUP_MIN_PAPERS} of today\'s papers carry the tag "clustering"' in block
    assert "is not in any of the groups above" in block
    assert "its tags" in block


def test_a_single_sentence_summary_does_not_arrive_with_two_full_stops(repo):
    block = render_facts(build_facts(DAY, _ungrouped(6)))

    assert ".." not in block


# --------------------------------------------------------------------------
# The last defence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        "Two of today's papers carry the tag \"Transportation Planning\".",
        "Three of today's papers address mobility.",
        "Several of today's papers use satellite imagery.",
        "This issue features research across several distinct areas.",
        "This digest presents work on cities.",
        "Today's issue brings together three strands.",
    ],
)
def test_a_claim_about_the_day_as_a_set_is_refused(repo, claim):
    """Both shapes the model actually produced, and their neighbours.

    The count is 0Z-B's D273 again — a claim about a population nobody
    measured. The sentence whose subject is the issue is the same error stated
    as a negative, which is how it slipped past the prompt's "no opening
    flourish" line on the first try.
    """
    facts = build_facts(DAY, _ungrouped(6))

    out = write_paragraph(facts, client=_model(claim + " A paper did a thing."))

    assert out["omitted"] is True
    assert "as a set" in out["reason"]
    assert out["text"] is None


def test_an_honest_ungrouped_paragraph_is_published(repo):
    facts = build_facts(DAY, _ungrouped(6))
    text = (
        "One paper measured street trees in Leipzig. Another modelled metro "
        "demand under uncertainty. A third reviewed wearables in city health "
        "services."
    )

    out = write_paragraph(facts, client=_model(text))

    assert out["omitted"] is False
    assert out["text"] == text


def test_the_guard_does_not_touch_a_day_that_really_has_a_group(repo):
    """On a grouped day "3 of today's papers carry the tag …" is the sentence
    the paragraph is *for*. The guard must not reach it."""
    facts = build_facts(DAY, _grouped(6))
    text = 'Three of today\'s papers carry the tag "clustering". One did a thing.'

    out = write_paragraph(facts, client=_model(text))

    assert out["omitted"] is False
    assert out["text"] == text


def test_the_model_may_still_refuse(repo):
    """The path 0i opened stays open. A rule that removes the model's ability
    to say "nothing here" would be the padding this design refuses."""
    facts = build_facts(DAY, _ungrouped(6))

    out = write_paragraph(facts, client=_model("NOTHING TO SAY"))

    assert out["omitted"] is True
    assert "too thin" in out["reason"]


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------


def test_the_prompt_tells_the_model_what_an_empty_facts_block_means(repo):
    prompt = " ".join(PROMPT_PATH.read_text(encoding="utf-8").split())

    assert "Some days have no groups, and those days still get a paragraph" in prompt
    assert "Do not say how many of today's papers do anything" in prompt
    assert "Do not write a sentence whose subject is the issue" in prompt
    assert "Open with the first paper" in prompt


def test_the_prompt_still_forbids_describing_the_shape_of_the_set(repo):
    """Saying the papers are varied is the forbidden claim with a minus sign.
    It is as unmeasured as saying they converge."""
    prompt = " ".join(PROMPT_PATH.read_text(encoding="utf-8").split()).lower()

    assert "do not say the papers are unalike, scattered, varied, or wide-ranging" in prompt
    assert "say nothing about the shape of the set they form" in prompt


def test_the_prompt_version_moved_with_the_prompt(repo):
    """CLAUDE.md's rule, and the reason for it: a cached response to the old
    prompt answers a question no longer being asked."""
    assert LLMClient(task="synthesis").prompt_version == "synthesis/daily@0.4.0"


# --------------------------------------------------------------------------
# D127 — a past issue is read, never rewritten
# --------------------------------------------------------------------------


def test_regenerating_a_paragraph_writes_nothing(repo):
    """The regression check this batch ran over 73 published days generated
    paragraphs and compared them. It must not be able to publish one.

    `build_facts` and `write_paragraph` are pure with respect to the archive;
    only `stage_issue` writes an issue. Pinned because the comparison is a
    thing somebody will want to run again.

    `content/state/` is excluded and is the one thing that legitimately moves:
    the cumulative LLM spend goes up when you spend, which is what cumulative
    means. It lives under `content/` only because that is the directory CI
    keeps. `verify_phase0`'s idempotency check draws the same line for the
    same reason.
    """
    from pipeline import paths

    def archive():
        return {
            p: p.read_bytes()
            for p in paths.CONTENT.rglob("*")
            if p.is_file() and "state" not in p.relative_to(paths.CONTENT).parts
        }

    before = archive()

    facts = build_facts(DAY, _ungrouped(6))
    write_paragraph(facts, client=_model("A paper did a thing. Another did too."))

    assert archive() == before
