"""The headline line, written as a title rather than quoted from a summary.

## What it replaced, and why

`score.headline.headline_line` took the first sentence of `summary.en.what`
verbatim, which produced things like *"The researchers built the Rail Transit
Station Knowledge Graph (RTSKG), a dataset that models spatial and semantic
interactions between…"*. That is a narration of an abstract sitting in the
position a title should occupy.

## ★ Why this is the most dangerous change in the batch

"A headline that grabs people" is one step from the thing this service has spent
every design decision refusing to do. Quiet days are declared instead of padded.
Unmeasured connections are not drawn. Papers without an open abstract are listed
rather than summarised. **The headline is the most visible line on the page, so
it is where overstatement would cost the most.**

The prompt therefore forbids superlatives, novelty claims, hype verbs,
questions, second person, and any fact not present in the three fields it is
given. `check()` below enforces the mechanical half of that after the fact,
because a prompt is a request and a check is a guarantee. **The goal is
compression, not excitement**, and a dull accurate line is a success.

## Shape of the work

- Its **own task and prompt version**, not part of `summarize`. Bumping the
  summarize prompt would invalidate the cache for 2,224 items and re-summarise
  the archive; this needs one call per issue.
- **One call per issue**, for the selected headline item only.
- **Falls back to the extractive first sentence** on any failure — no key, no
  budget, a refused rule, an empty answer. An issue must never fail to publish
  because a headline could not be written.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional, Sequence

from ..llm import LLMBudgetExceeded, LLMClient, LLMQuotaExhausted, LLMUnavailable
from ..models import Item

PROMPT_PATH = Path(__file__).parent / "prompts" / "headline.md"

MAX_WORDS = 12

# ★ How many of the day's papers the model is shown (phase 0Z-B, B1).
#
# Not all of them. A twenty-four paper day handed over whole is an invitation to
# generalise, and the failure this batch exists to prevent is a common theme
# nobody measured. Six is enough to describe a day and few enough that the line
# has to name things rather than summarise a corpus.
#
# **The material and `check()` see the same set**, deliberately. A check run
# against a subset of what the model was shown is a check that can be fooled by
# anything the model read and the checker did not.
MATERIAL_ITEMS = 6

# The mechanical half of the prompt's prohibitions. A prompt is a request; this
# is the part that does not depend on the model having complied.
BANNED = re.compile(
    r"\b("
    r"breakthrough|first-ever|revolutionary|game-chang\w*|unprecedented|finally|"
    r"landmark|groundbreaking|ground-breaking|"
    r"revolutioni[sz]es?|transforms?|cracks?|solves?|unlocks?|disrupts?|redefines?|"
    r"you|your|here'?s why|what if"
    r")\b",
    re.I,
)

# ★ The failure that only appears once a line covers several papers (0Z-B, B2).
#
# Inventing a number is caught above. Inventing a **theme** is not: "a day of
# urban mobility research" sounds like a fact and is not one — that three of
# nine papers share a field is true, that it constitutes a trend is something
# nobody measured. Two families are mechanical enough to catch here, and the
# rest is what B6's read-through is for.
#
#   quantity   we know how many papers we published and nothing about whether
#              that is many or few. `a wave of`, `several`, `most`, `a flurry`.
#   linkage    a claim that the papers relate to each other — converging,
#              contradicting, building on one another — which would require a
#              comparison we did not run.
QUANTITY = re.compile(
    r"\b("
    r"wave|flurry|surge|spate|burst|cluster of|slew|raft|host of|"
    r"several|many|numerous|multiple|various|dozens?|"
    r"most|majority|handful|plenty|abundance|"
    r"trend|trending|trends|momentum|"
    r"dominat\w*|prolifera\w*|abound\w*|"
    r"this week'?s|today'?s crop|the day'?s crop"
    r")\b",
    re.I,
)
LINKAGE = re.compile(
    r"\b("
    r"converg\w*|diverg\w*|contradict\w*|corroborat\w*|"
    r"echo(es|ing)?|mirror(s|ing)?|parallel(s|ing)?|"
    r"builds? on|follows? up|in response to|at odds with|"
    r"all point\w*|together (they|these)|taken together|"
    r"complement(s|ing|ary)?|reinforc\w*"
    r")\b",
    re.I,
)
# `first` only as a novelty claim — "the first dataset to…" — and not as an
# ordinary word: "first-mile access" and "first-order effects" are the subject
# matter of this field, and banning the token outright would reject good lines.
FIRST_CLAIM = re.compile(r"\bfirst\b(?!\s*[-–]\s*(mile|order|come|hand|person|author|stage|principle))", re.I)

# Counting is not quoting. A spelled-out number is checked against the material
# the same way a digit is — see the note in `check`.
WORD_NUMBERS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b",
    re.I,
)

# A question does not stop being a question when the mark is dropped, so the
# terminal-punctuation rule is not enough on its own. "Why transit graphs
# matter" is the headline register this service is refusing.
INTERROGATIVE = re.compile(
    r"^\s*(can|could|will|would|should|does|do|did|is|are|was|were|has|have|"
    r"why|how|what|when|where|who|which)\b",
    re.I,
)


def prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _one(item: Item, label: str = "") -> str:
    """One paper's three fields, as the model sees them."""
    summary = item.summary.en
    head = f"{label}\n" if label else ""
    return (
        f"{head}"
        f"title: {item.bibliography.title}\n"
        f"what: {(summary.what if summary else '') or '(none)'}\n"
        f"why: {(summary.why if summary else '') or '(none)'}"
    )


def _material(item: Item, others: Optional[Sequence[Item]] = None) -> str:
    """The fields the line may be built from — one paper's, or the day's.

    With `others`, the lead comes first and is labelled as such, so a line
    about a single paper is still available to the model on a day where nothing
    stood out. Everything after it is the rest of what was published, capped at
    `MATERIAL_ITEMS` (0Z-B, B1).
    """
    if not others:
        return _one(item)
    rest = [it for it in others if it.work_key != item.work_key][: MATERIAL_ITEMS - 1]
    # The first block is **not** advertised as the highest scoring, and that
    # is deliberate (0Z-B, B6). The day form is used precisely when nothing
    # cleared the threshold, and on those days every item sits on the same
    # 0.44 — so the block that comes first is whichever work_key sorts last.
    # Telling the model it is the best paper of the day would be our own
    # prompt asserting something we did not measure. It is first because it
    # has to be somewhere.
    blocks = [_one(item, "PAPER 1:")]
    blocks += [_one(it, f"PAPER {i + 2}:") for i, it in enumerate(rest)]
    return "\n\n".join(blocks)


def _quotable(item: Item, others: Optional[Sequence[Item]] = None) -> str:
    """The material with the scaffolding removed, for `check()` to test against.

    ★ Not the same string the model is shown, and the difference matters
    (0Z-B). The blocks are labelled `PAPER 1`, `PAPER 2`, `PAPER 3` so the model
    can tell them apart and can still choose the single-paper form — and those
    labels put the digits 1, 2 and 3 into the text. Checking a line against
    them let `3 models of urban infrastructure risk` through: the "3" was not
    from any paper, it was from **our own numbering**.

    A checker that accepts its own scaffolding as evidence is a checker that
    can be fooled by anything the harness happens to write, so the quotable set
    is the paper fields and nothing else.
    """
    items = list(others) if others else [item]
    parts = []
    for it in items:
        summary = it.summary.en
        parts += [
            it.bibliography.title or "",
            (summary.what if summary else "") or "",
            (summary.why if summary else "") or "",
        ]
    return "\n".join(parts)


def fallback(item: Item) -> str:
    """The pre-0R behaviour: the first sentence of `what`, or the title.

    Kept as a real code path rather than as an error, because a day with no
    headline is a day that does not publish, and a plain quoted sentence is a
    perfectly honest thing to lead with.
    """
    what = (item.summary.en.what if item.summary.en else "") or ""
    first = what.split(". ")[0].strip()
    if first:
        return first if first.endswith(".") else first + "."
    return item.bibliography.title


def check(
    line: str, item: Item, others: Optional[Sequence[Item]] = None
) -> Optional[str]:
    """Why this line is unusable, or None if it is fine.

    Length is checked here rather than trimmed in Python. **Cutting a line to
    fit would produce a truncated claim**, and a truncated claim is a different
    claim — the prompt is asked to keep to twelve words and the answer is
    rejected if it did not.
    """
    text = (line or "").strip()
    if not text:
        return "empty"
    if "\n" in text:
        return "more than one line"
    words = text.split()
    if len(words) > MAX_WORDS:
        return f"{len(words)} words, limit {MAX_WORDS}"
    if text.endswith((".", "!", "?")):
        return "ends with terminal punctuation"
    if text.startswith(('"', "'", "“")):
        return "wrapped in quotation marks"
    banned = BANNED.search(text)
    if banned:
        return f"banned wording: {banned.group(0)!r}"
    if FIRST_CLAIM.search(text):
        return "novelty claim: 'first'"
    if INTERROGATIVE.match(text):
        return "interrogative form"
    quantity = QUANTITY.search(text)
    if quantity:
        return f"quantity or trend claim: {quantity.group(0)!r}"
    linkage = LINKAGE.search(text)
    if linkage:
        return f"claims a relation between papers: {linkage.group(0)!r}"
    # A line the material cannot support. Not a full provenance check — that is
    # what the prompt's field restriction is for — but a digit that appears in
    # neither the title nor the summary is a number from nowhere, and numbers
    # from nowhere are the failure this service most has to avoid.
    # **The union of everything the model was shown** (0Z-B, B4), minus our own
    # labels — see `_quotable`. A number that appears in the third paper's
    # summary is quotable; one that appears in none of them is from nowhere,
    # and one that appears only in the words "PAPER 3" is from us.
    material = _quotable(item, others).lower()
    for number in re.findall(r"\d[\d,.]*", text):
        if number not in material:
            return f"number not in the material: {number!r}"
    # **Spelled-out numbers too**, and this is not pedantry. The first run
    # produced "Pedestrian waiting times…, across four cities" for a paper whose
    # summary says "Singapore and three US cities". Four is *correct* — it is
    # 1 + 3 — and it is still a number the material never states, so a reader
    # cannot check it against the source we gave them. A count the model worked
    # out is exactly as unverifiable as one it invented, and this service does
    # not publish either.
    #
    # ★ **A count of the papers is still refused** (0Z-B, B4). "Three studies of
    # street networks" is a number we do have — but it is a fact about *our
    # selection*, not about the field: the model is shown at most
    # `MATERIAL_ITEMS` of a day that may hold twenty-four, so "three" would
    # invite a reader to conclude the day held three such papers when we never
    # counted. It is also a quantity word by another name, and the line above
    # bans those; allowing the digit while banning the word would be incoherent.
    # A count the reader cannot check against what we showed them is exactly
    # what this rule has refused since 0R.
    for word in WORD_NUMBERS.findall(text):
        if word.lower() not in material:
            return f"number not in the material: {word!r}"
    return None


def write_headline(
    item: Item,
    client: Optional[LLMClient] = None,
    use_llm: bool = True,
    others: Optional[Sequence[Item]] = None,
) -> tuple[str, str]:
    """Return `(line, basis)` — `basis` is `llm`, `fallback` or a refusal reason.

    `basis` travels with the line so a headline written by the model and one
    quoted from a summary are never indistinguishable, which is the same rule
    the label files follow for what a judge could see.
    """
    if not use_llm:
        return fallback(item), "fallback:llm_disabled"

    client = client or LLMClient(task="headline")
    if not client.available():
        return fallback(item), "fallback:unavailable"

    def ask(user: str, cache_key: str) -> Optional[str]:
        try:
            resp = client.complete(
                system=prompt_text(), user=user, cache_key=cache_key
            )
        except (LLMBudgetExceeded, LLMQuotaExhausted, LLMUnavailable):
            return None
        except Exception:  # noqa: BLE001
            return None
        return (resp.text or "").strip().strip('"').strip()

    # The cache key carries the shape, so a day-wide line and a single-paper
    # line for the same lead item cannot read each other's answer back (0Z-B).
    shape = "day" if others else "lead"
    line = ask(_material(item, others), f"{item.work_key}#{shape}")
    if line is None:
        return fallback(item), "fallback:unavailable"

    problem = check(line, item, others)
    if problem is None:
        return line, f"llm:{shape}"

    # **One retry, told exactly what it broke.** This is still the prompt
    # enforcing the rule rather than Python trimming the answer — the model gets
    # to write a different twelve-word line, which is a different claim honestly
    # made, where a truncation would be the same claim dishonestly shortened.
    #
    # Its own cache key, or the retry would read back the answer that failed.
    # The hint is specific to what broke, because a generic "try again" gets the
    # same answer back. A rejected count in particular needs saying plainly:
    # the model is not hallucinating when it writes "four modes" for a list of
    # four, it is *counting*, and it will keep counting unless told that a
    # number the source does not state is not available to it.
    if "number not in the material" in problem:
        hint = (
            "Do not include any number, in digits or in words. If the material "
            "lists items, name them or describe them without counting them. "
            "Do not count the papers either."
        )
    elif "quantity or trend claim" in problem or "claims a relation" in problem:
        # "Try again" gets the same sentence back here: the model is not being
        # careless, it is doing what a headline usually does. It has to be told
        # that the connection itself is unavailable to it (0Z-B, B4).
        hint = (
            "Name what the papers are about without characterising the day. Do "
            "not say how many there were, whether that is many or few, or that "
            "they relate to, support or contradict one another — none of that "
            "was measured. If they share no subject, describe the highest "
            "scoring paper alone."
        )
    else:
        hint = f"At most {MAX_WORDS} words, no terminal punctuation."
    retry = ask(
        f"{_material(item, others)}\n\n"
        f"Your previous answer was rejected: {problem}.\n"
        f"Answer again, obeying every rule. {hint}",
        # The hint is part of the key. It was not, and the cache then replayed
        # the answer to a *different* question — a sharper retry prompt read
        # back the reply the vaguer one had produced, so the fix looked like it
        # had no effect. A cache key must cover everything the prompt varies by.
        f"{item.work_key}#{shape}#retry-{hashlib.sha1(hint.encode()).hexdigest()[:8]}",
    )
    if retry is not None:
        second = check(retry, item, others)
        if second is None:
            return retry, f"llm:{shape}:retry"
        problem = f"{problem}; retry {second}"

    return fallback(item), f"fallback:{problem}"
