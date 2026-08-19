"""A corrigendum is not a paper (phase 0P, Q3).

The subfield-check pool handed one of these over to be judged for scope — a
`Corrigendum to "Understanding employees' residential choices..."` — and the
question it was drawn to answer does not apply to it. It is not a paper in the
wrong subfield; it is not a paper.

Counted over the whole archive rather than fixed on the one that surfaced:
**31 of 11,119 candidates** carry a correction title, 26 of them are stored as
items, and **2 reached a published issue**. One of those two is titled, in
full, `Correction`.

## Why this is a sibling of the book-review filter and not the same rule

The genres are different but the mechanism is right for both: detected on the
title, because that is the field we already have for days already on disk, and
**demoted rather than dropped**, so nothing is removed unconditionally.

The arguments are not the same strength, though, and that is worth writing
down. A book review is demoted because it is *"still a true record of what a
tracked journal published"* — publish it on a thin day and the reader gets
something real. A correction notice is weaker than that: it announces a change
to a **different** paper, and its abstract is the notice. Demotion is chosen
anyway, for two reasons. The first is precedent: the same mechanism, doing the
same job, argued for once. The second is that a hard drop would be the first
unconditional content removal in the pipeline, and 23 of the 31 have no
abstract of their own and are already routed to `Also published today` by the
unreadable path — so a drop would buy the 8 remaining and cost a new kind of
rule.

## What is deliberately not matched

Only the **leading** formulaic phrase. A title that merely ends in "Correction"
— `...Cross-Domain Correction`, `Learning to Distort: ...` — is a paper about
correcting something, and the first draft of this caught three of those.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# The formulaic opening, allowing for a leading quote or bracket that some
# publishers put in front of the reviewed title.
NOTICE = re.compile(
    r"^\s*[\"'\u201c\u2018(\[]*\s*"
    r"(corrigendum|erratum|errata|correction|retraction|withdrawal|addendum|"
    r"publisher'?s? note|editorial expression of concern)\b",
    re.I,
)

# OpenAlex says so outright when it knows. Read as confirmation, never required
# — the whole point is to also catch the days already on disk.
OPENALEX_CORRECTION_TYPES = frozenset({"erratum", "correction", "retraction"})


def signals(title: str, openalex_type: Optional[str] = None) -> dict[str, Any]:
    """Every signal separately, so a match can be argued with."""
    return {
        "notice_phrase": bool(NOTICE.search(title or "")),
        "openalex_type": (openalex_type or "").lower() in OPENALEX_CORRECTION_TYPES,
    }


def is_correction(title: str, openalex_type: Optional[str] = None) -> bool:
    """Either signal is enough.

    Unlike `is_book_review`, one signal suffices: the opening phrase is
    formulaic rather than suggestive, and measured over all 11,119 candidates it
    matched 31 titles with no false positive among them.
    """
    s = signals(title, openalex_type)
    return s["notice_phrase"] or s["openalex_type"]


def demotion(title: str, openalex_type: Optional[str] = None) -> float:
    """Multiplier on the journal ranking score. Not a filter.

    0.0, for the same reason the book-review filter uses it: the score being
    multiplied is a weak placeholder proxy, so a proportional penalty would be
    arithmetic on a number that does not mean much.
    """
    return 0.0 if is_correction(title, openalex_type) else 1.0
