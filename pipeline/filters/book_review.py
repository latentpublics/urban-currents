"""Book reviews are real publications in our journals, and not our kind (V1-2).

Three of the nine `drop_not_our_kind` journal labels are book reviews, and they
are the one drop reason in that set that a machine can see: a review of *Exiles
in New York City* is not a paper about cities that happens to be qualitative, it
is a different genre with a different title grammar.

**Detected on the title, because that is the field we already have.** OpenAlex
carries a `type` and a page range that would settle it outright, but neither is
on `Bibliography` today, so keying on them would only work for items collected
after the field was added — and the whole point is to fix the days already on
disk. `openalex_type` is read when present and treated as confirmation, never as
a requirement.

**Demoted, not dropped.** A filter would remove them unconditionally; a ranking
penalty removes them only while there is something better to publish. On a thin
journal day a book review is still a true record of what a tracked journal
published, and the alternative is a hole in the issue. That is also why this is
not routed into `Also published today`: that section means "we could not read
it", and overloading it with "we could read it and chose not to lead with it"
would make one section answer two questions.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# `<i>…</i>` and `<scp>…</scp>` reach us from Crossref/OpenAlex titles, where a
# publisher marked up the reviewed book's title. A research paper's title does
# not usually contain another work's title in italics.
MARKUP = re.compile(r"</?(i|scp|em)>", re.I)

# ", by Alvin Wong" — the reviewed book's author, appended after the title. The
# capital is required: ", by the numbers" is a subtitle, not an attribution.
BY_AUTHOR = re.compile(r",\s*by\s+[A-Z][\w'’.\-]+", re.U)

# Explicit genre labels, which some journals put in the title itself.
GENRE = re.compile(
    r"^\s*(book\s+review|review\s+essay|reviews?\s*:)|^\s*review\s+of\s+[A-Z\"“<]",
    re.I,
)

# **Only `book-review`.** The first draft of this included OpenAlex's `review`
# type and it caught 52 works, nearly all of them systematic literature reviews
# — "A Systematic Literature Review of Urban Noise Modeling", "A scientometric
# review of UAVs and transportation science". Those are a research genre this
# digest covers, not the genre it is trying to demote, and demoting them would
# have quietly removed survey papers from the journal path. `editorial` and
# `paratext` are excluded for the same reason: they are a different problem
# (front matter with no abstract, already handled by the unreadable path) and
# folding them in here would make one rule answer three questions.
OPENALEX_REVIEW_TYPES = frozenset({"book-review"})


def signals(title: str, openalex_type: Optional[str] = None) -> dict[str, Any]:
    """Every signal, reported separately so a match can be argued with."""
    t = title or ""
    return {
        "markup": bool(MARKUP.search(t)),
        "by_author": bool(BY_AUTHOR.search(t)),
        "genre_phrase": bool(GENRE.search(t)),
        "openalex_type": (openalex_type or "").lower() in OPENALEX_REVIEW_TYPES,
    }


def is_book_review(title: str, openalex_type: Optional[str] = None) -> bool:
    """Two independent signals, or an explicit genre label, or OpenAlex saying so.

    One signal alone is not enough. Italic markup appears in perfectly ordinary
    titles that cite a species name or a ship; `, by X` appears in a handful of
    essayistic titles. Requiring two of them, measured over the 90-day backfill,
    is what took the false positives to zero without losing any of the three
    labelled reviews.
    """
    s = signals(title, openalex_type)
    if s["genre_phrase"] or s["openalex_type"]:
        return True
    return s["markup"] and s["by_author"]


def demotion(title: str, openalex_type: Optional[str] = None) -> float:
    """Multiplier applied to the journal ranking score. Not a filter.

    0.0 rather than a small fraction: the score being multiplied is a weak
    placeholder proxy, so a proportional penalty would be arithmetic on a number
    that does not mean much. Zero puts reviews at the bottom of the journal
    ranking, where they publish if and only if the day has nothing else — which
    is the behaviour asked for.
    """
    return 0.0 if is_book_review(title, openalex_type) else 1.0
