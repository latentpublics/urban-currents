"""The daily synthesis layer (phase 0i, V3).

An issue has been a headline and a list of cards with nothing in between. What
goes in between is the question this module answers, and the answer is
deliberately not the obvious one.

**Why not "what is the field doing".** A synthesis of 5,000 messages can report
consensus because the consensus is in the data. Twenty-four papers contain no
such signal. "These three papers suggest a shift towards X" is plausible,
unfalsifiable, and precisely the thing this service exists not to produce.

**So the object of synthesis changes.** Not *where is the field going* but *how
does today connect to what came before* — a question the citation graph can
actually answer, and one nobody else is in a position to ask, because answering
it requires an archive of one's own.

Four measured sections and one written one:

1. **Today's shape** — composition, and deviations from a 30-day baseline.
   Nothing when nothing deviates. Silence is a finding.
2. **What today stands on** — foundation-canon anchors, with the rare-event
   version ("first paper in 90 days to cite Arnstein"). Instruments are
   excluded: that a paper uses random forests says nothing about which field it
   belongs to.
3. **What clusters** — same-day bibliographic coupling, **named by the shared
   references themselves** rather than by a topic. Naming the topic is where an
   LLM would invent one; printing the shared works makes that structurally
   impossible.
4. **Institutions and authors** — frequencies, never ranked. The moment this
   produces a leaderboard we have become an evaluation body, which is not the
   position v1.0 chose.
5. **One paragraph** — written from the facts above and nothing else, and
   **omitted entirely when the facts are thin**. Forcing a paragraph every day
   produces filler, and filler is the failure mode this whole design is built
   against. Same principle as the quiet day.

   What "thin" means changed in 0Z-F (S1) and what it protects did not. The
   paragraph asks two questions of a day and they have separate answers:
   *what arrived together*, which needs a measured group and is silent without
   one, and *what arrived*, which needs only the day's own papers. Losing the
   second because the first was empty is what silenced every issue from
   2026-08-21. A day with no group now gets a paragraph that **makes no
   grouping claim** — the facts block it is written from contains no count of
   papers sharing anything, so there is nothing to copy one from.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from . import paths, store
from .config import cfg
from .models import Item

# How far back the baseline looks. Long enough that a weekly rhythm does not
# read as a deviation, short enough to still be "lately".
BASELINE_DAYS = 30

# A tag has to clear both to be worth a line: enough occurrences that it is not
# a single paper, and enough of a multiple that it is not noise.
DEVIATION_MIN_TODAY = 3
DEVIATION_MIN_RATIO = 3.0

# Days of archive required inside the window before a comparison is made at all.
# Without it, the very first issues report every tag as a spike against a
# baseline of zero — true arithmetic, no information.
MIN_BASELINE_DAYS = 7

# The canon is a moving object; "first in N days" is only sayable over the range
# the reference base actually covers.
RARE_EVENT_DAYS = 90


# Publisher markup reaches us inside titles — `<i>`, `<scp>`, `<sub>`. It is
# harmless in a card, where the title is one field, and not harmless here: a
# title is quoted into the facts block and then into the paragraph, and the
# first day this ran the model dutifully reproduced "<scp>NOCTURNAL
# INFORMALITY</scp>" in its prose.
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def clean_title(title: str) -> str:
    return _TAG.sub("", title or "").strip()


def _first_sentence(what: str) -> str:
    """The opening sentence of a summary, without its full stop.

    The full stop is stripped because every caller adds one, and a `what` that
    is a single sentence therefore reached the facts block as "It measured a
    thing.." — a doubled period in the one text a model is asked to write from.

    **The split itself is naive and stays naive**, which is worth saying rather
    than hiding: it breaks on the first ". " and so cuts
    *"…across 109 U.S. Cities"* down to *"…across 109 U.S."*. Doing better
    needs an abbreviation-aware splitter, which is a change with its own tests
    and its own failure modes; guessing at one here — "split only before a
    capital" — would trade a visible truncation for an invisible one. Recorded
    in 0Z-F rather than half-fixed.
    """
    first = (what or "").split(". ")[0].strip()
    return first[:-1].strip() if first.endswith(".") else first


def _foundation_canon() -> dict[str, dict]:
    """Foundation canon entries by OpenAlex id, with the titles for display."""
    import json

    path = paths.CONTENT / "canon" / "candidates.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {
        c["openalex_id"]: c
        for c in (doc.get("candidates") or [])
        if c.get("class") == "foundation"
    }


def _references_by_work() -> dict[str, list[str]]:
    from .graph.citation import load_reference_base

    return {
        r["work_key"]: (r.get("referenced_works") or []) for r in load_reference_base()
    }


def _resolved_titles() -> dict[str, dict]:
    from .graph.daily_canon import load_resolved

    return load_resolved()


# --------------------------------------------------------------------------
# 1. Today's shape
# --------------------------------------------------------------------------


def composition(items: list[Item], unreadable_n: int) -> dict[str, int]:
    from .run_stages import _is_whitelist_journal

    journal = sum(1 for it in items if _is_whitelist_journal(it))
    return {
        "published": len(items),
        "journal": journal,
        "arxiv": len(items) - journal,
        "unreadable": unreadable_n,
    }


def _tags_of(item: Item) -> list[str]:
    e = item.entities
    return [t.label for t in (e.methods + e.data + e.tools + e.topics)]


def deviations(
    d: date, items: list[Item], window_days: int = BASELINE_DAYS
) -> dict[str, Any]:
    """Tags today against their own recent average. Only what stands out.

    **An empty baseline is not a low baseline.** The first version of this
    reported four deviations for 2026-08-05 against a baseline built from one
    day of archive, every one of them "3 today against 0.0 per day" — which says
    nothing except that the archive had barely started. Below
    `MIN_BASELINE_DAYS` the section reports that it cannot compare yet, and
    reports nothing else.

    **This section is bounded by the vocabulary, not by the method.** Overlay
    tags run about 1.2 per item and the method vocabulary tops out in the
    thirties, so even with a full baseline most days have nothing here that
    clears the bar — and the count of deviations found is itself the measurement
    of what the pending vocabulary curation is worth.

    ★ **The baseline counts published items, not collected ones (1A, B2).**
    It used to walk `store.iter_items()`, which is everything the archive
    holds whether or not any issue ever carried it. So `today` was tags among
    the ~6 papers an issue publishes and `baseline_per_day` was tags among the
    ~29 papers a day collects — two different populations, one divided by the
    other. Measured over the 30 days to 2026-09-02: 864 items in the store
    against 297 in the issues, **567 of them never published**, a 2.9x bias
    against the ratio ever clearing.

    It was not a threshold that was too high; it was a fraction whose halves
    counted different things, and no threshold can be set correctly against a
    denominator that means something else. The measured effect: `Urban
    Transport and Accessibility` reached 3 occurrences on 2026-08-25 against a
    baseline of 5.80/day, requiring 17.4 — from a day that published 7 papers
    in total. The bar was above the ceiling.
    """
    today = Counter(t for it in items for t in _tags_of(it))

    start = d - timedelta(days=window_days)
    history: Counter = Counter()
    days_seen: set[str] = set()
    for issue in store.iter_issues():
        if not (start <= issue.date < d):
            continue
        # A day that published nothing is still a day the comparison saw: it
        # is a zero in the average, not an absence from it. Skipping it would
        # raise every baseline by shrinking the divisor, which is the same
        # class of error this fix is undoing.
        days_seen.add(str(issue.date))
        for key in issue.items:
            item = store.load_item(key)
            if not item:
                continue
            history.update(set(_tags_of(item)))

    return _verdict(today, history, len(days_seen), window_days)


def _verdict(
    today: Counter, history: Counter, n_days: int, window_days: int
) -> dict[str, Any]:
    """The rule itself, given both halves already counted.

    ★ Split out in 1B so the pipeline and the renderer cannot drift.
    `deviations` gathers the two counters from the store for one day;
    `deviations_over_archive` gathers them for every issue at once out of an
    index a render already holds. Both then land here, so there is exactly one
    place where a threshold meets a baseline — which is the point, because a
    stored value and a derived one that disagreed would be the next bug rather
    than a fix.
    """

    def scan(n: int) -> list[dict]:
        out = []
        for tag, count in today.most_common():
            if count < DEVIATION_MIN_TODAY:
                continue
            baseline = history.get(tag, 0) / max(1, n)
            if count < max(DEVIATION_MIN_RATIO * baseline, DEVIATION_MIN_TODAY):
                continue
            out.append({
                "label": tag,
                "today": count,
                "baseline_per_day": round(baseline, 2),
                "window_days": n,
            })
        return out[:5]

    if n_days < MIN_BASELINE_DAYS:
        return {
            "found": [],
            "baseline_days": n_days,
            "status": "NO_BASELINE",
            "note": (
                f"{n_days} day(s) of archive inside the {window_days}-day window; "
                f"{MIN_BASELINE_DAYS} needed before a deviation means anything"
            ),
            # Nothing here reaches the issue. It exists so the vocabulary
            # bottleneck can be *measured* while the baseline is too short to
            # publish from: how many tags would clear the bar if the archive
            # were long enough. That count is the size of the prize for the
            # pending vocabulary curation.
            "would_find_if_baseline_were_long_enough": scan(n_days),
            "distinct_tags_today": len(today),
        }

    return {
        "found": scan(n_days)[:5],
        "baseline_days": n_days,
        "status": "OK",
        "distinct_tags_today": len(today),
    }


def deviations_over_archive(
    issues: list, index: dict[str, Item], window_days: int = BASELINE_DAYS
) -> dict[date, dict[str, Any]]:
    """`tag shift` for **every** issue, from an archive already in memory (1B).

    Why this exists: the fix in 1A is real but it only reaches days the pipeline
    runs after it. Every issue already published carries the old, structurally
    empty number in its file, and **an issue is immutable once published**
    (D127) — a rule `backfill_issues.py`, `pages.yml` and `site.py` all stand
    on, and which D312 refused to bend for a smaller reason than this one.

    So the value is derived at render instead, exactly as `site.py` already
    derives the per-issue code and data counts and deliberately does not store
    them. Both halves of the fraction are in the archive: an issue names its
    items, an item carries its tags. Nothing is rewritten, and every past issue
    gets the corrected number — 29 of 80 issues show one, against the 10 whose
    stored value is non-empty.

    **Cost is why this takes the whole archive at once rather than one call per
    page.** Building the per-day tag counters once and walking a 30-entry window
    per issue is 0.044s over 80 issues, against the 152s `build_issue_pages`
    already spends rendering them — 0.02%. Calling `deviations()` once per page
    would instead re-read every item file 80 times.

    ★ **The value moves as the archive fills, and that is intended.** Four
    backfilled days (08-05, 08-07, 08-10, 08-11) were recorded `NO_BASELINE`
    because when they were assembled there were not yet seven days of archive
    behind them; there are now, so they get a real comparison. The measurement
    is defined against the archive's own 30-day average, the archive is what it
    is today, and freezing the number would mean storing it — the thing this
    function exists not to do. The home page's methodology note says so, because
    a number that can change has to admit that it can.
    """
    day_tags: dict[date, Counter] = {}
    for issue in issues:
        c: Counter = Counter()
        for key in issue.items:
            item = index.get(key)
            if item:
                c.update(set(_tags_of(item)))
        day_tags[issue.date] = c

    dates = sorted(day_tags)
    out: dict[date, dict[str, Any]] = {}
    for d in dates:
        start = d - timedelta(days=window_days)
        history: Counter = Counter()
        n_days = 0
        for other in dates:
            if start <= other < d:
                n_days += 1
                history.update(day_tags[other])
        out[d] = _verdict(day_tags[d], history, n_days, window_days)
    return out


# --------------------------------------------------------------------------
# 2. What today stands on
# --------------------------------------------------------------------------


def canon_anchors(d: date, items: list[Item], limit: int = 4) -> list[dict]:
    """Foundation works today's papers cite, and the ones long unseen.

    Foundation only. An instrument — random forests, a transformer — is cited by
    every field at once and says nothing about which one this is.
    """
    canon = _foundation_canon()
    if not canon:
        return []
    refs = _references_by_work()
    keys = {it.work_key for it in items}

    citing: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        for ref in refs.get(key, []):
            if ref in canon:
                citing[ref].append(key)

    if not citing:
        return []

    # When was each of these last cited by the archive, before today?
    start = d - timedelta(days=RARE_EVENT_DAYS)
    last_seen: dict[str, date] = {}
    covered_from: Optional[date] = None
    for item in store.iter_items():
        pub = item.first_published
        if not pub or pub >= d or pub < start:
            continue
        covered_from = pub if covered_from is None else min(covered_from, pub)
        for ref in refs.get(item.work_key, []):
            if ref in citing and (ref not in last_seen or last_seen[ref] < pub):
                last_seen[ref] = pub

    out = []
    for ref, cited_by in sorted(citing.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        entry = canon[ref]
        seen = last_seen.get(ref)
        first_in_window = (
            seen is None
            and covered_from is not None
            and (d - covered_from).days >= RARE_EVENT_DAYS - 1
        )
        # One paper citing one foundational work is not an anchor, it is a
        # citation. Two or more is a shared footing; a single one is worth a
        # line only when it is the first in the whole window.
        if len(cited_by) < 2 and not first_in_window:
            continue
        out.append({
            "openalex_id": ref,
            "title": clean_title(entry.get("title") or ref),
            "year": (entry.get("publication_date") or "")[:4] or None,
            "authors": (entry.get("authors") or [])[:2],
            "citing_today": len(cited_by),
            "citing_work_keys": sorted(cited_by),
            "days_since_last_cited": (d - seen).days if seen else None,
            # Only sayable if the archive actually covers the window. "First in
            # 90 days" over 12 days of archive would be a lie made of true
            # numbers.
            "first_in_window": first_in_window,
        })
    return out[:limit]


# --------------------------------------------------------------------------
# 3. What clusters
# --------------------------------------------------------------------------


def clusters(d: date, items: list[Item], limit: int = 3) -> list[dict]:
    """Papers sharing references — today with today, and today with the archive.

    The shared references are printed by name. That is the whole design: a
    cluster described by its members' shared bibliography cannot acquire an
    invented theme, because there is no place in the output for one.
    """
    from .graph.citation import compute_coupling

    refs = _references_by_work()
    resolved = _resolved_titles()
    keys = [it.work_key for it in items if refs.get(it.work_key)]
    if len(keys) < 2:
        return []

    titles = {it.work_key: clean_title(it.bibliography.title) for it in items}
    min_shared = int(cfg("graph.coupling.min_shared", 3))

    out: list[dict] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = [r for r in refs[a] if r in set(refs[b])]
            if len(shared) < min_shared:
                continue
            named = [
                clean_title(resolved[r]["title"])
                for r in shared
                if r in resolved and resolved[r].get("title")
            ]
            out.append({
                "scope": "today",
                "work_keys": [a, b],
                "titles": [titles.get(a, a), titles.get(b, b)],
                "shared": len(shared),
                "shared_titles": named[:3],
                "partner_date": str(d),
            })
    out.sort(key=lambda r: (-r["shared"], r["work_keys"]))

    # Today against the archive. `compute_coupling` already applies the moving
    # window, so this reuses it rather than re-deriving the rule.
    todays = set(keys)
    archive: list[dict] = []
    for pair in compute_coupling():
        a, b = pair["a"], pair["b"]
        # Exactly one side in today's issue. Both sides means a same-day pair,
        # which the loop above already handled with its shared references named.
        # (The row's own `date` is the later of the two, so filtering on it
        # would have dropped every backward-looking pair instead — which is
        # what it did, silently, until the five-day check showed no archive
        # clusters on any day that had them.)
        if (a in todays) != (b in todays):
            here, there = (a, b) if a in todays else (b, a)
            other = store.load_item(there)
            # Backwards only. The coupling window is symmetric, so without this
            # the 2026-08-07 issue cites a paper published on 2026-08-11 — a
            # true statement about the archive and an impossible one for an
            # issue that went out four days earlier.
            if not other or not other.first_published or other.first_published >= d:
                continue
            shared_refs = [r for r in refs.get(here, []) if r in set(refs.get(there, []))]
            archive.append({
                "scope": "archive",
                "work_keys": [here, there],
                "titles": [
                    titles.get(here, here),
                    clean_title(other.bibliography.title) if other else there,
                ],
                "shared": pair["shared"],
                "shared_titles": [
                    clean_title(resolved[r]["title"])
                    for r in shared_refs
                    if r in resolved and resolved[r].get("title")
                ][:3],
                "partner_date": str(other.first_published) if other and other.first_published else None,
            })
    archive.sort(key=lambda r: -r["shared"])
    return (out + archive)[:limit]


def first_internal_citation(d: date, items: list[Item]) -> Optional[dict]:
    """The day the archive first cites itself — a one-time event, marked as one."""
    from .graph.citation import internal_citation_edges

    edges = internal_citation_edges()
    if not edges:
        return None
    todays = {it.work_key for it in items}
    today_edges = [(a, b) for a, b in edges if a in todays]
    if not today_edges:
        return None

    # "First" means no earlier *published day* produced one, so the test is on
    # the citing item's publication date, not on whether it is in today's list.
    earlier = False
    for citing, _cited in edges:
        if citing in todays:
            continue
        item = store.load_item(citing)
        if item and item.first_published and item.first_published < d:
            earlier = True
            break
    return {"edges": today_edges[:3], "is_first": not earlier}


# --------------------------------------------------------------------------
# 4. Institutions and authors
# --------------------------------------------------------------------------


def _institutions_of(item: Item) -> set[str]:
    names = {org.label for org in item.entities.orgs}
    for author in item.bibliography.authors:
        for inst in author.institutions:
            if inst.name:
                names.add(inst.name)
    return names


def affiliations(
    d: date, items: list[Item], limit: int = 4, window_days: int = BASELINE_DAYS
) -> dict[str, Any]:
    """Who appears more than once. Frequencies, never a ranking.

    **Within a day this is almost always empty, and that is a property of the
    data rather than a bug.** Measured over the five prepared days, no
    institution appears in two of a day's 24 papers — the field is global and 24
    papers is a thin slice of it. So the same question is also asked over the
    window, where repetition does happen and means something.

    No score, no ordering claim beyond the count, and nothing carried forward as
    "top" anything. An institution appearing three times in a month is a fact;
    an institution being *important* is a judgement this service does not make,
    and the moment it publishes a ranking it has become an evaluation body.
    """
    today: Counter = Counter()
    for item in items:
        for name in _institutions_of(item):
            today[name] += 1

    start = d - timedelta(days=window_days)
    window: Counter = Counter()
    for item in store.iter_items():
        pub = item.first_published
        if not pub or not (start <= pub <= d):
            continue
        for name in _institutions_of(item):
            window[name] += 1

    return {
        "today": [
            {"name": n, "papers": c} for n, c in today.most_common(limit) if c >= 2
        ],
        "in_window": [
            {"name": n, "papers": c} for n, c in window.most_common(limit) if c >= 3
        ],
        "window_days": window_days,
        "distinct_institutions_today": len(today),
    }


def repeat_authors(items: list[Item], limit: int = 3) -> list[dict]:
    counts: Counter = Counter()
    for item in items:
        for author in {a.name for a in item.bibliography.authors if a.name}:
            counts[author] += 1
    return [
        {"name": n, "papers_today": c} for n, c in counts.most_common(limit) if c >= 2
    ]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


# How many of today's papers must share a controlled-vocabulary tag before that
# tag counts as a grouping worth naming.
#
# Measured over the 63-issue archive (0S, U2):
#
#   condition                          days speaking   of 63
#   OLD: 3 facts incl. >=1 link                   57     90%
#   a tag on >= 2 of the day's items              63    100%
#   a tag on >= 3 of the day's items              54     86%
#   a tag on >= 4 of the day's items              30     48%
#   >= 2 tags on >= 3 items                       34     54%
#
# **Two is not a condition**: it fires on every day in the archive, and a gate
# that never closes is not a gate. With fifteen items and dozens of tags, some
# pair shares something by arithmetic. **Four silences 31 days** the old rule
# let speak, including days that plainly do have a shared subject.
#
# Three is the smallest count that reads as "several" rather than "a pair", and
# it lands slightly *stricter* than the condition it replaces — 86% against 90%.
GROUP_MIN_PAPERS = 3

# How many groups the paragraph is offered. Measured: the mean day has 1.87
# groups at this threshold and the richest has 8.
#
# **A paragraph that names eight groups is a catalogue, not a paragraph.**
# 2026-06-24 has seven, and the first draft of that day read as a list of tags
# with papers hung off each — every sentence true, and nothing a reader could
# hold. The biggest groups are the ones worth naming; the rest are still in the
# issue, on the cards, where they belong.
GROUP_LIMIT = 4

# How many ungrouped papers get a clause of their own.
HIGHLIGHT_LIMIT = 3

# ★ The floor under a paragraph written **without** a measured group (0Z-F, S1).
#
# From 2026-08-21 the paragraph stopped appearing, and nothing was broken: no
# controlled-vocabulary tag was shared by three papers on any of those days, and
# `write_paragraph` treated that as "nothing to say". It is not. It is "nothing
# to say *about grouping*", and the 2026-08-18 paragraph shows the difference —
# it opens with a measured claim ("4 of today's papers carry the tag …") and
# then does something else entirely: *"Outside this group, another paper
# introduces … while another paper utilizes …"*. That second half needs no group
# and invents nothing. It says which paper did what.
#
# So a day with no group can still have a paragraph, and the condition is not
# how much grouped but **whether the paragraph selects**. `highlights` names at
# most `HIGHLIGHT_LIMIT` papers; a day with that many papers or fewer would get
# a paragraph that names all of them, which is the issue read aloud rather than
# a summary of it. Hence one more than the limit, derived rather than typed so
# the two cannot drift apart.
#
# **The archive cannot choose this number and is not being asked to.** Over the
# 73 published issues the item counts are 1, 2, 2, 6, 7, 9, … — there is no day
# with 3, 4 or 5 papers, so every floor from 3 to 6 behaves identically on
# everything ever published. What the archive does say is which days are
# affected: 16 issues have no group, 13 of them have 6 or more papers and gain a
# paragraph, and 3 (2026-08-20, 08-23, 08-24 — 2, 1 and 2 papers) stay silent.
PARAGRAPH_MIN_ITEMS = HIGHLIGHT_LIMIT + 1


def _all_groups(items: list[Item]) -> list[dict]:
    """Every group, uncapped — what `highlights` must measure "ungrouped" against."""
    return tag_groups(items, limit=None)


def tag_groups(
    items: list[Item],
    min_papers: int = GROUP_MIN_PAPERS,
    limit: Optional[int] = GROUP_LIMIT,
) -> list[dict]:
    """Controlled-vocabulary tags that at least `min_papers` of today share.

    **This is the thing V3 deliberately had no place for**, and the reason it
    is safe now is that the group's name is not invented. 0i removed the slot
    for a theme because asking a model "what is today about" with nothing to
    read produces fiction. The name here is **the tag itself** — a string that
    matched the controlled vocabulary — so naming the group states a fact rather
    than proposing one.

    The papers in each group travel with it, because the paragraph is asked to
    name them and it may only name what it was given.
    """
    by_tag: dict[str, list[Item]] = {}
    for item in items:
        for tag in sorted(set(_tags_of(item))):
            by_tag.setdefault(tag, []).append(item)

    groups = []
    for tag, members in by_tag.items():
        if len(members) < min_papers:
            continue
        groups.append({
            "tag": tag,
            "papers": len(members),
            "work_keys": [m.work_key for m in members],
            "titles": [clean_title(m.bibliography.title) for m in members],
            "whats": [
                (m.summary.en.what if m.summary.en else "") or "" for m in members
            ],
        })
    # Biggest first, then alphabetically so the order is stable across runs.
    groups.sort(key=lambda g: (-g["papers"], g["tag"]))
    return groups[:limit] if limit else groups


def highlights(
    items: list[Item], groups: list[dict], limit: int = HIGHLIGHT_LIMIT,
    all_groups: Optional[list[dict]] = None,
) -> list[dict]:
    """The best-scoring papers that no group already covers.

    Without this the paragraph would describe only what clustered, and a day's
    single strongest paper could go unmentioned because nothing else resembled
    it. Ranked by the same headline score the issue already uses, so the
    paragraph and the card order do not disagree about what stood out.
    """
    # Against **every** group, not only the ones the paragraph was shown. A
    # paper in a group that fell outside `GROUP_LIMIT` is still a paper that
    # grouped, and offering it as "in no group" would be false.
    grouped = {k for g in (all_groups or groups) for k in g["work_keys"]}
    loose = [it for it in items if it.work_key not in grouped]
    loose.sort(key=lambda it: (-it.scores.headline, it.work_key))
    out = []
    for item in loose[:limit]:
        what = (item.summary.en.what if item.summary.en else "") or ""
        if not what:
            continue
        out.append({
            "work_key": item.work_key,
            "title": clean_title(item.bibliography.title),
            "what": what,
            "tags": sorted(set(_tags_of(item)))[:4],
        })
    return out


def build_facts(
    d: date, items: list[Item], unreadable_n: int = 0
) -> dict[str, Any]:
    """Everything measurable about how today connects to before. No LLM."""
    facts = {
        "date": str(d),
        "composition": composition(items, unreadable_n),
        "deviations": deviations(d, items),
        "anchors": canon_anchors(d, items),
        "clusters": clusters(d, items),
        "affiliations": affiliations(d, items),
        "repeat_authors": repeat_authors(items),
        "first_internal_citation": first_internal_citation(d, items),
    }
    # New material for the paragraph (0S, U2). The citation facts above are not
    # removed and not demoted — they keep every label row they had. What
    # changed is which of them the *paragraph* is written from.
    facts["tag_groups"] = tag_groups(items)
    facts["highlights"] = highlights(
        items, facts["tag_groups"], all_groups=_all_groups(items)
    )
    facts["material_count"] = (
        len(facts["deviations"]["found"])
        + len(facts["anchors"])
        + len(facts["clusters"])
        + len(facts["affiliations"]["today"])
        + len(facts["affiliations"]["in_window"])
        + len(facts["repeat_authors"])
        + (1 if facts["first_internal_citation"] else 0)
    )
    return facts


# --------------------------------------------------------------------------
# 5. The paragraph
# --------------------------------------------------------------------------

PROMPT_PATH = Path(__file__).parent / "prompts" / "synthesis.md"

# What counts as enough to write from.
#
# **The condition changed in 0S; the principle did not.** The old bar was three
# connective facts including at least one measured citation link, and it was the
# right bar for a paragraph made of citation links. The paragraph is now made of
# the day's own items, so the question "was there a measured link" is no longer
# the question being asked.
#
# What replaces it is the same idea in the new material's terms: **if nothing
# groups, there is no paragraph.** A day where no controlled-vocabulary tag is
# shared by three papers is a day whose items do not resemble each other, and
# the only paragraph available would be "today's seven papers are not much
# alike" — which is precisely the filler sentence mockup 6a wrote and this
# project refused. 0i's 08-05 silence was the test of that design and it stays
# a test; only the trigger moves.
#
# Kept for the label rows and for `material_count`, which still measure the
# citation facts.
PARAGRAPH_MIN_MATERIAL = 3

REFUSAL = "NOTHING TO SAY"


def render_facts(facts: dict[str, Any]) -> str:
    """The FACTS block the model may write from, and nothing else.

    ## What changed in 0S (U2), and what did not

    This used to render the **citation graph**: tag deviations, canon anchors,
    shared-reference clusters, repeated institutions and authors. The paragraph
    written from it said who cited whom, which is what YJUN read and asked to
    change — the cards below already say what each paper did, and the paragraph
    was spending itself on relationships rather than on **what arrived today**.

    So the material is now the day's own items: the controlled-vocabulary tags
    at least three of them share, and the best-scoring papers no group covers.

    **The citation facts have not been deleted and have not been demoted.** They
    keep every label row they had — `tag shift`, `canon`, `coupling`,
    `institutions`, `authors` — and those rows are the measured record. What
    moved is narrower than it sounds: **they went from being the prose to being
    the instrument panel.** A reader who wants the citation structure reads the
    rows, which state it exactly; a reader who wants to know what showed up
    reads the paragraph, which no longer pretends the two are the same question.

    Rendered as flat statements rather than JSON: a model asked to prosify JSON
    narrates the schema, and a model given sentences joins them.
    """
    # Composition is deliberately absent. It is printed directly above the
    # paragraph in the issue, and when it was included the model opened with it
    # every time — a sentence that costs a line and adds nothing.
    lines: list[str] = []

    for group in facts.get("tag_groups") or []:
        lines.append(
            f"- {group['papers']} of today's papers carry the tag "
            f"\"{group['tag']}\"."
        )
        for title, what in zip(group["titles"], group["whats"]):
            first = _first_sentence(what)
            lines.append(f"    - \"{title}\"" + (f": {first}." if first else "."))

    # ★ On a day with no measured group the ungrouped lines lose two things
    # (0Z-F, S1), and the second was found by running it.
    #
    # **"is not in any of the groups above"** is only true when there are groups
    # above. On a day with none it points at nothing, and pointing a model at an
    # absent group is how an absent group acquires a name.
    #
    # **The tag list goes too, and that is the part that mattered.** The first
    # generation for 2026-08-03 opened *"Two of today's papers carry the tag
    # 'Transportation Planning and Optimization'"* — a group that does not exist
    # and was never in the facts block, because two is below `GROUP_MIN_PAPERS`.
    # The model had not invented the tag: it read the same string in two of the
    # `(its tags: ...)` clauses and did the arithmetic itself. Leaving the
    # material there and asking the prompt not to use it is the arrangement 0i
    # rejected — *"we did not write in the prompt that it must not name a topic,
    # we removed the place in the output where a topic could go"*. So on a day
    # with no group there is no shared string to count, and the claim is
    # unavailable rather than merely forbidden. The regex in `write_paragraph`
    # is what is left after that, not instead of it.
    grouped = bool(facts.get("tag_groups"))
    for hl in facts.get("highlights") or []:
        first = _first_sentence(hl["what"])
        if grouped:
            tags = ", ".join(f'"{t}"' for t in hl["tags"])
            head = f'- "{hl["title"]}" is not in any of the groups above'
            head += f" (its tags: {tags})" if tags else ""
        else:
            head = f'- "{hl["title"]}"'
        lines.append(head + (f". {first}." if first else "."))

    return "\n".join(lines)


# Two shapes of claim about the day **as a set**, refused on a day where no set
# was measured. See `write_paragraph` for why these two and not a general
# detector.
#
# The first is a count: "4 of today's papers …". On an ungrouped day no such
# count is in the FACTS block, so any sentence of this shape was invented — the
# claim-about-an-unmeasured-population that 0Z-B recorded as D273.
#
# The second is subtler and the model produced it on the first try. Asked for a
# paragraph about 2026-08-28's nine unrelated papers it opened *"This issue
# features research across several distinct areas of urban data science."* —
# grammatical, harmless-looking, and an assertion about the spread of the day
# that nobody measured. It is the negative of the forbidden claim rather than
# the claim, which is why it slipped past both the prompt's "no opening
# flourish" line and the count pattern. On a day with no measured grouping
# there is **no true sentence whose subject is the issue**, so the subject
# itself is the thing to refuse.
_GROUP_CLAIM = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|several|many|"
    r"most|both|half)\s+of\s+(today'?s|the\s+day'?s)\s+papers\b"
    r"|\b(this|today'?s)\s+(issue|digest|research\s+digest|collection|selection|"
    r"day'?s\s+papers)\b",
    re.IGNORECASE,
)


def material_for_paragraph(facts: dict[str, Any]) -> int:
    """Connective facts only. Composition is not one of them."""
    return (
        len(facts["deviations"]["found"])
        + len(facts["anchors"])
        + len(facts["clusters"])
        + len(facts["affiliations"]["today"])
        + len(facts["affiliations"]["in_window"])
        + len(facts["repeat_authors"])
        + (1 if facts.get("first_internal_citation") else 0)
    )


def write_paragraph(facts: dict[str, Any], client=None) -> dict[str, Any]:
    """One paragraph from the measured facts, or an honest absence.

    Two ways to get nothing, and both are correct outcomes rather than failures:
    the material is too thin to reach the bar, or the model itself judges the
    facts too thin and returns the refusal token. Neither is retried and neither
    is padded — a day with no measured connections is a day with nothing to say
    about connections, and the quiet-day rule already established that saying so
    is better than filling.
    """
    material = material_for_paragraph(facts)
    groups = facts.get("tag_groups") or []
    highlights_ = facts.get("highlights") or []
    published = int((facts.get("composition") or {}).get("published") or 0)

    # ★ No group is no longer the end of the question (0Z-F, S1).
    #
    # It used to be: no group, no paragraph, and from 2026-08-21 that silenced
    # every issue. The gate was right about what it was protecting — a day whose
    # papers do not resemble each other must not be given a sentence saying they
    # do — and wrong about the only remedy being silence. The ungrouped
    # paragraph makes no claim about resemblance; it says which paper did what,
    # which is measured for each paper separately.
    #
    # Two conditions replace it, and both are about whether the paragraph is
    # doing work the cards do not already do:
    if not groups:
        if published < PARAGRAPH_MIN_ITEMS:
            return {
                "text": None,
                "omitted": True,
                "reason": (
                    f"no controlled-vocabulary tag is shared by "
                    f"{GROUP_MIN_PAPERS} or more of today's papers, and "
                    f"{published} paper(s) is too few to summarise separately"
                ),
                "material": material,
                "groups": 0,
            }
        if len(highlights_) < 2:
            return {
                "text": None,
                "omitted": True,
                "reason": (
                    "no controlled-vocabulary tag is shared by "
                    f"{GROUP_MIN_PAPERS} or more of today's papers, and fewer "
                    "than two of them have a summary to compress"
                ),
                "material": material,
                "groups": 0,
            }

    from .llm import LLMBudgetExceeded, LLMClient

    client = client or LLMClient(task="synthesis")
    if not client.available():
        return {
            "text": None,
            "omitted": True,
            "reason": "no usable LLM credentials",
            "material": material,
        }

    system = PROMPT_PATH.read_text(encoding="utf-8")
    user = f"FACTS for {facts['date']}:\n\n{render_facts(facts)}"
    try:
        # **The key covers the input, not just the date.** It was
        # `synthesis-<date>` alone, so a change to what the FACTS block
        # contains — exactly what 0S is — would have been served the old
        # answer with no sign that anything was stale. That is the third time
        # in three batches a cache key has been narrower than the thing it
        # keys: 0Q's extract comparison found the model missing from the key,
        # and 0R's headline retry found the hint missing from it.
        #
        # Hashing the rendered facts means the next change to the material
        # invalidates itself and nobody has to remember to bump anything.
        digest = hashlib.sha1(user.encode("utf-8")).hexdigest()[:12]
        resp = client.complete(
            system, user, cache_key=f"synthesis-{facts['date']}-{digest}"
        )
    except LLMBudgetExceeded as e:
        return {"text": None, "omitted": True, "reason": str(e), "material": material}

    text = (resp.text or "").strip()
    if not text or REFUSAL in text.upper():
        return {
            "text": None,
            "omitted": True,
            "reason": "the model judged the facts too thin",
            "material": material,
        }

    # ★ The last defence on an ungrouped day, and deliberately a narrow one
    # (0Z-F, S1). `_GROUP_CLAIM` above says which two shapes and why.
    #
    # **It does not try to detect invention in general.** A regex that claimed
    # to would be worse than none: it would read as a guarantee and catch one
    # phrasing out of twenty. What it does is refuse the two claims whose
    # subject is known to be unmeasured, and record why, so a day that trips it
    # is visible rather than published. Everything else rests on the FACTS
    # block having nothing to invent from, which is the older and better
    # defence — 0i's *"we removed the place in the output where a topic could
    # go"*.
    if not groups and _GROUP_CLAIM.search(text):
        return {
            "text": None,
            "omitted": True,
            "reason": (
                "the paragraph made a claim about today's papers as a set and "
                "no grouping among them was measured"
            ),
            "material": material,
            "groups": 0,
        }

    return {
        "text": text,
        "omitted": False,
        "reason": None,
        "material": material,
        "groups": len(groups),
    }


def build(
    d: date, items: list[Item], unreadable_n: int = 0, client=None, use_llm: bool = True
) -> "Any":
    """The whole layer, as the model the Issue carries.

    Measured first, written last. If the LLM is unavailable or the day has no
    measured link, everything except the paragraph still stands — the sections
    are the substance and the paragraph is the reading of them.
    """
    from .models import (
        Synthesis,
        SynthesisAnchor,
        SynthesisCluster,
        SynthesisDeviation,
        SynthesisName,
    )

    facts = build_facts(d, items, unreadable_n)
    para = (
        write_paragraph(facts, client=client)
        if use_llm
        else {"text": None, "omitted": True, "reason": "use_llm=False"}
    )
    fic = facts.get("first_internal_citation") or {}
    return Synthesis(
        composition=facts["composition"],
        deviations=[SynthesisDeviation(**dev) for dev in facts["deviations"]["found"]],
        deviation_status=facts["deviations"]["status"],
        deviation_note=facts["deviations"].get("note"),
        anchors=[SynthesisAnchor(**a) for a in facts["anchors"]],
        clusters=[SynthesisCluster(**c) for c in facts["clusters"]],
        institutions_today=[
            SynthesisName(**a) for a in facts["affiliations"]["today"]
        ],
        institutions_in_window=[
            SynthesisName(**a) for a in facts["affiliations"]["in_window"]
        ],
        repeat_authors=[
            SynthesisName(name=a["name"], papers=a["papers_today"])
            for a in facts["repeat_authors"]
        ],
        window_days=facts["affiliations"]["window_days"],
        first_internal_citation=bool(fic.get("is_first")),
        paragraph=para["text"],
        paragraph_omitted_reason=para["reason"] if para["omitted"] else None,
    )
