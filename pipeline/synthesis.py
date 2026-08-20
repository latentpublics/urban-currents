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
    """
    today = Counter(t for it in items for t in _tags_of(it))

    start = d - timedelta(days=window_days)
    history: dict[str, Counter] = defaultdict(Counter)
    days_seen: set[str] = set()
    for item in store.iter_items():
        pub = item.first_published
        if not pub or not (start <= pub < d):
            continue
        days_seen.add(str(pub))
        for tag in set(_tags_of(item)):
            history[tag][str(pub)] += 1

    def scan(n: int) -> list[dict]:
        out = []
        for tag, count in today.most_common():
            if count < DEVIATION_MIN_TODAY:
                continue
            baseline = sum(history.get(tag, Counter()).values()) / max(1, n)
            if count < max(DEVIATION_MIN_RATIO * baseline, DEVIATION_MIN_TODAY):
                continue
            out.append({
                "label": tag,
                "today": count,
                "baseline_per_day": round(baseline, 2),
                "window_days": n,
            })
        return out[:5]

    n_days = len(days_seen)
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

    found = scan(n_days)
    return {
        "found": found[:5],
        "baseline_days": n_days,
        "status": "OK",
        "distinct_tags_today": len(today),
    }


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
            first = (what or "").split(". ")[0].strip()
            lines.append(f"    - \"{title}\"" + (f": {first}." if first else "."))

    for hl in facts.get("highlights") or []:
        first = (hl["what"] or "").split(". ")[0].strip()
        tags = ", ".join(f'"{t}"' for t in hl["tags"])
        lines.append(
            f"- \"{hl['title']}\" is not in any of the groups above"
            + (f" (its tags: {tags})" if tags else "")
            + (f". {first}." if first else ".")
        )

    return "\n".join(lines)


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
    if not groups:
        return {
            "text": None,
            "omitted": True,
            "reason": (
                f"no controlled-vocabulary tag is shared by "
                f"{GROUP_MIN_PAPERS} or more of today's papers"
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
