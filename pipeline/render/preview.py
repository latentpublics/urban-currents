"""Preview renderer (PRD §5.7).

Python here does data shaping only; every piece of markup and copy lives in
``templates/*.html.j2``. Phase 1's Astro components inherit that DOM and those
class names, so the card layout decision is made once, not twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .. import paths
from ..config import cfg
from ..models import PIPELINE_VERSION, Issue, Item

TEMPLATE_DIR = Path(__file__).parent / "templates"

FACET_ORDER = ["topics", "methods", "data", "tools", "places", "orgs"]


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _authors(item: Item) -> str:
    authors = item.bibliography.authors
    names = [a.name for a in authors[:3]]
    if len(authors) > 3:
        names.append("et al.")
    return ", ".join(names)


def _venue(item: Item) -> str:
    """Where it appeared, plus the arXiv category when it is an arXiv record."""
    loc = item.bibliography.primary_location
    name = loc.source_name or ""
    cats = item.bibliography.categories
    if name.lower() == "arxiv" and cats:
        return f"arXiv {cats[0]}"
    return name


def _was_preprint(item: Item) -> Optional[str]:
    """"was arXiv" — the line where Item/Issue separation first becomes visible.

    An item that carries an arXiv identifier and is now published somewhere else
    has a history, and this is the only place a reader sees it. Three such items
    exist in the archive today, counted rather than assumed.

    The category is not part of it. All three have an empty `categories` list —
    the arXiv record was merged into the journal one and the category did not
    survive the merge — so the line says "was arXiv" and stops. Writing "was
    arXiv cs.AI" would mean inventing the category.
    """
    if item.publication_status.state != "published":
        return None
    if not item.ids.arxiv:
        return None
    cats = item.bibliography.categories
    return f"was arXiv {cats[0]}" if cats else "was arXiv"


def _badges(item: Item) -> list[dict]:
    """Blue for an artifact that exists, grey for a state.

    `published` carries the journal name because "published" alone tells the
    reader nothing they cannot see from the byline, and where it landed is the
    fact that changed.
    """
    out: list[dict] = []
    for badge in item.badges:
        if badge == "published":
            journal = item.publication_status.journal or ""
            out.append({
                "kind": "artifact",
                "label": f"published: {journal}" if journal else "published",
            })
        elif badge == "preprint":
            out.append({"kind": "state", "label": "preprint"})
        else:
            out.append({"kind": "artifact", "label": badge})
    return out


def _links(item: Item) -> list[dict]:
    links: list[dict] = []
    loc = item.bibliography.primary_location
    if loc.landing_page_url:
        links.append({"label": loc.source_name or "source", "url": loc.landing_page_url})
    if loc.pdf_url:
        links.append({"label": "pdf", "url": loc.pdf_url})
    doi = item.ids.doi
    if doi and not doi.lower().startswith("10.48550"):
        links.append({"label": "doi", "url": f"https://doi.org/{doi}"})
    code = item.signals.code_available
    if code and code.value is True and code.url:
        links.append({"label": "code", "url": code.url})
    return links


def _facet_tags(item: Item, limit: int = 6) -> list[dict]:
    """Facet tags in `FACET_ORDER`, flattened, empty facets simply absent.

    The mockup's own prose lists the facets in one order and its markup draws
    them in another; neither matches the code. `FACET_ORDER` wins, for two
    reasons: the first thing a reader asks is what a paper is *about*, which is
    topics, and Places was explicitly taken off the signature list.
    """
    e = item.entities
    buckets = {
        "topics": e.topics,
        "methods": e.methods,
        "data": e.data,
        "tools": e.tools,
        "places": e.places,
        "orgs": e.orgs,
    }
    out: list[dict] = []
    for name in FACET_ORDER:
        for tag in buckets.get(name) or []:
            out.append({"id": tag.id, "label": tag.label, "facet": name})
    return out[:limit]


def _affiliation(item: Item) -> str:
    """The first author's first institution, or "".

    Same source `build_unreadable_row` already reads, so the two rows of the
    issue cannot disagree about where a paper came from.

    **Returns "" rather than a placeholder when there is none.** An empty line
    in the meta rail reads as a missing value — as though we knew the
    affiliation and lost it — when the truth is that OpenAlex did not give one.
    The template omits the line entirely instead.
    """
    authors = item.bibliography.authors or []
    if authors and authors[0].institutions:
        return authors[0].institutions[0].name or ""
    return ""


def build_card(item: Item) -> dict:
    en = item.summary.en
    return {
        "work_key": item.work_key,
        "anchor": item.work_key.replace(":", "-").replace("/", "-"),
        "title": item.bibliography.title,
        "authors": _authors(item),
        "venue": _venue(item),
        "was_preprint": _was_preprint(item),
        "landing_url": item.bibliography.primary_location.landing_page_url,
        "what": (en.what if en else "") or "",
        "why": (en.why if en else "") or "",
        "caveats": (en.caveats if en else None) or None,
        "badges": _badges(item),
        "affiliation": _affiliation(item),
        "facet_tags": _facet_tags(item),
        "links": _links(item),
    }


def build_unreadable_row(item: Item) -> dict:
    """One line of `Also published today` — facts only (P3).

    No LLM has touched this item and none will: it has no abstract, so there is
    nothing to summarise and any sentence about it would be invented. What can
    be stated is what the bibliography already says, plus the controlled-
    vocabulary terms that match its **title**.

    Those terms are computed here rather than written into `entities`. A tag
    inferred from a title alone is a display affordance, not evidence; storing
    it would feed the novelty term and the entity graph from a source we have
    said is not good enough to summarise from.
    """
    from ..linking.vocab_match import Vocabulary, scan_text

    authors = item.bibliography.authors
    names = [a.name for a in authors[:3]]
    if len(authors) > 3:
        names.append("et al.")

    # First author's institution: the one affiliation a reader uses to place a
    # paper at a glance. Absent for many records, and absent is fine.
    affiliation = None
    if authors and authors[0].institutions:
        affiliation = authors[0].institutions[0].name

    title = item.bibliography.title or ""
    tags: list[str] = []
    for facet in ("methods", "data", "tools"):
        try:
            vocab = Vocabulary.load(facet)
        except Exception:  # noqa: BLE001 - a missing vocab file is not fatal here
            continue
        for ref in scan_text(title, facet, vocab):
            if ref.label not in tags:
                tags.append(ref.label)

    doi = item.ids.doi
    return {
        "work_key": item.work_key,
        "title": title,
        "url": item.bibliography.primary_location.landing_page_url
        or (f"https://doi.org/{doi}" if doi else None),
        "authors": ", ".join(names),
        "affiliation": affiliation,
        "journal": item.bibliography.primary_location.source_name,
        "topics": [t.label for t in item.entities.topics[:3]],
        "title_terms": tags[:4],
    }


_REFERENCE_KEYS: Optional[set] = None


def _items_with_references(items: Iterable[Item]) -> int:
    """How many of a day's items have a reference list on file.

    The measurability question, and it has to be asked before a zero is
    printed. "No items share references today" is a measurement; "no items
    share references today" said about a day where nothing had a bibliography
    is a sentence about our coverage wearing the costume of a finding.
    """
    global _REFERENCE_KEYS
    if _REFERENCE_KEYS is None:
        from ..graph.citation import load_reference_base

        _REFERENCE_KEYS = {
            r["work_key"] for r in load_reference_base() if r.get("referenced_works")
        }
    return sum(1 for it in items if it.work_key in _REFERENCE_KEYS)


def build_synthesis(issue: Issue, items: Iterable[Item] = ()) -> dict | None:
    """The synthesis layer, shaped for the template.

    Flattened into strings here rather than in Jinja: the template's job is
    markup, and a sentence assembled across three template lines is a sentence
    nobody can review.

    **A row appears only when its measurement was possible, and then it may say
    zero.** The distinction runs through the whole section. The synthesis
    paragraph is forbidden from mentioning absence, and that rule stands — but
    it governs an LLM writing prose. These rows are instrument readings, and an
    instrument that reads zero is doing its job. The one thing neither may do is
    report a zero it could not have measured, so:

    - `tag shift` requires a 30-day baseline with at least seven days in it.
      Below that the row is absent, not zero.
    - `canon` and `coupling` require items that have reference lists at all.
      With none, the rows are absent; with some and nothing found, they read
      zero.
    - `institutions` and `authors` are always measurable — every item has a
      byline — so those rows may always read zero, and simply do not appear
      when there is nothing above the repeat threshold.
    """
    syn = issue.synthesis
    if syn is None:
        return None

    items = list(items)
    with_refs = _items_with_references(items) if items else 0

    anchors = []
    for a in syn.anchors:
        who = ", ".join(a.authors)
        cite = f"{who} ({a.year})" if who and a.year else (a.year or "")
        if a.first_in_window:
            note = "not cited by this archive before today"
        elif a.days_since_last_cited is not None:
            note = f"last cited {a.days_since_last_cited} days ago"
        else:
            note = None
        anchors.append({
            "title": a.title,
            "cite": cite,
            "count": a.citing_today,
            "note": note,
        })

    clusters = []
    for c in syn.clusters:
        clusters.append({
            "titles": c.titles,
            "shared": c.shared,
            "shared_titles": c.shared_titles,
            "partner_date": c.partner_date if c.scope == "archive" else None,
        })

    deviations = [
        {
            "label": d.label,
            "today": d.today,
            "baseline": d.baseline_per_day,
            "window_days": d.window_days,
        }
        for d in syn.deviations
    ]
    institutions = [
        {"name": i.name, "papers": i.papers, "scope": "today"}
        for i in syn.institutions_today
    ]
    in_window = [
        {"name": i.name, "papers": i.papers, "scope": "window"}
        for i in syn.institutions_in_window
    ]
    authors = [{"name": a.name, "papers": a.papers} for a in syn.repeat_authors]

    # label -> (measurable, value-or-None). The template renders a row only for
    # the measurable ones, and prints the zero sentence when the value is empty.
    rows = [
        {
            "label": "tag shift",
            "measurable": syn.deviation_status == "OK",
            "entries": deviations,
            "empty_text": "no tag ran above its 30-day average",
        },
        {
            "label": "canon",
            "measurable": with_refs > 0,
            "entries": anchors,
            "empty_text": "no foundational work is cited twice today",
        },
        {
            "label": "coupling",
            "measurable": with_refs >= 2,
            "entries": clusters,
            "empty_text": "no items share references today",
        },
        {
            "label": "institutions",
            "measurable": True,
            "entries": institutions + in_window,
            "empty_text": None,
        },
        {
            "label": "authors",
            "measurable": True,
            "entries": authors,
            "empty_text": None,
        },
    ]

    return {
        "composition": syn.composition,
        "rows": rows,
        "deviations": deviations,
        "deviation_note": syn.deviation_note,
        "anchors": anchors,
        "clusters": clusters,
        "institutions_today": institutions,
        "institutions_in_window": in_window,
        "repeat_authors": authors,
        "window_days": syn.window_days,
        "first_internal_citation": syn.first_internal_citation,
        "paragraph": syn.paragraph,
        "items_with_references": with_refs,
        # Rendered as an HTML comment, never as reader-facing text: the reason a
        # paragraph is absent is a fact about the pipeline, not about the field,
        # and the reader should simply see a shorter issue.
        "omitted_reason": syn.paragraph_omitted_reason,
        "has_content": any(
            r["measurable"] and (r["entries"] or r["empty_text"]) for r in rows
        ) or bool(syn.paragraph) or syn.first_internal_citation,
    }


def _status_change_text(change) -> str:
    journal = f" in {change.journal}" if change.journal else ""
    return f"{change.work_key}: {change.from_} → {change.to}{journal}"


def card_order(item: Item, headline_key: Optional[str]) -> tuple:
    """Headline, then published articles, then preprints — each by score.

    **This is a render decision, not an editorial one.** The issue published
    exactly what it published; only the order it is read in changes, so nothing
    in `content/` is rewritten and no issue's item list moves.

    The published/preprint split reads from `item.publication_status`,
    the primary fact, and **not from `badges`**. A badge is a display
    convenience derived from that fact, and sorting on the derived value would
    make the order depend on the rendering of the thing being rendered.

    Within each group the existing headline score still decides, so "most
    important first" is unchanged — it is now applied inside two groups instead
    of across one.
    """
    return (
        item.work_key != (headline_key or ""),
        0 if item.publication_status.state == "published" else 1,
        -item.scores.headline,
        item.work_key,
    )


def render_issue(
    issue: Issue, items: Iterable[Item], unreadable: Iterable[Item] = ()
) -> str:
    by_key = {it.work_key: it for it in items}
    ordered = [by_key[k] for k in issue.items if k in by_key]
    ordered.sort(key=lambda it: card_order(it, issue.headline.work_key))
    env = _env()
    css = (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8")
    return env.get_template("preview.html.j2").render(
        issue=issue,
        scan_meta=issue.scan_meta,
        cards=[build_card(it) for it in ordered],
        synthesis=build_synthesis(issue, ordered),
        still_cited=build_still_cited(issue, ordered),
        unreadable=[build_unreadable_row(it) for it in unreadable],
        status_changes=[
            {"work_key": c.work_key, "text": _status_change_text(c)}
            for c in issue.status_changes
        ],
        # Trusted local stylesheet: it must not be HTML-escaped into the <style>.
        css=Markup(css),
        pipeline_version=PIPELINE_VERSION,
    )


def email_subject(issue: Issue) -> str:
    """The subject line rule. Nothing is sent — there is no domain and no list.

    Date first because a daily arrives in a stack of its own back issues and the
    date is what a reader scans for. Then the count, then the headline's own
    opening clause, truncated at a word boundary — the only place in this
    codebase where a string is shortened in Python rather than in CSS, because a
    mail client truncates a subject at a byte count with no ellipsis and no
    tooltip.
    """
    if issue.quiet_day or not issue.items:
        return f"Urban Currents {issue.date} — a quiet day"
    line = (issue.headline.line or "").strip()
    head = line.split(". ")[0].rstrip(".")
    if len(head) > 60:
        head = head[:60].rsplit(" ", 1)[0] + "…"
    n = len(issue.items)
    return f"Urban Currents {issue.date} — {n} papers" + (f": {head}" if head else "")


def write_email(
    issue: Issue,
    items: Iterable[Item],
    out_path: Path,
    unreadable: Iterable[Item] = (),
) -> Path:
    """The email edition, derived from the same render — never a second template."""
    from .inline import to_email

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        to_email(render_issue(issue, items, unreadable)),
        encoding="utf-8",
        newline="\n",
    )
    return out_path


def write_preview(
    issue: Issue,
    items: Iterable[Item],
    out_path: Path,
    unreadable: Iterable[Item] = (),
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_issue(issue, items, unreadable), encoding="utf-8", newline="\n"
    )
    return out_path



def build_still_cited(issue: Issue, items: Iterable[Item] = ()) -> Optional[dict]:
    """The canon card: one foundational work, what today does with it (W5).

    **Off by default and the reason is a finding, not caution.** V4 compared our
    canon against 48,753 externally ranked works and the 174 we are missing from
    its top 200 are concentrated in physical activity and travel behaviour —
    Environmental correlates of walking and cycling, Discrete Choice Analysis,
    Urban transportation networks. A canon with a hole that shape, published
    daily as "what this field stands on", would put a transport paper forward
    every day as a pillar of urban data science. YJUN's judgement on the canon
    has not been made, and this renderer must not pre-empt it by shipping.

    So it is built, tested, and gated on `render.still_cited`, which is false.
    """
    if not bool(cfg("render.still_cited", False)):
        return None

    import json as _json

    path = paths.CONTENT / "canon" / "candidates.json"
    if not path.exists():
        return None
    doc = _json.loads(path.read_text(encoding="utf-8"))
    foundation = [c for c in (doc.get("candidates") or []) if c.get("class") == "foundation"]
    if not foundation:
        return None

    from ..graph.citation import load_reference_base

    refs = {
        r["work_key"]: set(r.get("referenced_works") or []) for r in load_reference_base()
    }
    today = [it.work_key for it in items]

    # The day's own most-cited foundational work, not the archive's favourite.
    # A card that shows the same paper every day is a banner, not a reading.
    counts = {
        c["openalex_id"]: sum(1 for k in today if c["openalex_id"] in refs.get(k, ()))
        for c in foundation
    }
    best = max(foundation, key=lambda c: (counts.get(c["openalex_id"], 0),
                                          c.get("archive_citations") or 0))
    cited_today = counts.get(best["openalex_id"], 0)
    if not cited_today:
        return None

    authors = (best.get("authors") or [])[:2]
    year = (best.get("publication_date") or "")[:4]
    return {
        "citation": f"{', '.join(authors)} ({year})" if authors and year else (year or ""),
        "title": best.get("title") or best["openalex_id"],
        "cited_today": cited_today,
        "archive_citations": best.get("archive_citations") or 0,
        "archive_total": len(refs),
    }
