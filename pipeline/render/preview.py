"""Preview renderer (PRD §5.7).

Python here does data shaping only; every piece of markup and copy lives in
``templates/*.html.j2``. Phase 1's Astro components inherit that DOM and those
class names, so the card layout decision is made once, not twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

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
        # A text label, never an emoji: emoji render inconsistently in mail
        # clients and a screen reader says "counterclockwise arrows button".
        "lens": item.lens,
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


def render_issue(
    issue: Issue, items: Iterable[Item], unreadable: Iterable[Item] = ()
) -> str:
    by_key = {it.work_key: it for it in items}
    ordered = [by_key[k] for k in issue.items if k in by_key]
    # Headline first, then descending headline score.
    ordered.sort(
        key=lambda it: (it.work_key != (issue.headline.work_key or ""), -it.scores.headline)
    )
    env = _env()
    css = (TEMPLATE_DIR / "base.css.j2").read_text(encoding="utf-8")
    return env.get_template("preview.html.j2").render(
        issue=issue,
        scan_meta=issue.scan_meta,
        cards=[build_card(it) for it in ordered],
        synthesis=build_synthesis(issue, ordered),
        unreadable=[build_unreadable_row(it) for it in unreadable],
        status_changes=[
            {"work_key": c.work_key, "text": _status_change_text(c)}
            for c in issue.status_changes
        ],
        # Trusted local stylesheet: it must not be HTML-escaped into the <style>.
        css=Markup(css),
        pipeline_version=PIPELINE_VERSION,
    )


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

