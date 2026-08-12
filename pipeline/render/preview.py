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


def _byline(item: Item) -> str:
    authors = item.bibliography.authors
    names = [a.name for a in authors[:3]]
    if len(authors) > 3:
        names.append("et al.")
    parts = [", ".join(names)] if names else []
    loc = item.bibliography.primary_location.source_name
    if loc:
        parts.append(loc)
    if item.bibliography.publication_date:
        parts.append(str(item.bibliography.publication_date))
    return " · ".join(p for p in parts if p)


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


def _facets(item: Item) -> list[dict]:
    e = item.entities
    buckets = {
        "topics": [{"id": t.id, "label": t.label} for t in e.topics],
        "methods": [{"id": t.id, "label": t.label} for t in e.methods],
        "data": [{"id": t.id, "label": t.label} for t in e.data],
        "tools": [{"id": t.id, "label": t.label} for t in e.tools],
        "places": [{"id": t.id, "label": t.label} for t in e.places],
        "orgs": [{"id": t.id, "label": t.label} for t in e.orgs],
    }
    return [
        {"name": name, "tags": buckets[name][:8]}
        for name in FACET_ORDER
        if buckets[name]
    ]


def build_card(item: Item) -> dict:
    en = item.summary.en
    return {
        "work_key": item.work_key,
        "anchor": item.work_key.replace(":", "-").replace("/", "-"),
        "title": item.bibliography.title,
        "byline": _byline(item),
        "landing_url": item.bibliography.primary_location.landing_page_url,
        "what": (en.what if en else "") or "",
        "why": (en.why if en else "") or "",
        "caveats": (en.caveats if en else None) or None,
        "badges": list(item.badges),
        "facets": _facets(item),
        "links": _links(item),
    }


def _status_change_text(change) -> str:
    journal = f" in {change.journal}" if change.journal else ""
    return f"{change.work_key}: {change.from_} → {change.to}{journal}"


def render_issue(issue: Issue, items: Iterable[Item]) -> str:
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
        status_changes=[
            {"work_key": c.work_key, "text": _status_change_text(c)}
            for c in issue.status_changes
        ],
        # Trusted local stylesheet: it must not be HTML-escaped into the <style>.
        css=Markup(css),
        pipeline_version=PIPELINE_VERSION,
    )


def write_preview(issue: Issue, items: Iterable[Item], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_issue(issue, items), encoding="utf-8", newline="\n")
    return out_path


def preview_path(run_dir: Path, date: Optional[str] = None) -> Path:
    return run_dir / "preview.html"
