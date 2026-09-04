"""The JSON the site serves (Launch A, A1).

**Nothing here is new.** Every value is already in `content/`; this reshapes it
into stable addresses so a consumer does not have to know the repository layout,
and so the shape can be promised while the layout stays ours to change. That is
the whole justification for the endpoints existing at all — `content/` is
already public JSON on `raw.githubusercontent.com`, and what was missing was a
promise and a document, not data.

Three addresses, and the reasoning for each:

  `api/index.json`        The catalogue. One row per day with its state and its
                          counts. This is the one an aggregator actually needs:
                          without it, finding out what exists means guessing
                          dates. It is also the only place the four outcomes are
                          published as a machine-readable field.
  `api/latest.json`       The newest issue, whole. Same body as a dated issue,
                          at an address that does not change — the one thing a
                          scheduled fetch wants.
  `api/issues/{date}.json` One day, whole.

**Not published here, each for its own reason** — the long form is in the API
document, and the short form is:

  * **Abstracts.** Third-party expressive text. We do not own it and cannot
    relicense it, and an API is a redistribution.
  * **Authors, ORCIDs, affiliations.** A per-day machine-readable list of people
    is a different artefact from a citation, and not one we cleared. Every item
    carries a DOI or an arXiv id, so the authoritative record is one fetch away
    from whoever actually needs it.
  * **An issue number.** There isn't one. Inventing a sequence here would make
    this file the source of a fact that exists nowhere else (A1-1).
  * **A generation timestamp.** It would make the build non-idempotent for no
    reader's benefit; `latest_date` answers the question a timestamp is usually
    asked for.

`summary` **is** published and is deliberately outside the CC BY grant — see the
licence section of `api.html.j2` and D285.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .. import paths
from ..models import Issue, Item

SCHEMA_VERSION = "1"


def _state(row: dict) -> str:
    """The outcome, as one word, from the same derivation the screen uses.

    Read from the row rather than from `quiet_day` on disk. That stored flag
    answers "did anything clear the headline bar", and 0Z's Z1 found it set on
    days that published nine papers — an API field called `quiet` carrying that
    meaning would export the exact confusion the screen just stopped making.
    """
    if row.get("missing"):
        return "not_seen"
    if row["quiet"]:
        return "quiet"
    if row.get("unranked"):
        return "no_headline"
    return "published"


def _item_json(item: Item) -> dict[str, Any]:
    bib = item.bibliography
    loc = bib.primary_location
    summary = item.summary.en
    return {
        "work_key": item.work_key,
        "title": bib.title,
        "url": loc.landing_page_url,
        "source": loc.source_name,
        "ids": {
            k: v
            for k, v in (
                ("doi", item.ids.doi),
                ("arxiv", item.ids.arxiv),
                ("openalex", item.ids.openalex),
            )
            if v
        },
        "first_published": str(item.first_published) if item.first_published else None,
        "status": item.publication_status.state,
        "journal": item.publication_status.journal,
        "badges": list(item.badges),
        "topics": [{"id": t.id, "label": t.label} for t in item.entities.topics],
        "scores": {
            "relevance": round(item.scores.relevance, 4),
            "headline": round(item.scores.headline, 4),
        },
        # Ours in the sense that we caused it to be written, and **not offered
        # under the open licence** — it is a paraphrase of an abstract we do not
        # own, and we have not established that we can sublicense that. Said
        # here as well as in the document because this is where it is served.
        "summary": (
            {"what": summary.what, "why": summary.why, "caveats": summary.caveats}
            if summary
            else None
        ),
    }


def _issue_json(
    issue: Issue,
    row: dict,
    items: dict[str, Item],
    base: str,
    tag_shift: dict | None = None,
) -> dict[str, Any]:
    scan = issue.scan_meta
    keys = [k for k in issue.items if k in items]
    shift = tag_shift or {"status": "NO_BASELINE", "found": [], "baseline_days": 0}
    return {
        "schema_version": SCHEMA_VERSION,
        "date": str(issue.date),
        "url": f"{base}/issues/{issue.date}.html" if base else f"issues/{issue.date}.html",
        "state": _state(row),
        # Both facts, never one standing in for the other (0Y, Y1-2).
        "backfilled": bool(row.get("backfilled")),
        "withheld": row.get("withheld"),
        "recent": bool(row.get("recent")),
        "counts": {
            # ★ One definition, shared with the archive row, the home stat rail
            # and the issue page (1D). It counts what the issue published, not
            # what happens to be loadable — those differed only if a key had no
            # file, which is a broken archive rather than a smaller day.
            "published": issue.published_count,
            "candidates_scanned": scan.candidates_scanned,
            "arxiv_categories": scan.arxiv_categories,
            "journals": scan.journals,
            # Papers we could see existed and could not read. The one blind
            # spot this pipeline measures exactly, so it is published.
            "unreadable": scan.unreadable_count,
        },
        "headline": {
            # Derived, like `state`. `headline.present` on disk has meant
            # different things across phases; this one means "there is a line
            # below".
            "present": bool(issue.headline.line),
            "line": issue.headline.line,
            "work_key": issue.headline.work_key,
            "basis": issue.headline.basis,
        },
        # ★ `tag shift`, the same derived value the issue page shows (1B).
        # The API said nothing about it before, which was not a disagreement
        # but was not an answer either; now the page and the JSON come from one
        # call to `deviations_over_archive`. `status` is carried because an
        # absent measurement and a measured zero are different claims and this
        # project refuses to collapse them: `NO_BASELINE` means there was not
        # enough archive behind the day to compare against, and an empty
        # `found` under `OK` means the comparison ran and nothing stood out.
        "synthesis": {
            "tag_shift": {
                "status": shift["status"],
                "baseline_days": shift.get("baseline_days", 0),
                "found": [
                    {
                        "label": d["label"],
                        "today": d["today"],
                        "baseline_per_day": d["baseline_per_day"],
                        "window_days": d["window_days"],
                    }
                    for d in shift["found"]
                ],
            }
        },
        "items": [_item_json(items[k]) for k in keys],
    }


def _dump(target: Path, payload: Any) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def build_api(out_dir: Optional[Path] = None) -> list[Path]:
    """`api/index.json`, `api/latest.json`, and one file per issue."""
    from .site import _base_url, archive_rows, item_index, latest_issue, load_issues

    from .. import synthesis

    base = _base_url()
    issues = load_issues()
    items = item_index()
    rows = {r["date"]: r for r in archive_rows(issues, items)}
    # The same call the issue pages make, so the page and the JSON cannot
    # disagree about a day (1B).
    shifts = synthesis.deviations_over_archive(issues, items)
    target_dir = out_dir or (paths.ROOT / "site" / "api")

    written: list[Path] = []
    catalogue = []
    for issue in issues:
        row = rows.get(str(issue.date))
        if row is None:  # an issue with no row cannot happen; not a reason to crash
            continue
        body = _issue_json(issue, row, items, base, tag_shift=shifts.get(issue.date))
        written.append(_dump(target_dir / "issues" / f"{issue.date}.json", body))
        catalogue.append(
            {k: body[k] for k in ("date", "url", "state", "backfilled", "withheld", "recent")}
            | {"published": body["counts"]["published"], "headline": body["headline"]["line"]}
        )

    # ★ Days with no issue are in the catalogue too. A gap would say nothing
    # happened on a day the sources did not answer, which is the claim the whole
    # outcome model exists to avoid making — and a consumer iterating dates
    # would silently reinvent it.
    for date_str, row in rows.items():
        if row.get("missing"):
            catalogue.append({
                "date": date_str,
                "url": None,
                "state": "not_seen",
                "backfilled": False,
                "withheld": row.get("reason"),
                "recent": bool(row.get("recent")),
                "published": 0,
                "headline": None,
            })
    catalogue.sort(key=lambda d: d["date"], reverse=True)

    latest = latest_issue(issues)
    index = {
        "schema_version": SCHEMA_VERSION,
        "name": "Urban Currents",
        "description": "A daily scan of urban data science research.",
        "site": f"{base}/" if base else None,
        "documentation": f"{base}/api.html" if base else "api.html",
        "feed": f"{base}/feed.xml" if base else "feed.xml",
        "licence": {
            # Three buckets, because they are three different things and one
            # notice over all of them would be a claim we cannot support.
            "selection_and_metadata": "CC BY 4.0",
            "summaries": "no open licence offered — derived from third-party abstracts",
            "bibliographic_records": "from arXiv and OpenAlex under their own terms",
            "details": f"{base}/api.html" if base else "api.html",
        },
        "latest_date": str(latest.date) if latest else None,
        "days": len(catalogue),
        "issues": catalogue,
    }
    written.append(_dump(target_dir / "index.json", index))

    if latest is not None:
        row = rows.get(str(latest.date))
        if row is not None:
            written.append(
                _dump(target_dir / "latest.json", _issue_json(latest, row, items, base))
            )
    return written
