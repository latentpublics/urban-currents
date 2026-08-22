"""The plain-text edition (phase 0k, X5-5).

**Not the HTML with its tags removed.** That produces a wall of run-together
sentences with the navigation in the middle of it, and a reader on a text-only
client gets a worse issue rather than the same one. This renders from the same
`Issue` and `Item` objects the HTML comes from, so the two cannot disagree about
content while differing in form.

The identity that matters, and that the tests enforce: every card's title, what
and why appears in all three outputs — web, HTML mail, plain text.
"""

from __future__ import annotations

import textwrap
from typing import Iterable

from ..models import Issue, Item

WIDTH = 72


def _wrap(text: str, indent: str = "", bullet: str = "") -> str:
    """Wrap to the measure, with a bullet that appears once.

    `bullet` and `indent` are separate because passing a list marker as the
    indent repeats it on every continuation line, which turns one long title
    into three bullets that read as three references.

    **Two of textwrap's defaults are wrong for this text and both were on.**

    `break_on_hyphens` split "physics-informed" across two lines, so the plain
    text carried a string the HTML did not. Standard typesetting for prose,
    wrong for a digest whose compound terms *are* the content — and it was
    caught by checking a real issue rather than the short synthetic titles the
    first test used: 7 of 216 strings failed to appear in all three outputs.

    `break_long_words` is the more serious one: it breaks a long DOI URL in the
    middle, and **a link a reader cannot copy is worse than a line that runs
    past the margin.** Turning it off means a long URL overflows the measure,
    which is the right trade — the line is still one token, and the reader can
    still click it.
    """
    first = f"{indent}{bullet}"
    rest = indent + " " * len(bullet)
    return textwrap.fill(
        " ".join((text or "").split()),
        width=WIDTH,
        initial_indent=first,
        subsequent_indent=rest,
        break_on_hyphens=False,
        break_long_words=False,
    )


def _rule(char: str = "-") -> str:
    return char * WIDTH


def render_text(
    issue: Issue, items: Iterable[Item], unreadable: Iterable[Item] = ()
) -> str:
    from .preview import build_card, build_synthesis, build_unreadable_row

    by_key = {it.work_key: it for it in items}
    ordered = [by_key[k] for k in issue.items if k in by_key]
    ordered.sort(
        key=lambda it: (it.work_key != (issue.headline.work_key or ""), -it.scores.headline)
    )

    out: list[str] = []
    out.append("URBAN CURRENTS")
    out.append(str(issue.date))
    if issue.covers_from and issue.covers_to:
        out.append(f"covering work published {issue.covers_from} to {issue.covers_to}")
    out.append(_rule("="))
    out.append("")

    scan = issue.scan_meta
    out.append(_wrap(
        f"{scan.arxiv_categories} arXiv categories, {scan.journals} journals, "
        f"{scan.candidates_scanned} candidates — {scan.items_published} worth "
        f"your time"
        + (f", {scan.unreadable_count} without an open abstract"
           if scan.unreadable_count else "")
    ))
    out.append("")

    # Derived, not read from the file (0Z, Z1): `quiet_day` on disk means "no
    # item cleared the headline bar", and printing "a quiet day" over nine
    # papers is what that confusion looked like to a reader.
    if issue.is_quiet:
        out.append(_wrap("A quiet day in urban data science."))
    elif issue.headline.line:
        out.append(_wrap(issue.headline.line))
    else:
        out.append(_wrap(
            "No single paper stood out today — nothing cleared the headline "
            "bar, so the day is below without one."
        ))
    out.append("")

    synthesis = build_synthesis(issue, ordered)
    if synthesis and synthesis["has_content"]:
        out.append("TODAY, TOGETHER")
        out.append(_rule())
        for row in synthesis["rows"]:
            if not row["measurable"] or not (row["entries"] or row["empty_text"]):
                continue
            if not row["entries"]:
                out.append(_wrap(f"{row['label']}: {row['empty_text']}"))
                continue
            if row["label"] == "coupling":
                for c in row["entries"]:
                    out.append(_wrap(
                        f"coupling: \"{c['titles'][0]}\" shares {c['shared']} "
                        f"references with \"{c['titles'][1]}\""
                        + (f" ({c['partner_date']})" if c["partner_date"] else "")
                    ))
                    for shared in c["shared_titles"]:
                        out.append(_wrap(shared, indent="    ", bullet="- "))
            elif row["label"] == "canon":
                for a in row["entries"]:
                    out.append(_wrap(
                        f"canon: {a['count']} cite \"{a['title']}\""
                        + (f" ({a['cite']})" if a["cite"] else "")
                    ))
            elif row["label"] == "tag shift":
                for d in row["entries"]:
                    out.append(_wrap(
                        f"tag shift: {d['label']} on {d['today']} items, against a "
                        f"{d['window_days']}-day average of {d['baseline']}"
                    ))
            else:
                names = "; ".join(
                    f"{e['name']} ({e['papers']})" for e in row["entries"]
                )
                out.append(_wrap(f"{row['label']}: {names}"))
        if synthesis["paragraph"]:
            out.append("")
            out.append(_wrap(synthesis["paragraph"]))
        out.append("")

    for n, item in enumerate(ordered, start=1):
        card = build_card(item)
        out.append(_rule())
        out.append(_wrap(f"{n}. {card['title']}"))
        byline = " · ".join(filter(None, [card["authors"], card["venue"], card["was_preprint"]]))
        if byline:
            out.append(_wrap(byline))
        if card["badges"]:
            out.append(_wrap(", ".join(b["label"] for b in card["badges"])))
        out.append("")
        if card["what"]:
            out.append(_wrap(card["what"]))
        if card["why"]:
            out.append("")
            out.append(_wrap(f"Why it matters — {card['why']}"))
        if card["caveats"]:
            out.append("")
            out.append(_wrap(f"Caveat: {card['caveats']}"))
        for link in card["links"]:
            out.append(_wrap(f"{link['label']}: {link['url']}"))
        out.append("")

    rows = [build_unreadable_row(it) for it in unreadable]
    if rows:
        out.append(_rule("="))
        out.append("ALSO PUBLISHED TODAY")
        out.append(_wrap(
            "These appeared in journals we track. Their abstracts are not "
            "openly available, so we cannot summarise them."
        ))
        out.append("")
        for row in rows:
            out.append(_wrap(row["title"]))
            meta = " · ".join(filter(None, [row["authors"], row["affiliation"], row["journal"]]))
            if meta:
                out.append(_wrap(meta, indent="    "))
            if row["url"]:
                out.append(_wrap(row["url"], indent="    "))
            out.append("")

    out.append(_rule("="))
    out.append(_wrap("Institute for Latent Publics · free, no ads, no metrics"))
    out.append(_wrap(
        "To stop receiving this, reply with the word unsubscribe."
    ))
    out.append("")
    out.append(_wrap(
        "Thank you to arXiv for use of its open access interoperability. This "
        "service was not reviewed or approved by, nor does it necessarily "
        "express or reflect the policies or opinions of, arXiv."
    ))
    return "\n".join(out) + "\n"
