"""Shared collector helpers: identifier normalisation and text cleanup.

``work_key`` priority is arXiv ID → DOI → OpenAlex ID (PRD §5.2), and once
assigned it never changes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

ARXIV_SOURCE_ID = "S4306400194"

# Matches 2608.01234, 2608.01234v3, arXiv:2608.01234, and the pre-2007 form
# cs.CY/0701001.
#
# ★ The lookbehind excludes `-` as well as digits and dots (1D). It was there to
# stop the pattern biting into the middle of a longer number — Elsevier's
# `10.1016/j.trc.2026.105852` is safe because a dot precedes `2026` — and a
# hyphen was not considered, because nothing in the whitelist put one in front
# of a five-digit run until IJURR arrived. See `normalize_arxiv_id`.
_ARXIV_NEW = re.compile(r"(?<![\d.\-])(\d{4}\.\d{4,5})(v\d+)?", re.I)
_ARXIV_OLD = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$", re.I)
# A DOI, with or without a resolver prefix. Used to refuse the whole string
# rather than to parse it.
_DOI_SHAPE = re.compile(r"(?:^|/)(10\.\d{4,9}/\S+)$")


def normalize_arxiv_id(value: Optional[str]) -> Optional[str]:
    r"""Extract a bare arXiv ID (version stripped) from an ID, URL, or DOI.

    ★ **A DOI that is not an arXiv DOI never yields an arXiv ID (1D).** This is
    the structural half of the fix and the lookbehind above is the defensive
    half; either alone would have stopped the bug that prompted it, and the two
    fail in different directions.

    What happened: `10.1111/1468-2427.70128` is an IJURR article, and
    `1468-2427` is that journal's **ISSN**. The old pattern read the ISSN's
    second half plus the article number — `2427.70128` — as an arXiv ID, so
    twenty journal articles were stored under `arxiv:` work keys and eighteen of
    them were published as preprints.

    **Why it passed for three months.** The lookbehind `(?<![\d.])` was written
    against the failure everybody had seen: matching inside a longer digit run,
    which is what every Elsevier and Springer DOI looks like
    (`10.1016/j.trc.2026.105852` — a dot precedes, so it is refused). A hyphen
    reaches the same place and was never in the guard, and no journal in the
    whitelist put an ISSN in its DOI until IJURR was added. The pattern was not
    wrong about the cases it was written for; it had simply never met this one.

    **The same shape does occur elsewhere.** Measured over the 2,403 non-arXiv
    DOIs in the archive, exactly two prefixes reached it: `10.1111` (the twenty
    above) and `10.4108`, where EAI encodes a date —
    `10.4108/eai.16-4-2021.169337` yields `2021.16933`. That one never produced
    a bad work key, because the item was a real arXiv preprint whose ID came
    from arXiv itself; what it did instead was silently drop that DOI from
    `dedup.merge_keys`, since that function skips a DOI it believes is an arXiv
    DOI. A wrong identifier does not have to be visible to cost something.

    Other identifier families were checked and do not reach it: an ISBN has no
    dot-separated five-digit tail, and a PMID is a bare integer.
    """
    if not value:
        return None
    v = value.strip()
    m = _ARXIV_DOI.search(v)
    if m:
        v = m.group(1)
    elif _DOI_SHAPE.search(v):
        # A DOI from some other registrant. Whatever digits it contains, they
        # are that publisher's, not arXiv's.
        return None
    m = _ARXIV_NEW.search(v)
    if m:
        return m.group(1)
    m = _ARXIV_OLD.search(v)
    if m:
        return m.group(1).lower()
    return None


def normalize_doi(value: Optional[str]) -> Optional[str]:
    """Lower-cased bare DOI, with any URL prefix stripped."""
    if not value:
        return None
    v = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if v.startswith(prefix):
            v = v[len(prefix) :]
    return v or None


def normalize_openalex_id(value: Optional[str]) -> Optional[str]:
    """``https://openalex.org/W123`` → ``W123``."""
    if not value:
        return None
    return value.strip().rsplit("/", 1)[-1] or None


# Hosts that mean "the artifact is deposited here", as opposed to a mirror of
# the paper. A deposit is evidence in a way a sentence is not.
REPOSITORY_HOSTS = (
    "zenodo.org", "figshare.com", "datadryad.org", "dataverse",
    "osf.io", "pangaea.de", "dataverse.harvard.edu",
)


def repository_urls_from_work(work: dict) -> list[str]:
    """Repository deposits among a Work's locations (phase 0k, X0-3)."""
    out: list[str] = []
    for loc in work.get("locations") or []:
        url = (loc.get("landing_page_url") or "") or ""
        source = ((loc.get("source") or {}).get("display_name") or "").lower()
        haystack = f"{url.lower()} {source}"
        if any(host in haystack for host in REPOSITORY_HOSTS) and url:
            if url not in out:
                out.append(url)
    return out


def normalize_ror(value: Optional[str]) -> Optional[str]:
    """``https://ror.org/02mhbdp94`` → ``02mhbdp94``.

    OpenAlex reports institution RORs as full URLs, which produced entity IDs
    like ``ror:https://ror.org/02mhbdp94`` — the prefix announces the scheme and
    then the value repeats it. Every other canonical prefix in this schema
    carries a bare identifier (``orcid:0000-…``, ``openalex:W123``), and it also
    made entity filenames read ``https___ror.org_02mhbdp94.json``.
    """
    if not value:
        return None
    return value.strip().rstrip("/").rsplit("/", 1)[-1] or None


def arxiv_doi(arxiv_id: str) -> str:
    return f"10.48550/arxiv.{arxiv_id}"


def work_key_from(
    arxiv_id: Optional[str] = None,
    doi: Optional[str] = None,
    openalex_id: Optional[str] = None,
) -> Optional[str]:
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if doi:
        return f"doi:{normalize_doi(doi)}"
    if openalex_id:
        return f"openalex:{normalize_openalex_id(openalex_id)}"
    return None


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

# JATS and HTML markup arrives inside OpenAlex and Crossref titles and
# abstracts. Found in the Q1b label data as
# `<scp>DIFFERENTIATED INFRASTRUCTURAL CITIZENSHIP</scp> : Claims-Making…` —
# a title the labeller had to read past, and one that would have shipped on a
# card. Stripped at collection so no stage downstream has to know about it.
_TAG = re.compile(r"</?([a-zA-Z][a-zA-Z0-9:_-]*)(?:\s[^<>]*?)?/?>")

# Inline tags vanish; block-level tags leave a space behind. The distinction is
# not cosmetic: `H<sub>2</sub>O` must not become `H 2 O`, and
# `<jats:p>One.</jats:p><jats:p>Two.</jats:p>` must not become `One.Two.`
_INLINE_TAGS = {
    "sub", "sup", "i", "b", "em", "strong", "scp", "sc", "italic", "bold",
    "underline", "monospace", "roman", "sans-serif", "span", "a", "code",
    "inline-formula", "tex-math", "styled-content",
}
_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&apos;": "'", "&nbsp;": " ", "&#x2010;": "-", "&#8208;": "-",
}
# Markup removal leaves the space that used to sit outside the tag stranded
# before punctuation: `</scp> : Claims` → `  : Claims`.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]}])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")


def strip_markup(value: Optional[str]) -> Optional[str]:
    """Remove JATS/HTML tags and normalise the whitespace they leave behind.

    Only tags are removed, never their content: `<scp>CITIZENSHIP</scp>`
    carries a real word. Unrecognised entities are left alone rather than
    guessed at — a literal `&` in a title is more likely than a typo'd entity.
    """
    if value is None:
        return None

    def _replace(m: re.Match) -> str:
        name = m.group(1).lower()
        return "" if name.split(":")[-1] in _INLINE_TAGS else " "

    text = _TAG.sub(_replace, value)
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    text = _WS.sub(" ", text.replace("\n", " "))
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return text.strip() or None


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return strip_markup(value)


def normalize_title(title: str) -> str:
    """Lower-case, punctuation-stripped, whitespace-collapsed (PRD §5.2 rule 3)."""
    t = unicodedata.normalize("NFKD", title).lower()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def last_name(author_name: str) -> str:
    """Best-effort surname. Handles 'Jane Q. Doe' and 'Doe, Jane'."""
    name = author_name.strip()
    if "," in name:
        return normalize_title(name.split(",", 1)[0])
    parts = name.split()
    return normalize_title(parts[-1]) if parts else ""


def invert_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """Rebuild plain text from OpenAlex's ``abstract_inverted_index``."""
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    if not positions:
        return None
    positions.sort()
    return clean_text(" ".join(w for _, w in positions))
