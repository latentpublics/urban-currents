"""Can a repository be found that names the paper? (phase 0Q, R3 — gate only)

`scripts/repo_link_audit.py` measured what we can prove from text already on
disk. This measures the two external paths, **on the same sample**, so the
decision to build or not build rests on a number.

## The rule, and why it is inverted

Searching GitHub for a paper's **title** and linking a repository that looks
similar is a guess, and the mockup's standing rule — *"Only verified URLs; never
a DOI guess"* — forbids exactly that. **A wrong repository link is worse than no
link**: it attributes someone else's code to an author who did not write it.

So the search runs the other way. We search for the **arXiv ID or the DOI
string**, and accept a repository only when **the repository itself names that
identifier** — in its description, its README, or a `CITATION.cff`. Then the
connection is the repository's own claim about itself, and we are recording it
rather than inferring it.

## Paths measured

1. **DataCite `relatedIdentifiers`** — the GitHub↔Zenodo integration mints a DOI
   per release and that record points back at the paper. Public REST, no auth.
   The most structural of the routes: it is a citation, not a text match.
2. **GitHub code search for the identifier** — 60 requests/hour unauthenticated,
   5,000 with `GITHUB_TOKEN`. **An absent token is a supported state**, not a
   degraded one; it only means fewer requests per hour.

## What this does not do

It writes no badge and links nothing. It reports a recovery rate. If that rate
is low, the honest outcome is to report the number and not build the feature.

Usage:
    uv run python scripts/repo_backlink_probe.py --n 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.config import cfg, github_token  # noqa: E402

UA = f"urban-currents/0.1 (mailto:{cfg('contact_email', 'unknown')})"
DATACITE = "https://api.datacite.org/dois"
# `/search/code` requires authentication and returns 401 without a token, so
# the unauthenticated path uses `/search/repositories`, which reads name,
# description and README — the three places the rule above says the repository
# may state its own identifier. Code search is the upgrade a token buys, not
# the baseline.
GITHUB_SEARCH = "https://api.github.com/search/repositories"
GITHUB_CODE_SEARCH = "https://api.github.com/search/code"

# Which relation types mean "this deposit belongs to that paper". `IsCitedBy`
# and friends are deliberately excluded: a repository citing a paper is not the
# paper's repository, and that distinction is the whole point of the rule above.
SUPPLEMENT = {
    "IsSupplementTo",
    "IsSourceOf",
    "IsDerivedFrom",
    "IsPartOf",
    "IsVersionOf",
    "Documents",
    "IsDocumentedBy",
}


def _get(url: str, headers: dict, timeout: int = 20) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:  # noqa: BLE001
        return 0, {}


def datacite_related(doi: str) -> list[dict]:
    """DataCite records that declare themselves related to this DOI."""
    q = urllib.parse.quote(f'relatedIdentifiers.relatedIdentifier:"{doi}"')
    status, body = _get(f"{DATACITE}?query={q}&page[size]=25", {})
    if status != 200:
        return []
    out = []
    for rec in body.get("data", []):
        attrs = rec.get("attributes", {})
        for rel in attrs.get("relatedIdentifiers", []) or []:
            same = (rel.get("relatedIdentifier") or "").lower() == doi.lower()
            # **A preprint is not a repository.** The first run of this counted
            # arXiv's own DataCite record — `IsVersionOf`, resourceType
            # `Preprint`, URL `arxiv.org/abs/...` — as a hit, which is the paper
            # pointing at itself. Filtering on the resource type is definitional
            # (we are looking for software or data), not a rule fitted to that
            # case.
            kind = (attrs.get("types") or {}).get("resourceTypeGeneral")
            if kind not in ("Software", "Dataset", "Collection"):
                continue
            if same and rel.get("relationType") in SUPPLEMENT:
                out.append({
                    "doi": attrs.get("doi"),
                    "url": attrs.get("url"),
                    "relation": rel.get("relationType"),
                    "types": (attrs.get("types") or {}).get("resourceTypeGeneral"),
                })
    return out


def github_backlink(identifier: str, token: str | None) -> tuple[list[dict], int]:
    """Repositories that name this identifier, and **how many of them there are**.

    The count is returned because it is the number that decides whether a hit
    means anything. Searching for `1706.03762` — *Attention Is All You Need* —
    returns **4,147 repositories**, and essentially none of them is that paper's
    own code: they are reading lists, paper collections and survey repos that
    cite it. So "the repository names the identifier" is **necessary and not
    sufficient**, and a rule that stopped there would attach a stranger's
    reading list to an author's paper.

    Reported rather than silently thresholded. Deciding here that "fewer than N
    matches means it is the real one" would be tuning a rule to cases we have
    not looked at.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    q = urllib.parse.quote(f'"{identifier}" in:readme,description,name')
    status, body = _get(f"{GITHUB_SEARCH}?q={q}&per_page=10&sort=stars", headers)
    if status != 200:
        return [{"_status": status}], 0
    total = int(body.get("total_count") or 0)
    hits = []
    for repo in body.get("items", []):
        hits.append({
            "repo": repo.get("full_name"),
            "url": repo.get("html_url"),
            "description": (repo.get("description") or "")[:100],
            "stars": repo.get("stargazers_count"),
        })
    return hits, total


def identifiers(item) -> dict:
    key = item.work_key
    out = {}
    if key.startswith("arxiv:"):
        out["arxiv"] = key.split(":", 1)[1]
    if key.startswith("doi:"):
        out["doi"] = key.split(":", 1)[1]
    ids = item.ids
    for name in ("doi", "arxiv"):
        v = getattr(ids, name, None)
        if v and name not in out:
            out[name] = str(v).replace("https://doi.org/", "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--skip-github", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    audit_path = paths.RUNS / "repo_link_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    keys = [r["work_key"] for r in audit["rows"]][: a.n]
    token = github_token()
    print(f"sample: {len(keys)} published items (same draw as the audit)")
    print(f"GITHUB_TOKEN: {'present' if token else 'absent - 60 requests/hour'}")

    rows = []
    requests_made = {"datacite": 0, "github": 0}
    for i, key in enumerate(keys, start=1):
        item = store.load_item(key)
        if item is None:
            continue
        ids = identifiers(item)
        rec = {
            "work_key": key,
            "title": item.bibliography.title[:70],
            "ids": ids,
            "datacite": [],
            "github": [],
            "github_total": 0,
            "github_status": None,
        }

        if ids.get("doi"):
            rec["datacite"] = datacite_related(ids["doi"])
            requests_made["datacite"] += 1
            time.sleep(0.4)

        if not a.skip_github:
            ident = ids.get("arxiv") or ids.get("doi")
            if ident:
                hits, total = github_backlink(ident, token)
                requests_made["github"] += 1
                if hits and "_status" in hits[0]:
                    rec["github_status"] = hits[0]["_status"]
                else:
                    rec["github"] = hits
                    rec["github_total"] = total
                # GitHub's search limit is 10/minute unauthenticated, so 6.5s
                # between calls stays under it without a token.
                time.sleep(6.5 if not token else 1.0)

        found = bool(rec["datacite"]) or bool(rec["github"])
        rec["found"] = found
        rows.append(rec)
        mark = "HIT " if found else "    "
        extra = f" (status {rec['github_status']})" if rec["github_status"] else ""
        print(
            f"{mark}[{i}/{len(keys)}] {key}  datacite={len(rec['datacite'])} "
            f"github={len(rec['github'])}{extra}"
        )

    n = len(rows) or 1
    dc = sum(1 for r in rows if r["datacite"])
    gh = sum(1 for r in rows if r["github"])
    either = sum(1 for r in rows if r["found"])
    blocked = sum(1 for r in rows if r["github_status"])
    print(f"\n=== feasibility gate, n={len(rows)} ===")
    print(f"  DataCite relatedIdentifiers: {dc}  ({dc / n:.2f})")
    print(f"  GitHub names the identifier: {gh}  ({gh / n:.2f})")
    ambiguous = [r for r in rows if r.get("github_total", 0) > 3]
    if ambiguous:
        print(f"    of which {len(ambiguous)} matched more than 3 repositories "
              f"(max {max(r['github_total'] for r in ambiguous)}) - a hit count "
              f"like that is a reading list, not an author's repository")
    print(f"  either path:                 {either}  ({either / n:.2f})")
    if blocked:
        print(f"  GitHub requests refused:     {blocked} - rate limit or auth")
    print(f"  requests made: {requests_made}")

    out = paths.RUNS / "repo_backlink_probe.json"
    out.write_text(
        json.dumps({
            "n": len(rows),
            "token_present": bool(token),
            "requests": requests_made,
            "datacite_hits": dc,
            "github_hits": gh,
            "either": either,
            "github_blocked": blocked,
            "rows": rows,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
