"""How much are we missing? Count before building (phase 0Q, R3).

The question is whether papers with a public repository are going unbadged. We
do **not** know that yet: 1,027 published items carry 15 `code` and 45 `data`
badges, and whether that is a miss or the actual release rate of urban research
is exactly what has not been measured.

So this measures a **lower bound on what we are missing**, using only text
already on disk. **No API call is made from here.** A repository mentioned in an
abstract or an arXiv comment that carries no badge is a miss we can prove
without asking anyone; anything beyond that needs the external paths, and those
should only be built if this number justifies them.

Writes `docs/repo-link-audit.md` — a table a person can check by hand, because
the whole point is that the answer is currently a guess.

Usage:
    uv run python scripts/repo_link_audit.py --n 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402
from pipeline.signals import _CODE_PHRASE, _CODE_URL, _DATA_PHRASE  # noqa: E402

# Hosts that are a deposit rather than a code repository. Kept apart because
# "there is a repository" and "there is code" are different claims (R3).
DEPOSIT = re.compile(
    r"https?://(?:www\.)?(?:zenodo\.org|figshare\.com|datadryad\.org|dryad\.org|"
    r"osf\.io|dataverse\.[a-z.]+|data\.mendeley\.com)/\S*", re.I
)


def published_keys() -> list[str]:
    keys: list[str] = []
    seen = set()
    for p in sorted(paths.ISSUES.glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        for k in list(doc.get("items") or []):
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def evidence(item) -> dict:
    abstract = item.bibliography.abstract or ""
    comment = item.bibliography.comment or ""
    code_url = _CODE_URL.search(abstract) or _CODE_URL.search(comment)
    dep = DEPOSIT.search(abstract) or DEPOSIT.search(comment)
    return {
        "code_url": code_url.group(0).rstrip(".,);") if code_url else "",
        "deposit_url": dep.group(0).rstrip(".,);") if dep else "",
        "code_phrase": bool(_CODE_PHRASE.search(abstract)),
        "data_phrase": bool(_DATA_PHRASE.search(abstract)),
        "openalex_repo": list(item.bibliography.repository_urls or []),
        "has_abstract": bool(abstract.strip()),
        "has_comment": bool(comment.strip()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260819)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    keys = published_keys()
    rng = random.Random(a.seed)
    sample = rng.sample(keys, min(a.n, len(keys)))

    rows = []
    for key in sample:
        item = store.load_item(key)
        if item is None:
            continue
        ev = evidence(item)
        badges = list(item.badges)
        # A miss we can prove without leaving the disk: the text names a
        # repository and no badge says so.
        missed_code = bool(ev["code_url"]) and "code" not in badges
        missed_data = bool(ev["deposit_url"] or ev["openalex_repo"]) and "data" not in badges
        rows.append({
            "work_key": key,
            "title": item.bibliography.title,
            "badges": badges,
            **ev,
            "missed_code": missed_code,
            "missed_data": missed_data,
        })

    tally = Counter()
    for r in rows:
        tally["n"] += 1
        tally["with_abstract"] += r["has_abstract"]
        tally["with_comment"] += r["has_comment"]
        tally["code_badge"] += "code" in r["badges"]
        tally["data_badge"] += "data" in r["badges"]
        tally["names_a_repo_in_text"] += bool(r["code_url"])
        tally["names_a_deposit_in_text"] += bool(r["deposit_url"] or r["openalex_repo"])
        tally["missed_code"] += r["missed_code"]
        tally["missed_data"] += r["missed_data"]

    lines = [
        "# Repository link audit — what we can prove we are missing",
        "",
        f"Sample: **{tally['n']} of {len(keys)} published items**, drawn with "
        f"seed `{a.seed}` so it can be re-drawn identically.",
        "",
        "**No API call was made to produce this.** Every column comes from text",
        "already on disk — the abstract, the arXiv comment, and OpenAlex",
        "`locations[]`. That makes this a **lower bound**: a repository we cannot",
        "see from here may still exist. It is the honest starting number, because",
        "whether 15 code badges in 1,027 published items is a miss or the real",
        "release rate of urban research had never been measured.",
        "",
        "## Totals",
        "",
        "| | count | of sample |",
        "|---|---:|---:|",
    ]
    for k in ("with_abstract", "with_comment", "code_badge", "data_badge",
              "names_a_repo_in_text", "names_a_deposit_in_text",
              "missed_code", "missed_data"):
        lines.append(f"| `{k}` | {tally[k]} | {tally[k] / tally['n']:.2f} |")

    lines += [
        "",
        "`missed_*` is the number this exists to produce: the text names a",
        "repository or a deposit and **no badge says so**. Those are misses that",
        "need no external service to prove.",
        "",
        "## The sample",
        "",
        "| # | title | id | badges | repo in text? | deposit in text? |",
        "|---:|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, start=1):
        title = r["title"].replace("|", chr(92)+"|")[:70]
        repo = r["code_url"] or ("phrase only" if r["code_phrase"] else "—")
        dep = r["deposit_url"] or (", ".join(r["openalex_repo"]) if r["openalex_repo"]
                                   else ("phrase only" if r["data_phrase"] else "—"))
        flag = ""
        if r["missed_code"] or r["missed_data"]:
            flag = " **← unbadged**"
        lines.append(
            f"| {i} | {title} | `{r['work_key']}` | "
            f"{', '.join(r['badges']) or '—'} | {repo[:60]} | {dep[:60]}{flag} |"
        )

    out = paths.DOCS / "repo-link-audit.md"
    # The second half of this document — the feasibility gate — is written by
    # `scripts/repo_backlink_probe.py` and finished **by hand**: its
    # correctness column is a human verdict on each hit, which is the entire
    # point of it. Re-running this must not silently delete that. Learned the
    # direct way: it did, once.
    marker = "# The feasibility gate"
    tail = ""
    if out.exists():
        existing = out.read_text(encoding="utf-8")
        if marker in existing:
            tail = "\n---\n\n" + marker + existing.split(marker, 1)[1]
    out.write_text("\n".join(lines) + "\n" + tail, encoding="utf-8", newline="\n")

    json_out = paths.RUNS / "repo_link_audit.json"
    json_out.write_text(
        json.dumps({"seed": a.seed, "n": tally["n"], "population": len(keys),
                    "totals": dict(tally), "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8", newline="\n",
    )
    print(json.dumps(dict(tally), indent=2))
    print(f"-> {out}\n-> {json_out}")


if __name__ == "__main__":
    main()
