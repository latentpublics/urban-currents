"""Vendor the three OFL webfonts into `site/assets/fonts/` (phase 0R, T1).

## Why this exists at all

`base.css.j2` has named Source Serif 4, Public Sans and IBM Plex Mono since 0j,
and **0j W0-6 banned webfonts**, so on any machine without them installed the
site fell back to Georgia and the system sans. The mockup loaded Google Fonts.
**The two could never have looked alike**, and most of the difference YJUN is
seeing is that.

W0-6's reasons were self-containment of the single-file preview and safety in
email. **The site is a different artefact and neither reason reaches it**: these
are same-origin files served from a relative path, so the site still makes no
external request. `preview.html` and `email.html` keep the stack alone — the
first is defined by being one self-contained file, and mail clients strip
`@font-face` regardless.

## What is downloaded

**Latin subset only, woff2 only.** The full family with every unicode range
would be megabytes for a page that is entirely Latin plus the occasional Korean
run, and the Korean stack deliberately stays webfont-free.

**Two of the three are variable fonts.** Google serves one file for every weight
in the query, so asking for 400/600/700 separately wrote identical bytes three
times — 291 KB of duplicate on the first run here. Files are deduplicated by
content and the `@font-face` declares a weight *range*, which is what a variable
font is for.

All three are SIL Open Font License 1.1, which permits redistribution; the
licence text is fetched alongside and committed with the fonts, because
redistributing without it is the one thing the OFL actually forbids.

Usage:
    uv run python scripts/vendor_fonts.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Inside the package, not in .  is build output — a clean rebuild
# would drop anything kept there, and the fonts are a source asset.
# copies them into  so the published tree is complete and
# reproducible from a checkout.
OUT = ROOT / "pipeline" / "render" / "assets" / "fonts"

# A modern browser UA, because the css2 endpoint serves TTF to anything it does
# not recognise and woff2 only to browsers it believes support it.
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

FAMILIES = {
    "source-serif-4": (
        "Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400",
        "body serif — headline, lead, card titles",
    ),
    "public-sans": (
        "Public+Sans:ital,wght@0,400;0,600;0,700;1,400",
        "UI sans — nav, meta rail, labels",
    ),
    "ibm-plex-mono": (
        "IBM+Plex+Mono:wght@400;600",
        "figures — stat rail, dates, counts",
    ),
}

LICENCES = {
    "source-serif-4": "https://raw.githubusercontent.com/google/fonts/main/ofl/sourceserif4/OFL.txt",
    "public-sans": "https://raw.githubusercontent.com/google/fonts/main/ofl/publicsans/OFL.txt",
    "ibm-plex-mono": "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexmono/OFL.txt",
}

BLOCK = re.compile(r"/\*\s*([a-z0-9-]+)\s*\*/\s*(@font-face\s*\{[^}]*\})", re.I)
STYLE = re.compile(r"font-style:\s*([a-z]+)", re.I)
WEIGHT = re.compile(r"font-weight:\s*([\d\s]+)", re.I)
SRC = re.compile(r"url\((https://[^)]+\.woff2)\)", re.I)


def fetch(url: str, ua: bool = True) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA} if ua else {})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.woff2"):
        stale.unlink()

    faces: list[dict] = []
    by_digest: dict[str, dict] = {}
    total = 0

    for slug, (query, note) in FAMILIES.items():
        url = f"https://fonts.googleapis.com/css2?family={query}&display=swap"
        css = fetch(url).decode("utf-8")
        for subset, block in BLOCK.findall(css):
            # **Latin only.** Cyrillic, Greek and Vietnamese are pure weight on
            # a page that never renders them.
            if subset.lower() != "latin":
                continue
            src = SRC.search(block)
            if not src:
                continue
            m_style = STYLE.search(block)
            style = m_style.group(1) if m_style else "normal"
            m_weight = WEIGHT.search(block)
            raw = (m_weight.group(1) if m_weight else "400").split()
            weight = raw[-1]

            data = fetch(src.group(1))
            digest = hashlib.md5(data).hexdigest()
            seen = by_digest.get(digest)
            if seen is not None:
                seen["weights"].append(weight)
                print(f"  {seen['file']:<28} also serves weight {weight}")
                continue

            # Named by weight, deduplicated by content. IBM Plex Mono is NOT
            # variable — its 400 and 600 are different files — and naming by
            # family alone made the second overwrite the first, silently
            # shipping semibold as the body mono.
            suffix = '-italic' if style == 'italic' else ''
            name = f"{slug}-{weight}{suffix}.woff2"
            (OUT / name).write_bytes(data)
            total += len(data)
            face = {"family": slug, "file": name, "style": style,
                    "weights": [weight], "bytes": len(data)}
            by_digest[digest] = face
            faces.append(face)
            print(f"  {name:<28} {len(data) / 1024:6.1f} KB")
        print(f"{slug}  [{note}]")

        licence = fetch(LICENCES[slug], ua=False).decode("utf-8")
        (OUT / f"OFL-{slug}.txt").write_text(licence, encoding="utf-8", newline="\n")

    summary = []
    for f in sorted(faces, key=lambda x: (x["family"], x["style"])):
        ws = sorted(f["weights"], key=int)
        f["range"] = f"{ws[0]} {ws[-1]}" if len(ws) > 1 else ws[0]
        summary.append(f)

    print(f"\ntotal woff2: {total / 1024:.1f} KB across {len(faces)} file(s)")
    for f in summary:
        print(f"  {f['family']:<16} {f['range']:<9} {f['style']:<7} {f['file']}")

    (OUT / "manifest.json").write_text(
        json.dumps({"total_bytes": total, "faces": summary}, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    main()
