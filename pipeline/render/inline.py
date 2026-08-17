"""Turn one rendered issue into the email version of itself (phase 0j, W6).

**There is no second template.** The email is derived from `preview.html`, not
written beside it — the moment two templates exist, "web = email" stops being a
structural fact and becomes a promise someone has to keep.

What derivation means here: take the stylesheet the page already carries, work
out which of its declarations apply to which elements, and write them onto the
elements as `style` attributes. Mail clients strip `<style>` blocks
inconsistently and Gmail's clipping is famously unpredictable, so an email that
depends on a stylesheet is an email that depends on luck.

**A small inliner rather than a dependency.** `premailer` would work and brings
lxml with it; the CSS in this repo is ours and its selector vocabulary is
narrow — classes, elements, and a few descendant and pseudo forms. So the
subset is implemented here and everything outside it is *reported*, never
silently dropped:

- `.class`, `element`, `.a .b`, `.a > b`, `.a, .b` — inlined.
- `:hover`, `:focus-within`, `::before`, `:first-child` — skipped. A mail client
  has no hover and no reliable pseudo-element support, and a rule that cannot
  apply must not be pretended into a style attribute.
- `@media` — skipped entirely. Email is one column at one width; the mobile
  block would either duplicate or contradict the base rule.

Anything the parser does not recognise raises rather than passing through, so a
new selector shape shows up as a test failure instead of as an email that looks
subtly wrong in a client nobody here runs.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional

# The email body width. 600px is the conventional safe measure and the mockup's
# own email pass uses it.
EMAIL_WIDTH_PX = 600

VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
})

# Selectors that cannot apply in an email and must not be inlined. `:root` is
# here because it defines custom properties rather than styling an element —
# those get resolved into the declarations that reference them instead, since
# no mail client can be relied on to support var().
_SKIP_SELECTOR = re.compile(
    r"(::|:hover|:focus|:active|:first-child|:last-child|:not\(|^:root$|^\*$)"
)
_SIMPLE = re.compile(r"^(?P<element>[a-zA-Z][\w-]*)?(?P<classes>(?:\.[\w-]+)*)$")


class UnsupportedSelector(RuntimeError):
    """Raised when the stylesheet grows a shape this inliner cannot honour."""


def parse_rules(css: str) -> list[tuple[str, str]]:
    """(selector, declarations) pairs, with media queries and pseudos removed.

    Media blocks are dropped whole rather than flattened: their declarations
    contradict the base rules by design, and an email has one width.
    """
    # Strip comments first so a brace inside one cannot confuse the scan.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    out: list[tuple[str, str]] = []
    i = 0
    while i < len(css):
        at = css.find("@media", i)
        brace = css.find("{", i)
        if brace == -1:
            break
        if at != -1 and at < brace:
            depth = 0
            j = css.index("{", at)
            for k in range(j, len(css)):
                if css[k] == "{":
                    depth += 1
                elif css[k] == "}":
                    depth -= 1
                    if depth == 0:
                        i = k + 1
                        break
            else:
                break
            continue
        close = css.find("}", brace)
        if close == -1:
            break
        selector = css[i:brace].strip()
        body = css[brace + 1:close].strip()
        i = close + 1
        if not selector or not body:
            continue
        for part in selector.split(","):
            part = part.strip()
            if not part or _SKIP_SELECTOR.search(part):
                continue
            out.append((part, body))
    return out


def custom_properties(css: str) -> dict[str, str]:
    """The `--uc-*` values defined on `:root`, light mode only.

    Dark mode lives in a `prefers-color-scheme` block, which `parse_rules`
    drops with every other media query. An email gets the light palette, which
    is the right call anyway: a mail client that honours the media query would
    apply it to our inlined light values and produce a half-inverted page.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@media[^{]*\{(?:[^{}]|\{[^{}]*\})*\}", "", css, flags=re.S)
    out: dict[str, str] = {}
    for block in re.findall(r":root\s*\{(.*?)\}", css, flags=re.S):
        for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", block):
            out[name.strip()] = value.strip()
    return out


def resolve_vars(declarations: str, properties: dict[str, str]) -> str:
    """Replace `var(--x)` with its value. Unresolved names are left alone.

    Left alone rather than blanked: a declaration that silently loses its value
    is worse than one a client ignores, because the first is invisible.
    """
    def sub(match: re.Match) -> str:
        name = match.group(1).strip()
        return properties.get(name, match.group(0))

    previous = None
    out = declarations
    # Values can reference other properties; two passes is plenty for this
    # stylesheet and the loop stops as soon as nothing changes.
    while out != previous:
        previous = out
        out = re.sub(r"var\((--[\w-]+)\)", sub, out)
    return out


def _clean(declarations: str) -> str:
    """Make a declaration block safe and compact inside a `style="..."` attribute.

    Double quotes are the sharp edge: a font stack is written
    `font-family: "Source Serif 4", Georgia` in the stylesheet, and pasted
    verbatim into a double-quoted attribute it closes the attribute early and
    the rest of the declaration becomes bogus markup. Single quotes are valid
    CSS and survive.
    """
    out = declarations.replace('"', "'")
    return re.sub(r"\s+", " ", out).strip()


def _specificity(selector: str) -> tuple[int, int, int]:
    """(classes, elements, source order stands in for the rest)."""
    return (selector.count("."), len(re.findall(r"(?:^|[\s>])([a-zA-Z][\w-]*)", selector)), 0)


def _matches(selector: str, tag: str, classes: set[str], ancestors: list[tuple[str, set[str]]]) -> bool:
    """Match a descendant/child selector against an element and its ancestors."""
    parts = [p for p in re.split(r"\s*>\s*|\s+", selector) if p]
    if not parts:
        return False

    def unit_matches(unit: str, t: str, cs: set[str]) -> bool:
        m = _SIMPLE.match(unit)
        if not m:
            raise UnsupportedSelector(unit)
        element = m.group("element")
        wanted = {c for c in m.group("classes").split(".") if c}
        if element and element != t:
            return False
        return wanted <= cs

    if not unit_matches(parts[-1], tag, classes):
        return False
    remaining = parts[:-1]
    pool = list(ancestors)
    for unit in reversed(remaining):
        while pool:
            atag, acls = pool.pop()
            if unit_matches(unit, atag, acls):
                break
        else:
            return False
    return True


class _Inliner(HTMLParser):
    def __init__(self, rules: list[tuple[str, str]]):
        super().__init__(convert_charrefs=False)
        self.rules = rules
        self.out: list[str] = []
        self.stack: list[tuple[str, set[str]]] = []
        self.in_style = False

    def handle_starttag(self, tag, attrs):
        self._emit_tag(tag, attrs, self_closing=tag in VOID_ELEMENTS)
        if tag not in VOID_ELEMENTS:
            self.stack.append((tag, self._classes(attrs)))

    def handle_startendtag(self, tag, attrs):
        self._emit_tag(tag, attrs, self_closing=True, explicit=True)

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False
            return
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
        self.out.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.in_style:
            self.out.append(data)

    def handle_comment(self, data):
        self.out.append(f"<!--{data}-->")

    def handle_decl(self, decl):
        self.out.append(f"<!{decl}>")

    def handle_entityref(self, name):
        self.out.append(f"&{name};")

    def handle_charref(self, name):
        self.out.append(f"&#{name};")

    @staticmethod
    def _classes(attrs) -> set[str]:
        for k, v in attrs:
            if k == "class" and v:
                return set(v.split())
        return set()

    def _emit_tag(self, tag, attrs, self_closing: bool, explicit: bool = False) -> None:
        if tag == "style":
            # The block itself goes; its content is being written onto elements.
            self.in_style = True
            return

        classes = self._classes(attrs)
        declarations: list[tuple[tuple[int, int, int], str]] = []
        for selector, body in self.rules:
            try:
                if _matches(selector, tag, classes, self.stack):
                    declarations.append((_specificity(selector), body))
            except UnsupportedSelector:
                raise
        declarations.sort(key=lambda d: d[0])

        merged = _clean("; ".join(b.strip().rstrip(";") for _, b in declarations))
        rendered = []
        seen_style = False
        for k, v in attrs:
            if k == "style":
                seen_style = True
                v = f"{merged}; {v}" if merged else v
            rendered.append(f'{k}="{v}"' if v is not None else k)
        if merged and not seen_style:
            rendered.append(f'style="{merged}"')

        space = " " + " ".join(rendered) if rendered else ""
        self.out.append(f"<{tag}{space}{' /' if explicit else ''}>")


def inline_css(html: str) -> str:
    """Rewrite a rendered page with its stylesheet applied per element."""
    styles = re.findall(r"<style[^>]*>(.*?)</style>", html, flags=re.S)
    properties: dict[str, str] = {}
    for block in styles:
        properties.update(custom_properties(block))
    rules = []
    for block in styles:
        rules.extend(
            (selector, resolve_vars(body, properties))
            for selector, body in parse_rules(block)
        )
    parser = _Inliner(rules)
    parser.feed(html)
    parser.close()
    return "".join(parser.out)


def to_email(html: str, width_px: int = EMAIL_WIDTH_PX) -> str:
    """The email edition: inlined, single column, fixed width, no stylesheet.

    The serif stack collapses to Georgia because no mail client will fetch a
    font and Georgia is on effectively every machine — which is what the
    mockup's own email pass specifies.
    """
    out = inline_css(html)
    out = out.replace(
        'font-family: var(--uc-font-serif)', 'font-family: Georgia, serif'
    )
    # Custom properties do not survive inlining in most clients; the values are
    # already resolved into the declarations that referenced them, so any
    # leftover var() is a rule that had no inlined counterpart.
    out = re.sub(r"var\(--uc-font-serif\)", "Georgia, serif", out)
    out = re.sub(r"var\(--uc-font-sans\)", "Helvetica, Arial, sans-serif", out)
    out = out.replace(
        "<body>",
        f'<body style="margin:0;padding:16px;background:#fdfdfc">'
        f'<div style="max-width:{width_px}px;margin:0 auto">',
    )
    out = out.replace("</body>", "</div></body>")
    return out
