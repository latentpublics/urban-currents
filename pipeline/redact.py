"""Scrub secrets out of anything on its way to disk (phase 1L, L3/L4).

**This exists because 1L was about to arm a leak that had been harmless.**

`collectors/abstracts.py` calls Springer with the key as a *query parameter*:

    self._http().get(SPRINGER_API, params={"q": ..., "api_key": key, "p": 1})

and reports a failure with the exception's own text:

    self.run.error(f"springer {doi}: {type(e).__name__}: {e}")

A `requests` exception stringifies with the full URL in it, key and all. That
line has always gone into `runs/{run_id}/metrics.json` — and `runs/` is
gitignored and thrown away when the runner dies, so nothing ever escaped.

1L changes both halves of that sentence. L3 persists failure reasons into
`content/runs_log/`, which is **committed to a public repository**, and L4
uploads `metrics.json` as a CI artefact. The same string that was safe becomes
published twice over.

So every error string is scrubbed at the one chokepoint it all flows through,
`Run.error()`, rather than at the call sites — there are dozens of those and
the next one will be written by someone who has not read this file.

Two passes, because they fail differently:

- **By parameter name**, which catches a key this process does not hold: a
  backfill re-running someone else's captured URL, a third-party service we
  add next month, a token in a redirect.
- **By value**, which catches a key that appears with no name attached at all —
  echoed in a JSON body, in a header dump, in a message that just concatenates.

Neither is sufficient alone, and the cost of both is a regex over a few
kilobytes of error text per run.
"""

from __future__ import annotations

import os
import re

PLACEHOLDER = "[REDACTED]"

# Query- or header-style `name=value`. Deliberately wide on the name and
# deliberately narrow on the value: it stops at the first delimiter so a
# redaction never eats the rest of a sentence a human needs to read.
_PARAM = re.compile(
    r"(?i)\b(api[_-]?key|apikey|key|token|access[_-]?token|secret|password|passwd|pwd|auth)"
    r"(\s*[=:]\s*)"
    r"([^&\s\"'<>,;)\]}]+)"
)

# `Authorization: Bearer xyz` and friends.
_BEARER = re.compile(r"(?i)\b(bearer|basic)(\s+)([A-Za-z0-9._~+/=-]{8,})")

# Environment variables whose values must never reach disk. `CONTACT_EMAIL` is
# not a secret but it is a personal address that ends up in a public repo, and
# the OpenAlex variable is `OPENALEX_KEY` — not `OPENALEX_API_KEY` (CLAUDE.md).
SECRET_ENV = (
    "OPENALEX_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "SPRINGER_API_KEY",
    "GITHUB_TOKEN",
    "UC_SMTP_PASSWORD",
    "UC_SMTP_USER",
    "UC_SMTP_HOST",
    "UC_ALERT_RECIPIENT",
    "CONTACT_EMAIL",
)

# Below this length a "secret" is more likely to be a substring of ordinary
# prose than a credential, and scrubbing it would corrupt readable text without
# protecting anything.
MIN_SECRET_LEN = 8


def secret_values() -> list[str]:
    """The values to scrub, longest first so a prefix never masks a longer one.

    Read straight from the environment rather than through `config.secret()`:
    this must work when the caller is an exception handler and must never
    itself raise, log, or cache.
    """
    out = set()
    for name in SECRET_ENV:
        value = (os.environ.get(name) or "").strip()
        if len(value) >= MIN_SECRET_LEN:
            out.add(value)
    return sorted(out, key=len, reverse=True)


def redact(text: str) -> str:
    """Return `text` with credentials replaced by `[REDACTED]`.

    Never raises. A scrubber that throws inside an error handler would turn a
    recorded failure into an unrecorded one, which is the opposite of what this
    whole batch is for.
    """
    if not text:
        return text
    try:
        for value in secret_values():
            if value in text:
                text = text.replace(value, PLACEHOLDER)
        text = _PARAM.sub(lambda m: f"{m.group(1)}{m.group(2)}{PLACEHOLDER}", text)
        text = _BEARER.sub(lambda m: f"{m.group(1)}{m.group(2)}{PLACEHOLDER}", text)
        return text
    except Exception:  # noqa: BLE001 - see docstring
        return PLACEHOLDER
