"""The difference between "could not run" and "ran and broke" (hotfix H2).

One base class, in a module with no imports of its own, so anything that needs
to say *"I was not configured"* can inherit it without dragging `run_stages`
into a cycle.

The distinction is not cosmetic. `OpenAlexUnavailable` was a plain
`RuntimeError`, so a missing `OPENALEX_KEY` was recorded as `FAILED` — the same
status OpenAlex returning 500s gets. `docs/OPERATIONS.md` had been promising
`collect.openalex: SKIPPED | OPENALEX_KEY missing` for eight batches, and the
code did not do that. On YJUN's first CI run the log said the source had failed
when the truth was that nobody had put a key in the repository settings, which
sends a person to look at the wrong thing.

**The verdict does not change.** A day missing a required source is still
`not_published`, because half the declared scope is a different claim and X3's
rule stands. What changes is that the reason is now accurate.
"""

from __future__ import annotations


class StageSkipped(RuntimeError):
    """A stage could not run — no key, no model, no SDK — and the run goes on.

    Inherit this for anything that means "not configured". Raising a bare
    RuntimeError instead makes a setup problem look like a broken dependency,
    and the two get investigated very differently.
    """
