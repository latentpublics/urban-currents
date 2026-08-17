# Retired issues

Files here were written into `content/issues/` and should not have been. They
are **kept, not deleted** — each one is evidence of a defect that has since been
closed, and deleting it would remove the only trace that the defect existed.

`archive_rows()` and the site build do not read this directory, so a retired
issue appears in no archive, no feed and no aggregate. It is still schema-valid
and `verify_phase0.py` still validates it: a retired file is a wrong file, not a
malformed one.

## 2026-08-14.json — the ghost issue

Written by `verify_phase0.py`, not by a day's work. Its own contents say so:
0 items, `quiet_day: true`, `candidates_scanned: 0`, no synthesis.

Phase 0h ran the full pipeline against live APIs as a verification step, on the
current date, and arXiv had not yet indexed that day's submissions. The run
therefore saw nothing and published a quiet day — **the exact failure phase 0k's
first invariant forbids**: a day we could not see, recorded as a day with
nothing in it.

Two changes closed it. D135 gave verification a `UC_CONTENT` sandbox so a test
run can no longer write into the archive at all, and phase 0k's outcome model
(X3) makes "we did not look" a `not_published` outcome that writes no issue.

This file is the boundary marker between those two eras.
