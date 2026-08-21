# Urban Currents — working conventions

Read `docs/PRD-phase0.md` before changing pipeline behaviour. The PRD is the
specification; if something in it looks wrong, record the problem rather than
editing the document.

## Standing rules

- **English for anything published or executed.** Code, comments, docstrings,
  identifiers, commit messages, `README.md`, render templates, and everything
  under `content/`.
  **Korean is allowed for internal working documents** — everything under
  `prompts/`, and `docs/PRD-phase0.md`, which is the specification and has been
  Korean and committed since the first milestone. The line is what a reader
  outside the project sees, not what language a thought arrived in.

  `docs/` no longer holds analysis or decision notes. Since G4b it holds five
  files and nothing else: the specification (`PRD-phase0.md`), the generated
  report (`phase0-report.md`), the phase 0 ledger (`phase0-ledger.md`), and the
  two operations manuals (`OPERATIONS.md`, `OPERATIONS.ko.md`). `README.md` has
  described `docs/` that way for a long time; G4b made the description true.

## Where the working record goes

**`latentpublics/urban-currents-notes`, private.** It holds `prompts/` — the
directives, the completion reports and `DECISIONS.md` — and the fourteen
analysis, review and decision documents that used to sit in `docs/`.

Neither appears in this repository's history. `prompts/` was removed in 0W and
the fourteen documents in G4b, both with `git filter-repo`, before this
repository was made public. The notes repository was built by the mirror of the
same operation, so those documents kept their history: `git log` on any of them
still reaches the batch that wrote it.

This is a decision about audience, not a demotion. The record names people,
quotes unfinished judgements and discusses third parties, and it remains the
only account of why the code is the way it is. `CLAUDE.md` once argued the
opposite — that keeping analysis out of git risked losing it, because *"a
re-clone would have lost them"*. That worry was right and is answered rather
than abandoned: the documents are still in git, still versioned, still
recoverable by clone, in a repository whose audience is the project.

The habit is unchanged:

- Read the directive in `prompts/`, work, and **write the completion report to
  `prompts/reports/`** as a file, as before.
- Continue the `D`-number sequence in `prompts/reports/DECISIONS.md`.
- Expect `git status` to show nothing for any of it. `.gitignore` excludes
  `prompts/` wholesale, and a commit that adds a file under it is a mistake.
- Do **not** point at either repository's private files from anything that is
  committed here. `docs/`, `README.md` and code comments are read by people who
  will not have them; cite the batch ("0Q") and restate the fact and its
  denominator instead of linking a path. Numbers and populations always survive
  the move — dropping them is what the pointer was worth. G4b rewrote sixteen
  such pointers that way, in workflow comments, the operations manuals, the
  ledger and one collector docstring.
- Two scripts still write into `docs/` — `repo_link_audit.py` and
  `journals_rebuild_review_render.py`. Their output is gitignored, because a
  re-run would otherwise put a deliberately removed document back into a public
  repository.

Commit SHAs quoted in documents written before 0W or G4b refer to a pre-rewrite
history and no longer resolve. They are left in place with a note rather than
deleted: evidence that has become unreachable is still evidence that existed.
- **`content/` is pipeline output. Never hand-edit it.** That includes
  `content/items/`, `content/issues/`, `content/entities/`, and
  `content/graph/edges.jsonl`. To change what is in there, change the stage that
  produces it and re-run. `uc review --edit` is the one sanctioned path, and it
  records what it touched in `review.edits`.
- **`pipeline/models.py` is the single source of schema truth.** JSON Schema
  under `pipeline/schemas/` is generated from it by `uc schema` — never written
  by hand.
- **New dependencies go in via `uv add`**, not by editing `pyproject.toml`.
- **Never print, log, or commit secret values.** Read them through
  `pipeline.config.secret()`. The OpenAlex variable is `OPENALEX_KEY` — not
  `OPENALEX_API_KEY`.

## Schema discipline

- Fields that come from OpenAlex keep their OpenAlex names: `referenced_works`,
  `related_works`, `cited_by_count`, `topics`, `primary_location`. Renaming one
  breaks the Phase 1 interface promise (PRD §12).
- `work_key` never changes once assigned. New identifiers accumulate in `ids`.
- Everything in `entities` carries a canonical ID prefix (`openalex:`, `orcid:`,
  `ror:`, `wikidata:`, `method:`, `data:`, `github:`). Free strings are a test
  failure, not a warning. Unmatched candidates belong in
  `runs/{run_id}/unmatched.jsonl`.
- Adding a field is fine. Changing or removing the meaning of an existing one
  requires a migration script.

## Pipeline discipline

- Every stage must run alone: `uc <stage> --date YYYY-MM-DD`. Re-running
  summarize must never require re-collecting.
- Raw API responses are preserved verbatim under `runs/{run_id}/raw/`.
- A stage that cannot run (missing key, missing model) records `SKIPPED` and the
  run continues. One item's failure never stops the day's issue.
- The pipeline is idempotent: running the same date twice leaves `content/`
  byte-identical.
- The LLM never supplies bibliography, authors, links, or publication status.

## Cost discipline

- Every LLM call goes through `pipeline/llm.py`, which caches responses at
  `runs/cache/{prompt_version}/{work_key}.json`. Editing a prompt means bumping
  `llm.prompt_version` in `config/pipeline.yaml`.
- Unit tests never hit a real API. Inject `LLMClient(caller=...)`.
- Backfills do not summarise (PRD §10). Classifier scores only.
- OpenAlex spend is accumulated from `meta.cost_usd` and the stage stops at 80%
  of the daily budget.

## Testing

`uv run pytest`. Tests must pass with no API keys present and no network.
