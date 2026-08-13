# Urban Currents — working conventions

Read `docs/PRD-phase0.md` before changing pipeline behaviour. The PRD is the
specification; if something in it looks wrong, record the problem rather than
editing the document.

## Standing rules

- **English for anything published or executed.** Code, comments, docstrings,
  identifiers, commit messages, `README.md`, render templates, and everything
  under `content/`.
  **Korean is allowed for internal working documents** — `docs/` analysis and
  decision notes, and all of `prompts/`. The earlier rule said "everything
  committed", which was read literally enough to keep three analysis documents
  out of git entirely; a re-clone would have lost them, and `docs/PRD-phase0.md`
  is Korean and has been committed since the first milestone. The line is what
  a reader outside the project sees, not what language a thought arrived in.
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
