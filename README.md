# Urban Currents

A daily scan of urban data science research: what appeared, what it found, and
why it matters.

This repository is **Phase 0** — not a product, but a measuring instrument for
four questions:

1. Can a classifier filter urban research out of the daily arXiv firehose well
   enough to be useful?
2. Is there enough signal in the field to sustain a daily edition?
3. Where should the "quiet day" line sit?
4. Does reviewing a day's output fit in fifteen minutes?

The deliverable is this code plus [`docs/phase0-report.md`](docs/phase0-report.md),
which holds the answers and the evidence for them. The specification is
[`docs/PRD-phase0.md`](docs/PRD-phase0.md).

## How it works

```
collect → dedup → gate → classify → select → link → summarize → score → issue → preview
```

- **collect** — arXiv (7 categories, 3-second throttle) and OpenAlex (whitelist
  journals). Raw responses are preserved verbatim.
- **dedup** — the same paper arrives as an arXiv preprint and as an OpenAlex
  journal Work. Three merge rules collapse them, and a preprint that later gets
  published updates in place rather than headlining twice.
- **gate** — the three high-volume arXiv categories (cs.LG, cs.CV, cs.AI) must
  clear a generous keyword filter. The other four skip it. The gate's recall is
  measured, not assumed.
- **enrich** — Springer Nature withdrew its non-OA abstracts from OpenAlex in
  2022 and Elsevier followed in 2024, so half the journal path arrives with no
  abstract at all. Crossref and Springer's own free API are asked, in that
  order, for the ones they can still supply. What no source has is published
  anyway, under *Also published today*, with the facts we do have and nothing
  invented — the abstract is the only evidence a summary is allowed to use, and
  without it there is no summary to write.
- **classify** — logistic regression over local `bge-base-en-v1.5` embeddings,
  trained on what Urban Studies / Geography-Planning / Transportation journals
  publish rather than on a hand-picked seed set. The output is a calibrated
  probability, so a threshold on it means something.
- **link** — topics, authors and institutions come from OpenAlex verbatim.
  Methods, data and tools are our overlay: extracted from the abstract, then
  matched against controlled vocabulary. Nothing unmatched becomes a tag.
- **summarize** — two layers per paper. *What* it did, with the measurements
  exposed as reported. *Why it matters*, stated as what would be different.
  The abstract is the only evidence; bibliography never comes from the model.
- **score / issue / preview** — a headline score picks the day's lead paper, or
  declares a quiet day. Output is JSON in `content/` plus a single self-contained
  HTML preview.

No database. The content is JSON in git.

## Quick start

```bash
uv sync --extra embed
cp .env.example .env          # OPENALEX_KEY, ANTHROPIC_API_KEY, CONTACT_EMAIL

# See the whole pipeline run on built-in sample papers — no network, no keys:
uv run uc run --date 2026-08-11 --fixture --no-llm
uv run uc preview --date 2026-08-11 --open
```

Then a real day:

```bash
uv run uc run --date 2026-08-14
uv run uc review --date 2026-08-14
uv run uc report
```

Every stage also runs alone (`uc summarize --date …`), which is deliberate:
re-running a summary should never mean re-collecting a day.

Setup for the classifier, the backfill and the threshold calibration is in
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## Repository

```
pipeline/      the stages; models.py is the single source of schema truth
scripts/       one-time builders: journal whitelist, training set, classifier
content/       published output — items, issues, entities, derived edges
vocab/         controlled vocabulary and source lists
config/        scoring weights and pipeline settings
runs/          raw responses, metrics, caches, labels (gitignored)
models/        trained classifiers with their metrics
docs/          the PRD, the report, the operations manual
tests/         run with `uv run pytest`; no network, no keys required
```

## Design commitments

These are the expensive ones to change later, so they are fixed early:

- **Schema.** `pipeline/models.py` is authoritative; JSON Schema is generated
  from it. Fields taken from OpenAlex keep their OpenAlex names.
- **`work_key` is permanent.** New identifiers accumulate in `ids`; the key never
  changes.
- **Every entity tag carries a canonical ID prefix.** `method:`, `data:`,
  `github:`, `openalex:`, `orcid:`, `ror:`, `wikidata:`. Free strings are a test
  failure. Unmatched candidates go to `runs/{run_id}/unmatched.jsonl` instead.
- **Item and Issue are separate.** Items are permanent and mutable; issues are
  daily and immutable. That is what makes a preprint-to-published transition a
  status update rather than a second headline.
- **Idempotence.** Running the same date twice leaves `content/` byte-identical.
- **Failure isolation.** A missing key or one bad LLM response degrades a stage,
  never the run.
- **Local embeddings.** Same input, same vector — so classifier experiments
  reproduce, and backfills are free.

## Status

Phase 0. Not deployed, not scheduled, no site. Phase 1 adds the Astro site and
inherits the schema, the `content/` layout, and the render templates' DOM.

## Licence and attribution

Content derived from OpenAlex, which is CC0. arXiv metadata is used under
arXiv's terms of use.

Thank you to arXiv for use of its open access interoperability. This service was
not reviewed or approved by, nor does it necessarily express or reflect the
policies or opinions of, arXiv.

Papers are always linked to arXiv; no PDF or source is served from here.
Abstracts are read but never republished — summaries are written from the facts
in them, because the abstracts themselves stay under their publishers'
copyright.
