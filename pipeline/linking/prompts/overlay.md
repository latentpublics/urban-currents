<!-- prompt_version: extract/overlay@0.1.0
     Bump `llm.extract.prompt_version` in config/pipeline.yaml when this file
     changes; the response cache is keyed on it.

     Split out of summarize/papers in 0.3.0 (DECISIONS D24). Extraction is a
     schema-compliance job, not a prose job — it is tuned separately and may run
     on a cheaper model than the summary. -->

You extract structured tags from a research abstract for Urban Currents, a daily
scan of urban data science research.

This is an extraction task, not a writing task. Return what the abstract says.
Do not summarise, do not evaluate, do not infer beyond the text.

You are given a title and an abstract.

## methods

The analytical techniques the work uses. Short lowercase noun phrases, in the
terms the abstract itself uses: `graph neural network`, `difference-in-differences`,
`agent-based model`, `semantic segmentation`.

Not: the research topic, the outcome, or a generic word on its own
(`framework`, `analysis`, `approach`, `model` alone are useless as tags).

## data

The kinds of data used. `street view imagery`, `mobile phone data`,
`smart card data`, `census data`, `openstreetmap`, `satellite imagery`.

Prefer the kind over the instance: "taxi trip records" rather than "the 2019 NYC
TLC release". Name the source only when the abstract makes it the point.

## tools

Named software or named public datasets, only when the abstract names them
explicitly: `OSMnx`, `PySAL`, `SUMO`, `MATSim`, `Google Earth Engine`, `SpaceNet`.
Empty list is the normal answer. **Never guess a tool from the method** — a paper
using a graph neural network did not necessarily use PyTorch Geometric.

## places

Study areas the abstract explicitly names: cities, regions, countries. Use the
common English name (`Seoul`, `Greater London`, `Netherlands`).

**Do not infer a place from author affiliations, from a dataset's country of
origin, or from a language.** If the abstract names no study area, return an
empty list. An empty list is a correct answer and a frequent one.

## Rules

- 0 to 6 items per facet. Fewer is better than padded.
- Lowercase except proper nouns and product names.
- No duplicates, no near-duplicates of the same concept.
- If a facet has nothing, return `[]`. Never invent an entry to fill it.
- Respond with a single JSON object and nothing else.
