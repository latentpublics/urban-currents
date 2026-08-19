"""Pydantic models — the single source of truth for the content schema.

JSON Schema files under ``pipeline/schemas/`` are generated from these models
(``uc schema``); never hand-edit them. Field names follow OpenAlex Work
conventions wherever the data comes from OpenAlex (PRD §4.2): ``referenced_works``,
``topics``, ``primary_location``, ``cited_by_count`` keep their upstream names.
Only fields we invent get new names.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "0.2.0"
PIPELINE_VERSION = "0.2.0"

# Alias for models that have a field literally named `date`: the assignment
# `date: Optional[date] = None` binds `date` in the class namespace, so pydantic
# would resolve the annotation to None rather than to datetime.date.
DateT = date

# Canonical ID prefixes (PRD §12). Every entity reference must carry one of these.
CANONICAL_PREFIXES = (
    "openalex:",
    "orcid:",
    "ror:",
    "wikidata:",
    "method:",
    "data:",
    "github:",
    "place:",
)

Confidence = Literal["high", "medium", "low"]
Basis = Literal["rule", "llm", "openalex"]


class StrictModel(BaseModel):
    """Base: reject unknown fields so schema drift surfaces as a test failure."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Bibliography
# --------------------------------------------------------------------------


class Institution(StrictModel):
    ror: Optional[str] = None
    name: Optional[str] = None

    @field_validator("ror")
    @classmethod
    def _bare_ror(cls, v: Optional[str]) -> Optional[str]:
        """`https://ror.org/02mhbdp94` -> `02mhbdp94`, wherever it comes from.

        The collector already normalises this, and phase 0d migrated the
        archive. Neither reaches a stage file written before the migration —
        so re-running `uc issue` over an old day merged the URL form back into
        published items, and the only thing that had been keeping the archive
        clean was nobody re-running those days. Normalising here makes the rule
        a property of the schema rather than of the order things ran in.
        """
        if not v:
            return None
        return v.strip().rstrip("/").rsplit("/", 1)[-1] or None


class Author(StrictModel):
    name: str
    orcid: Optional[str] = None
    openalex: Optional[str] = None
    institutions: list[Institution] = Field(default_factory=list)


class PrimaryLocation(StrictModel):
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    type: Optional[str] = None
    version: Optional[str] = None
    landing_page_url: Optional[str] = None
    pdf_url: Optional[str] = None


class Bibliography(StrictModel):
    """Written by collectors from source metadata only. The LLM never touches this."""

    title: str
    authors: list[Author] = Field(default_factory=list)
    publication_date: Optional[date] = None
    primary_location: PrimaryLocation = Field(default_factory=PrimaryLocation)
    abstract: Optional[str] = None
    # Source-provided subject categories (arXiv primary + cross-lists). Additive
    # field, not in PRD §3.2: the volume gate keys off category (§5.3) and the
    # per-category intake report needs it, so it has to survive to the Item.
    categories: list[str] = Field(default_factory=list)
    # arXiv's free-text `comment` — "Accepted to COLM 2026. Code and data:
    # https://github.com/…". The collector already parsed it and threw it away.
    # Measured over one raw response of 1,000 entries, 509 carry a comment and
    # 76 of those name a repository (phase 0k, X0-3).
    #
    # Corrected in 0P §Q0. The clause that used to sit here — "which is why the
    # archive has 0 code badges across 224 items" — was wrong twice over. The
    # archive now has **15** code badges across 2,224 items, so there was no
    # structural zero to explain; and the comment field is not the explanation
    # either. Re-measured over all 35,472 raw entries the pipeline has kept,
    # 17,994 carry a comment and 1,474 of those name a repository — but among
    # the 246 entries that survived the urban gate and became items, exactly
    # **one** carries a repository link the abstract does not already give.
    # General cs.* preprints release code far more often than urban ones do.
    comment: Optional[str] = None
    # Repository-ish hosts among OpenAlex `locations[]` — Zenodo, figshare,
    # Dryad. A deposit is evidence of released data in a way a phrase is not.
    repository_urls: list[str] = Field(default_factory=list)


class Ids(StrictModel):
    """OpenAlex convention: identifiers accumulate here, work_key never changes."""

    openalex: Optional[str] = None
    doi: Optional[str] = None
    arxiv: Optional[str] = None
    pmid: Optional[str] = None


class PublicationStatus(StrictModel):
    state: Literal["preprint", "published"] = "preprint"
    journal: Optional[str] = None
    source_id: Optional[str] = None
    doi: Optional[str] = None
    detected_at: Optional[datetime] = None


class Graph(StrictModel):
    """Taken from OpenAlex verbatim. The citation graph is free."""

    referenced_works: list[str] = Field(default_factory=list)
    related_works: list[str] = Field(default_factory=list)
    cited_by_count: int = 0


# --------------------------------------------------------------------------
# Summary and signals
# --------------------------------------------------------------------------


class SummaryEn(StrictModel):
    what: str = ""
    why: str = ""
    caveats: Optional[str] = None  # optional field (PRD §5.5)


class Summary(StrictModel):
    en: Optional[SummaryEn] = None
    # "ko" is added as a field in Phase 1, never by repurposing "en".


class Signal(StrictModel):
    """A structured judgement. Cheap, reusable as badge / filter / graph property."""

    value: Optional[bool | str] = None
    detail: Optional[str] = None
    url: Optional[str] = None
    confidence: Confidence = "medium"
    basis: Basis = "rule"


class Signals(StrictModel):
    geographic_scope: Optional[Signal] = None
    sample_size_reported: Optional[Signal] = None
    temporal_coverage_reported: Optional[Signal] = None
    code_available: Optional[Signal] = None
    data_available: Optional[Signal] = None
    is_retracted: Optional[Signal] = None


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------


class EntityRef(StrictModel):
    """A reference to an entity node. ``id`` must carry a canonical prefix —
    free strings in ``entities`` are a hard error (PRD §9)."""

    id: str
    label: str
    confidence: Optional[float] = None

    @field_validator("id")
    @classmethod
    def _canonical(cls, v: str) -> str:
        if not v.startswith(CANONICAL_PREFIXES):
            raise ValueError(
                f"entity id {v!r} lacks a canonical prefix {CANONICAL_PREFIXES}"
            )
        # `ror:https://ror.org/02mhbdp94` passes the prefix test and is still
        # wrong: the prefix announces the scheme and the value repeats it. It
        # comes back whenever a stage file written before the phase 0d
        # migration is merged into a published item.
        if v.startswith("ror:") and "ror.org/" in v:
            return f"ror:{v.rstrip('/').rsplit('/', 1)[-1]}"
        return v


class TopicRef(EntityRef):
    subfield: Optional[str] = None
    score: Optional[float] = None
    is_primary: bool = False


class PlaceRef(EntityRef):
    role: Optional[str] = None


PlacesStatus = Literal["resolved", "unspecified", "not_applicable", "not_attempted"]


class Entities(StrictModel):
    # OpenAlex-native: passed through, no LLM involvement.
    topics: list[TopicRef] = Field(default_factory=list)
    people: list[EntityRef] = Field(default_factory=list)
    orgs: list[EntityRef] = Field(default_factory=list)
    # Our overlay: this is the value we add.
    methods: list[EntityRef] = Field(default_factory=list)
    data: list[EntityRef] = Field(default_factory=list)
    tools: list[EntityRef] = Field(default_factory=list)
    # De-prioritised in v1.1. Empty is normal.
    places: list[PlaceRef] = Field(default_factory=list)
    places_status: PlacesStatus = "not_attempted"


# --------------------------------------------------------------------------
# Scores, cluster, provenance, review
# --------------------------------------------------------------------------


class ScoreComponents(StrictModel):
    relevance: float = 0.0
    source_multiplicity: float = 0.0
    artifact_completeness: float = 0.0
    novelty: float = 0.0


class Scores(StrictModel):
    relevance: float = 0.0
    headline: float = 0.0
    components: ScoreComponents = Field(default_factory=ScoreComponents)


MergeBasis = Literal["doi_match", "arxiv_location", "title_author_fuzzy", "singleton"]


class Cluster(StrictModel):
    cluster_id: Optional[str] = None
    members: list[str] = Field(default_factory=list)
    merge_basis: MergeBasis = "singleton"


class LlmProvenance(StrictModel):
    model: Optional[str] = None
    prompt_version: Optional[str] = None


class Tokens(StrictModel):
    input: int = Field(0, alias="in")
    output: int = Field(0, alias="out")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


AbstractSource = Literal["openalex", "crossref", "springer_api", "none"]


class Provenance(StrictModel):
    collected_at: Optional[datetime] = None
    collectors: list[str] = Field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION
    llm: Optional[LlmProvenance] = None
    classifier_version: Optional[str] = None
    cost_usd: float = 0.0
    tokens: Tokens = Field(default_factory=Tokens)
    # Where the abstract actually came from. Publishers withdrew abstracts from
    # OpenAlex at different times, so "we have an abstract" and "OpenAlex had an
    # abstract" stopped being the same statement; the enrichment order is
    # openalex → crossref → springer_api, and `none` is a measurable outcome.
    abstract_source: AbstractSource = "none"


class Review(StrictModel):
    status: Literal["pending", "approved", "rejected", "edited"] = "pending"
    reviewer_notes: Optional[str] = None
    # Field paths YJUN touched → raw material for prompt improvement.
    edits: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Item
# --------------------------------------------------------------------------

# DOIs are far less tidy than the examples suggest: parentheses, angle brackets
# and semicolons all occur in the wild (10.1016/s1361-9209(26)00356-1). Constrain
# the scheme, not the body; `work_key_to_filename` handles what a path cannot hold.
_WORK_KEY_RE = re.compile(r"^(arxiv|doi|openalex):\S+$")


class Item(StrictModel):
    """One artifact (one paper). Permanent and mutable."""

    schema_version: str = SCHEMA_VERSION
    work_key: str
    track: Literal["papers"] = "papers"
    first_published: Optional[date] = None
    updated: Optional[date] = None

    ids: Ids = Field(default_factory=Ids)
    bibliography: Bibliography
    publication_status: PublicationStatus = Field(default_factory=PublicationStatus)
    graph: Graph = Field(default_factory=Graph)
    summary: Summary = Field(default_factory=Summary)
    signals: Signals = Field(default_factory=Signals)
    badges: list[Literal["code", "data", "preprint", "published"]] = Field(
        default_factory=list
    )
    entities: Entities = Field(default_factory=Entities)
    # `lens` was declared here and never written — 224 items, all null. It is
    # removed rather than filled (phase 0k, X0-2). Filling it means asking the
    # LLM to sort papers into "behavior" and "system", which is a judgement with
    # no ground truth in this repo, no label supporting it, and no consumer
    # asking for it; a field in the schema is a promise to a reader that the
    # data exists, and this one could not be kept.
    scores: Scores = Field(default_factory=Scores)
    cluster: Cluster = Field(default_factory=Cluster)
    provenance: Provenance = Field(default_factory=Provenance)
    review: Review = Field(default_factory=Review)

    @field_validator("work_key")
    @classmethod
    def _work_key_shape(cls, v: str) -> str:
        if not _WORK_KEY_RE.match(v):
            raise ValueError(f"work_key {v!r} must look like 'arxiv:2608.01234'")
        return v

    @property
    def filename(self) -> str:
        """``content/items/{work_key}.json`` with ':' → '_' (PRD §6)."""
        return work_key_to_filename(self.work_key)


_PATH_HOSTILE = re.compile(r'[:/\\<>|?*"]')


def work_key_to_filename(work_key: str) -> str:
    """``content/items/{work_key}.json`` with path-hostile characters mapped to
    '_' (PRD §6). The mapping is one-way on purpose — ``work_key`` inside the
    file stays canonical, and nothing reconstructs it from the filename."""
    return _PATH_HOSTILE.sub("_", work_key) + ".json"


# --------------------------------------------------------------------------
# Issue
# --------------------------------------------------------------------------


class Headline(StrictModel):
    present: bool = False
    work_key: Optional[str] = None
    line: Optional[str] = None


class ScanMeta(StrictModel):
    arxiv_categories: int = 0
    journals: int = 0
    candidates_scanned: int = 0
    candidates_after_gate: int = 0
    items_published: int = 0
    minutes_saved_estimate: int = 0
    # Items that appeared in a tracked journal and could not be summarised
    # because no source exposes their abstract. Counted, and counted by
    # publisher, because it is the one blind spot this pipeline can measure
    # exactly and nobody else publishes.
    unreadable_count: int = 0
    unreadable_by_publisher: dict[str, int] = Field(default_factory=dict)


class StatusChange(StrictModel):
    work_key: str
    from_: str = Field(alias="from")
    to: str
    journal: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class SynthesisDeviation(StrictModel):
    label: str
    today: int
    baseline_per_day: float
    window_days: int


class SynthesisAnchor(StrictModel):
    """A foundation-canon work today's papers stand on."""

    openalex_id: str
    title: str
    year: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    citing_today: int = 0
    citing_work_keys: list[str] = Field(default_factory=list)
    days_since_last_cited: Optional[int] = None
    first_in_window: bool = False


class SynthesisCluster(StrictModel):
    """Two papers sharing references, named by the references themselves.

    `shared_titles` is the point: a cluster described by its shared bibliography
    has no room for an invented theme.
    """

    scope: Literal["today", "archive"]
    work_keys: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)
    shared: int = 0
    shared_titles: list[str] = Field(default_factory=list)
    partner_date: Optional[str] = None


class SynthesisName(StrictModel):
    name: str
    papers: int


class Synthesis(StrictModel):
    """How today connects to what came before (PRD §5.7, phase 0i).

    Everything except `paragraph` is measured. `paragraph` is written by an LLM
    from those measurements and nothing else, and is absent on days with no
    measured link — which is a correct outcome, not a missing field.
    """

    composition: dict[str, int] = Field(default_factory=dict)
    deviations: list[SynthesisDeviation] = Field(default_factory=list)
    deviation_status: Optional[str] = None
    deviation_note: Optional[str] = None
    anchors: list[SynthesisAnchor] = Field(default_factory=list)
    clusters: list[SynthesisCluster] = Field(default_factory=list)
    institutions_today: list[SynthesisName] = Field(default_factory=list)
    institutions_in_window: list[SynthesisName] = Field(default_factory=list)
    repeat_authors: list[SynthesisName] = Field(default_factory=list)
    window_days: int = 30
    first_internal_citation: bool = False
    paragraph: Optional[str] = None
    paragraph_omitted_reason: Optional[str] = None


class Issue(StrictModel):
    """One daily edition. Immutable once published.

    **`date` means the day we published, not the day the papers did** (phase 0k,
    X1). OpenAlex indexes a journal article a median of 1 day and a p90 of 2 days
    after its publication date, and no arXiv item has ever been visible to us on
    its own publication day. An issue dated by publication would therefore have
    to wait for the slow tail or silently omit it, and "today's papers" would be
    a claim we cannot keep.

    So an issue covers the window it drew from, and `covers_from` / `covers_to`
    record it. Each card carries its own publication date.
    """

    schema_version: str = SCHEMA_VERSION
    date: date
    # The publication-date window this issue drew from.
    #
    # **null means "this issue predates the change"** — that it was dated by
    # publication date, one day per issue, under the phase 0 rule. It does not
    # mean "unknown" and it does not mean "empty". The five issues written
    # before phase 0k keep null and are not migrated: an issue is immutable
    # once published, and back-filling a field would make them claim a window
    # nobody chose for them.
    covers_from: Optional[DateT] = None
    covers_to: Optional[DateT] = None
    # Built by `uc backfill-issues` from an archive of candidates rather than by
    # a live run (phase 0L, N1). A backfilled issue covers a **single** day,
    # where a live one covers a seven-day window, so the two answer "what did we
    # see" differently and an aggregate that mixes them without saying so is
    # comparing two things. Never inferred from the window: a one-day window is
    # also what `--smoke` produces.
    backfilled: bool = False
    headline: Headline = Field(default_factory=Headline)
    quiet_day: bool = False
    scan_meta: ScanMeta = Field(default_factory=ScanMeta)
    items: list[str] = Field(default_factory=list)
    # `Also published today`: work_keys that appeared in a tracked journal but
    # have no abstract from any source, so no card can be written about them.
    # A separate list rather than a flag inside `items` because they occupy no
    # slot and carry no summary — and because an item that later gains an
    # abstract is promoted into `items` without changing its identity.
    unreadable: list[str] = Field(default_factory=list)
    status_changes: list[StatusChange] = Field(default_factory=list)
    # The layer between the headline and the cards. Optional because an issue
    # written before phase 0i has none, and because a day with no measured
    # connection legitimately produces very little.
    synthesis: Optional[Synthesis] = None
    run_id: Optional[str] = None


# --------------------------------------------------------------------------
# Entity node
# --------------------------------------------------------------------------

Facet = Literal["topics", "methods", "data", "tools", "people", "orgs", "places"]


class Canonical(StrictModel):
    openalex: Optional[str] = None
    wikidata: Optional[str] = None


class Entity(StrictModel):
    id: str
    facet: Facet
    label: str
    aliases: list[str] = Field(default_factory=list)
    parent: Optional[str] = None
    canonical: Canonical = Field(default_factory=Canonical)
    item_count: int = 0
    first_seen: Optional[date] = None
    last_seen: Optional[date] = None

    @field_validator("id")
    @classmethod
    def _canonical_id(cls, v: str) -> str:
        if not v.startswith(CANONICAL_PREFIXES):
            raise ValueError(f"entity id {v!r} lacks a canonical prefix")
        return v


# --------------------------------------------------------------------------
# Edge (derived; content/graph/edges.jsonl)
# --------------------------------------------------------------------------

EdgeType = Literal[
    "uses_method", "uses_data", "uses_tool", "has_topic", "cites",
    "related_to", "authored_by", "affiliated_with", "studies_place",
]


class Edge(StrictModel):
    src: str
    dst: str
    type: EdgeType
    date: Optional[DateT] = None


# --------------------------------------------------------------------------
# Metrics (runs/{run_id}/metrics.json)
# --------------------------------------------------------------------------


class Counts(StrictModel):
    model_config = ConfigDict(extra="allow")

    arxiv_fetched: int = 0
    openalex_fetched: int = 0
    after_dedup: int = 0
    after_gate: int = 0
    classified: int = 0
    selected: int = 0
    summarized: int = 0
    published: int = 0


class Cost(StrictModel):
    embedding_usd: float = 0.0
    llm_usd: float = 0.0
    openalex_usd: float = 0.0
    total_usd: float = 0.0


class Linking(StrictModel):
    model_config = ConfigDict(extra="allow")

    topics_from_openalex: int = 0
    unmatched_methods: int = 0
    unmatched_data: int = 0
    openalex_enrich_pending: int = 0


class Metrics(StrictModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    date: Optional[DateT] = None
    # What this run was for. The citation layer's population is "works the
    # pipeline collected", and without this the only way to tell a collection
    # from a measurement that happened to call a collector is the directory
    # name. `collect` | `backfill` feed the reference base; anything else does
    # not (see `pipeline.graph.citation.CITATION_ORIGINS`).
    origin: str = "collect"
    counts: Counts = Field(default_factory=Counts)
    cost: Cost = Field(default_factory=Cost)
    tokens: Tokens = Field(default_factory=Tokens)
    timing: dict[str, float] = Field(default_factory=dict)
    linking: Linking = Field(default_factory=Linking)
    stages: dict[str, str] = Field(default_factory=dict)  # stage -> OK|SKIPPED|FAILED
    errors: list[str] = Field(default_factory=list)


