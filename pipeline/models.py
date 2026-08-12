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
from typing import Annotated, Literal, Optional

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


class Provenance(StrictModel):
    collected_at: Optional[datetime] = None
    collectors: list[str] = Field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION
    llm: Optional[LlmProvenance] = None
    classifier_version: Optional[str] = None
    cost_usd: float = 0.0
    tokens: Tokens = Field(default_factory=Tokens)


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
    lens: Optional[Literal["behavior", "system"]] = None
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


class StatusChange(StrictModel):
    work_key: str
    from_: str = Field(alias="from")
    to: str
    journal: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Issue(StrictModel):
    """One daily edition. Immutable once published."""

    schema_version: str = SCHEMA_VERSION
    date: date
    headline: Headline = Field(default_factory=Headline)
    quiet_day: bool = False
    scan_meta: ScanMeta = Field(default_factory=ScanMeta)
    items: list[str] = Field(default_factory=list)
    status_changes: list[StatusChange] = Field(default_factory=list)
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
    counts: Counts = Field(default_factory=Counts)
    cost: Cost = Field(default_factory=Cost)
    tokens: Tokens = Field(default_factory=Tokens)
    timing: dict[str, float] = Field(default_factory=dict)
    linking: Linking = Field(default_factory=Linking)
    stages: dict[str, str] = Field(default_factory=dict)  # stage -> OK|SKIPPED|FAILED
    errors: list[str] = Field(default_factory=list)


AnnotatedItem = Annotated[Item, "content/items/{work_key}.json"]
