"""Wire models: the stage-1 chunk contract plus this service's own DTOs.

``Chunk`` is a *local re-declaration* of the rag_ingestion output schema (spec §3
/ Appendix A). It is deliberately not imported from stage 1: the two services
share a wire contract, not code. ``extra="ignore"`` means stage 1 can add fields
without breaking us, and this service never re-derives structure from the text.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Stable namespace for point ids. uuid5(NAMESPACE, chunk_id) is deterministic, so
# re-indexing the same chunk overwrites its point instead of duplicating it (§6.1).
POINT_NAMESPACE = uuid.UUID("6f1b2f2e-6c2f-5a3d-9d8e-6a5f4c3b2a10")


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    SLIDE = "slide"
    ROW = "row"


class ChunkRole(str, Enum):
    PARENT = "parent"
    CHILD = "child"


# --------------------------------------------------------------------------- #
# Input contract (§3)
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    """One line of a stage-1 ``chunks.jsonl``. Opaque beyond these fields."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    doc_id: str
    parent_id: str | None = None
    role: ChunkRole
    block_type: BlockType
    text: str
    section_path: list[str] = Field(default_factory=list)
    source_uri: str
    file_type: str
    page: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    row_range: tuple[int, int] | None = None
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def point_id(self) -> str:
        return point_id_for(self.chunk_id)

    def to_payload(self) -> dict[str, Any]:
        """Full Qdrant payload — every §3 field, metadata as flat scalars.

        ``section_path`` stays a list so a keyword index matches any element of
        it (filter by "3. Architecture" without knowing the leaf); the joined
        form is kept alongside for display. Free-form ``metadata`` is nested
        under its own key with every value coerced to a scalar, so a new stage-1
        metadata key needs no code change here.
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "parent_id": self.parent_id,
            "role": self.role.value,
            "block_type": self.block_type.value,
            "text": self.text,
            "section_path": list(self.section_path),
            "section_path_text": " > ".join(self.section_path),
            "source_uri": self.source_uri,
            "file_type": self.file_type,
            "page": self.page,
            "slide_number": self.slide_number,
            "sheet_name": self.sheet_name,
            "row_start": self.row_range[0] if self.row_range else None,
            "row_end": self.row_range[1] if self.row_range else None,
            "token_count": self.token_count,
            "metadata": {k: flatten_scalar(v) for k, v in self.metadata.items()},
        }


def point_id_for(chunk_id: str) -> str:
    """Deterministic Qdrant point id (§6.1) — idempotent upserts, never dupes."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def flatten_scalar(value: Any) -> str | int | float | bool | None:
    """Coerce an arbitrary metadata value into a flat, filterable scalar."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return " > ".join(str(v) for v in value)
    return str(value)


# --------------------------------------------------------------------------- #
# Collections (§7.1)
# --------------------------------------------------------------------------- #
class CollectionSpec(BaseModel):
    """Everything needed to create — and to safely query — one collection.

    The model fingerprint travels with the collection so the query path embeds
    with the same model that indexed it; a mismatch is a hard error (§5.1).
    """

    name: str
    provider: str
    model_id: str
    dim: int
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    quantization: str = "int8"
    on_disk_vectors: bool = True
    on_disk_payload: bool = True
    sparse_enabled: bool = True


# --------------------------------------------------------------------------- #
# Search (§8)
# --------------------------------------------------------------------------- #
SearchMode = Literal["dense", "sparse", "hybrid"]


class SearchFilters(BaseModel):
    """Structured filters over indexed payload keys (§8.3). All ANDed."""

    model_config = ConfigDict(extra="forbid")

    doc_id: list[str] = Field(default_factory=list)
    file_type: list[str] = Field(default_factory=list)
    block_type: list[str] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.doc_id or self.file_type or self.block_type or self.section_path)


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    mode: SearchMode = "hybrid"
    filters: SearchFilters = Field(default_factory=SearchFilters)
    with_parents: bool | None = None
    prefetch_k: int | None = None
    # Per-request override of rerank_enabled. Lets the playground A/B the
    # reranker on one query without restarting the service.
    rerank: bool | None = None


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)
    source: str  # "dense" | "sparse" | "hybrid" (+"+rerank") — how the hit was scored
    # Pre-rerank fused score, kept when a cross-encoder replaces `score`. The two
    # are on different scales, so both are reported rather than one silently
    # standing in for the other.
    retrieval_score: float | None = None


class ParentContext(BaseModel):
    """A parent chunk returned as retrieval context, with its matched children."""

    chunk_id: str
    doc_id: str
    text: str
    score: float  # best score among matched children
    payload: dict[str, Any] = Field(default_factory=dict)
    children: list[SearchHit] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    collection: str
    reranked: bool = False  # whether a cross-encoder reordered the hits
    hits: list[SearchHit] = Field(default_factory=list)
    parents: list[ParentContext] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)  # milliseconds


# --------------------------------------------------------------------------- #
# Evaluation (§10)
# --------------------------------------------------------------------------- #
class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    # Optional references. Either literal context text or chunk_ids — both are
    # accepted and matched (see rag_embedding.eval.metrics).
    ground_truth_contexts: list[str] = Field(default_factory=list)
    reference_answer: str | None = None


class EvalRequest(BaseModel):
    dataset: list[EvalQuestion]
    top_k: int | None = None
    mode: SearchMode = "hybrid"
    with_parents: bool | None = None
    # Per-run rerank override — the point of evaluation is to measure whether the
    # reranker actually helps, so it has to be switchable per run.
    rerank: bool | None = None
    # LLM-judged metrics (slow, judge-quality dependent). Off by default (§10.2).
    llm_metrics: bool = False


class EvalItemResult(BaseModel):
    question: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    note: str | None = None


class EvalReport(BaseModel):
    collection: str
    mode: SearchMode
    top_k: int
    aggregate: dict[str, float] = Field(default_factory=dict)
    items: list[EvalItemResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
