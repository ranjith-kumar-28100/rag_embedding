"""API response DTOs. Kept separate from ORM rows on purpose."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from rag_embedding.models import SearchMode

from .models import CollectionRow, EvalRun, IndexFile, IndexJob
from .progress import compute_job_progress


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
class IndexFileOut(BaseModel):
    id: str
    filename: str
    status: str
    progress: int
    processed_children: int
    total_children: int
    total_chunks: int
    parents_stored: int
    skipped: int
    seconds: float
    warnings: list[str]
    error: str | None

    @classmethod
    def from_row(cls, row: IndexFile) -> "IndexFileOut":
        return cls(
            id=row.id,
            filename=row.filename,
            status=row.status.value,
            progress=row.progress,
            processed_children=row.processed_children,
            total_children=row.total_children,
            total_chunks=row.total_chunks,
            parents_stored=row.parents_stored,
            skipped=row.skipped,
            seconds=row.seconds,
            warnings=json.loads(row.warnings) if row.warnings else [],
            error=row.error,
        )


class IndexJobOut(BaseModel):
    id: str
    kind: str
    status: str
    progress: int
    collection: str
    source_collection: str | None
    reason: str | None
    config_diff: dict[str, Any] | None
    activate_on_success: bool
    total_files: int
    files_done: int
    files_failed: int
    children_indexed: int
    parents_stored: int
    created_at: datetime
    updated_at: datetime
    error: str | None
    files: list[IndexFileOut]

    @classmethod
    def from_row(cls, job: IndexJob) -> "IndexJobOut":
        files = list(job.files)
        return cls(
            id=job.id,
            kind=job.kind.value,
            status=job.status.value,
            progress=compute_job_progress([f.progress for f in files]),
            collection=job.collection,
            source_collection=job.source_collection,
            reason=job.reason,
            config_diff=job.config_diff,
            activate_on_success=job.activate_on_success,
            total_files=job.total_files,
            files_done=sum(1 for f in files if f.status.value == "done"),
            files_failed=sum(1 for f in files if f.status.value == "failed"),
            children_indexed=sum(f.processed_children for f in files),
            parents_stored=sum(f.parents_stored for f in files),
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=job.error,
            files=[IndexFileOut.from_row(f) for f in files],
        )


class IndexJobCreatedOut(BaseModel):
    index_job_id: str
    status: str
    collection: str
    total_files: int


# --------------------------------------------------------------------------- #
# Collections
# --------------------------------------------------------------------------- #
class CollectionOut(BaseModel):
    name: str
    version: int
    provider: str
    model_id: str
    dim: int
    hnsw_m: int
    hnsw_ef_construct: int
    quantization: str
    on_disk_vectors: bool
    on_disk_payload: bool
    sparse_enabled: bool
    is_active: bool
    point_count: int
    created_at: datetime

    @classmethod
    def from_row(cls, row: CollectionRow, *, is_active: bool) -> "CollectionOut":
        return cls(
            name=row.name,
            version=row.version,
            provider=row.provider,
            model_id=row.model_id,
            dim=row.dim,
            hnsw_m=row.hnsw_m,
            hnsw_ef_construct=row.hnsw_ef_construct,
            quantization=row.quantization,
            on_disk_vectors=row.on_disk_vectors,
            on_disk_payload=row.on_disk_payload,
            sparse_enabled=row.sparse_enabled,
            is_active=is_active,
            point_count=row.point_count,
            created_at=row.created_at,
        )


class CollectionsOut(BaseModel):
    alias: str
    active_collection: str | None
    collections: list[CollectionOut]


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class ReindexIn(BaseModel):
    """Rebuild with new model/params, then optionally take over the alias."""

    reason: str | None = None
    source_collection: str | None = None  # defaults to whatever the alias serves
    activate_on_success: bool = False


class ActivateOut(BaseModel):
    alias: str
    collection: str


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class EvalRunOut(BaseModel):
    id: str
    status: str
    progress: int
    collection: str
    mode: SearchMode
    top_k: int
    total_questions: int
    completed_questions: int
    aggregate: dict[str, float]
    config_snapshot: dict[str, Any]
    warnings: list[str]
    error: str | None
    created_at: datetime
    items: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_row(cls, run: EvalRun, *, include_items: bool = True) -> "EvalRunOut":
        return cls(
            id=run.id,
            status=run.status.value,
            progress=run.progress,
            collection=run.collection,
            mode=run.mode,
            top_k=run.top_k,
            total_questions=run.total_questions,
            completed_questions=run.completed_questions,
            aggregate={k: float(v) for k, v in (run.aggregate or {}).items()},
            config_snapshot=dict(run.config_snapshot or {}),
            warnings=json.loads(run.warnings) if run.warnings else [],
            error=run.error,
            created_at=run.created_at,
            items=(
                [
                    {
                        "question": item.question,
                        "retrieved_chunk_ids": item.retrieved_chunk_ids,
                        "metrics": item.metrics,
                        "note": item.note,
                    }
                    for item in run.items
                ]
                if include_items
                else []
            ),
        )


class EvalCreatedOut(BaseModel):
    eval_run_id: str
    status: str
    total_questions: int


class HealthOut(BaseModel):
    status: str
    qdrant: str
    redis: str
    active_collection: str | None
