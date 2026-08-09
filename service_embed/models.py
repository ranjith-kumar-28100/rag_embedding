"""ORM models (spec §14).

  * ``CollectionRow``  — the registry (§7.1): model fingerprint, HNSW/storage
    params, point count, which one the alias serves.
  * ``IndexJob`` / ``IndexFile`` — async indexing progress, mirroring stage 1's
    job/job_file shape so the UI polling contract is identical.
  * ``EvalRun`` / ``EvalItem`` — a config snapshot plus per-question and
    aggregate scores, so eval results stay comparable across reindexes.

Chunk *content* is never duplicated here: children live in the Qdrant payload,
parents in ``parent_chunks`` (owned by the library's docstore module).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    PARTIAL = "partial"  # finished, but some files failed
    FAILED = "failed"  # every file failed


class FileStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobKind(str, enum.Enum):
    INDEX = "index"
    REINDEX = "reindex"


TERMINAL_FILE_STATUSES: frozenset[FileStatus] = frozenset({FileStatus.DONE, FileStatus.FAILED})


# --------------------------------------------------------------------------- #
# Collection registry (§7.1)
# --------------------------------------------------------------------------- #
class CollectionRow(Base):
    __tablename__ = "collections"

    name: Mapped[str] = mapped_column(String(255), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)

    hnsw_m: Mapped[int] = mapped_column(Integer, nullable=False)
    hnsw_ef_construct: Mapped[int] = mapped_column(Integer, nullable=False)
    quantization: Mapped[str] = mapped_column(String(16), nullable=False)
    on_disk_vectors: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    on_disk_payload: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    sparse_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Mirror of the Qdrant alias. Qdrant is the source of truth for what serves
    # traffic; this column is a cheap read for the registry listing.
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    point_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


# --------------------------------------------------------------------------- #
# Indexing jobs
# --------------------------------------------------------------------------- #
class IndexJob(Base):
    __tablename__ = "index_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[JobKind] = mapped_column(Enum(JobKind), default=JobKind.INDEX, nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.QUEUED, nullable=False, index=True
    )
    collection: Mapped[str] = mapped_column(String(255), nullable=False)
    total_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # reindex auditability (§9): what changed, and why it was asked for.
    source_collection: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_diff: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    activate_on_success: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    files: Mapped[list["IndexFile"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="IndexFile.created_at",
    )


class IndexFile(Base):
    __tablename__ = "index_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("index_jobs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    input_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    status: Mapped[FileStatus] = mapped_column(
        Enum(FileStatus), default=FileStatus.QUEUED, nullable=False, index=True
    )
    # percent = processed_children / total_children (§7.3), written per batch.
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_children: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_children: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    total_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parents_stored: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    warnings: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    job: Mapped["IndexJob"] = relationship(back_populates="files")


# --------------------------------------------------------------------------- #
# Evaluation runs (§10.4)
# --------------------------------------------------------------------------- #
class EvalStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[EvalStatus] = mapped_column(
        Enum(EvalStatus), default=EvalStatus.QUEUED, nullable=False, index=True
    )
    collection: Mapped[str] = mapped_column(String(255), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)

    total_questions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_questions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # A snapshot, not a foreign key: comparing runs across reindexes is only
    # meaningful if each run remembers the config it was measured under.
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    aggregate: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    warnings: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    items: Mapped[list["EvalItem"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="EvalItem.position"
    )


class EvalItem(Base):
    __tablename__ = "eval_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped["EvalRun"] = relationship(back_populates="items")
