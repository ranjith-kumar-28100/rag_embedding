"""DB access layer — the only module that reads/writes ORM rows.

API handlers and Celery tasks call these; neither touches a session directly.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from rag_embedding.models import CollectionSpec

from .models import (
    CollectionRow,
    EvalItem,
    EvalRun,
    EvalStatus,
    FileStatus,
    IndexFile,
    IndexJob,
    JobKind,
)
from .progress import child_percent, compute_job_status


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class NewFile:
    filename: str
    input_path: str


# --------------------------------------------------------------------------- #
# Collections registry
# --------------------------------------------------------------------------- #
def upsert_collection(
    session: Session, spec: CollectionSpec, version: int, *, point_count: int = 0
) -> CollectionRow:
    row = session.get(CollectionRow, spec.name)
    if row is None:
        row = CollectionRow(name=spec.name, version=version)
        session.add(row)
    row.version = version
    row.provider = spec.provider
    row.model_id = spec.model_id
    row.dim = spec.dim
    row.hnsw_m = spec.hnsw_m
    row.hnsw_ef_construct = spec.hnsw_ef_construct
    row.quantization = spec.quantization
    row.on_disk_vectors = spec.on_disk_vectors
    row.on_disk_payload = spec.on_disk_payload
    row.sparse_enabled = spec.sparse_enabled
    if point_count:
        row.point_count = point_count
    session.flush()
    return row


def list_collections(session: Session) -> list[CollectionRow]:
    return list(
        session.execute(select(CollectionRow).order_by(CollectionRow.version.desc()))
        .scalars()
        .all()
    )


def get_collection(session: Session, name: str) -> CollectionRow | None:
    return session.get(CollectionRow, name)


def get_active_collection(session: Session) -> CollectionRow | None:
    return session.execute(
        select(CollectionRow).where(CollectionRow.is_active.is_(True))
    ).scalar_one_or_none()


def set_active_collection(session: Session, name: str) -> None:
    """Mirror the Qdrant alias swap in the registry — exactly one row is active."""
    for row in list_collections(session):
        row.is_active = row.name == name


def next_version(session: Session) -> int:
    versions = list(session.execute(select(CollectionRow.version)).scalars().all())
    return (max(versions) + 1) if versions else 1


def set_point_count(session: Session, name: str, count: int) -> None:
    row = session.get(CollectionRow, name)
    if row is not None:
        row.point_count = count


# --------------------------------------------------------------------------- #
# Index jobs
# --------------------------------------------------------------------------- #
def create_index_job(
    session: Session,
    job_id: str,
    collection: str,
    files: list[NewFile],
    *,
    kind: JobKind = JobKind.INDEX,
    source_collection: str | None = None,
    reason: str | None = None,
    config_diff: dict[str, Any] | None = None,
    activate_on_success: bool = False,
) -> IndexJob:
    job = IndexJob(
        id=job_id,
        kind=kind,
        collection=collection,
        total_files=len(files),
        source_collection=source_collection,
        reason=reason,
        config_diff=config_diff,
        activate_on_success=activate_on_success,
    )
    session.add(job)
    for spec in files:
        session.add(
            IndexFile(
                id=new_id(),
                job_id=job_id,
                filename=spec.filename,
                input_path=spec.input_path,
                status=FileStatus.QUEUED,
            )
        )
    session.flush()
    return job


def get_index_job(session: Session, job_id: str) -> IndexJob | None:
    return session.execute(
        select(IndexJob).where(IndexJob.id == job_id).options(selectinload(IndexJob.files))
    ).scalar_one_or_none()


def list_index_jobs(session: Session, limit: int = 50) -> list[IndexJob]:
    return list(
        session.execute(
            select(IndexJob)
            .options(selectinload(IndexJob.files))
            .order_by(IndexJob.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )


def mark_file_running(session: Session, file_id: str, total_children: int = 0) -> None:
    row = session.get(IndexFile, file_id)
    if row is None:
        return
    row.status = FileStatus.RUNNING
    row.total_children = total_children
    row.processed_children = 0
    row.progress = 0
    _refresh_job_status(session, row.job_id)


def update_file_progress(session: Session, file_id: str, processed: int, total: int) -> None:
    row = session.get(IndexFile, file_id)
    if row is None:
        return
    row.total_children = total
    row.processed_children = processed
    row.progress = child_percent(processed, total)


def mark_file_result(
    session: Session,
    file_id: str,
    *,
    status: FileStatus,
    total_chunks: int = 0,
    children_indexed: int = 0,
    parents_stored: int = 0,
    skipped: int = 0,
    seconds: float = 0.0,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> None:
    row = session.get(IndexFile, file_id)
    if row is None:
        return
    row.status = status
    row.total_chunks = total_chunks
    row.processed_children = children_indexed
    row.total_children = max(row.total_children, children_indexed)
    row.progress = 100
    row.parents_stored = parents_stored
    row.skipped = skipped
    row.seconds = seconds
    row.warnings = json.dumps(warnings) if warnings else None
    row.error = error
    _refresh_job_status(session, row.job_id)


def fail_job(session: Session, job_id: str, error: str) -> None:
    job = session.get(IndexJob, job_id)
    if job is not None:
        job.error = error
    _refresh_job_status(session, job_id)


def refresh_job_status(session: Session, job_id: str) -> None:
    _refresh_job_status(session, job_id)


def _refresh_job_status(session: Session, job_id: str) -> None:
    job = session.get(IndexJob, job_id)
    if job is None:
        return
    files = list(
        session.execute(select(IndexFile).where(IndexFile.job_id == job_id)).scalars().all()
    )
    job.status = compute_job_status([f.status for f in files])


# --------------------------------------------------------------------------- #
# Eval runs
# --------------------------------------------------------------------------- #
def create_eval_run(
    session: Session,
    run_id: str,
    *,
    collection: str,
    mode: str,
    top_k: int,
    total_questions: int,
    config_snapshot: dict[str, Any],
) -> EvalRun:
    run = EvalRun(
        id=run_id,
        collection=collection,
        mode=mode,
        top_k=top_k,
        total_questions=total_questions,
        config_snapshot=config_snapshot,
    )
    session.add(run)
    session.flush()
    return run


def get_eval_run(session: Session, run_id: str) -> EvalRun | None:
    return session.execute(
        select(EvalRun).where(EvalRun.id == run_id).options(selectinload(EvalRun.items))
    ).scalar_one_or_none()


def list_eval_runs(session: Session, limit: int = 50) -> list[EvalRun]:
    return list(
        session.execute(
            select(EvalRun)
            .options(selectinload(EvalRun.items))
            .order_by(EvalRun.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )


def mark_eval_running(session: Session, run_id: str) -> None:
    run = session.get(EvalRun, run_id)
    if run is not None:
        run.status = EvalStatus.RUNNING


def finish_eval_run(
    session: Session,
    run_id: str,
    *,
    aggregate: dict[str, float],
    items: list[tuple[str, list[str], dict[str, float], str | None]],
    warnings: list[str],
    collection: str,
) -> None:
    run = session.get(EvalRun, run_id)
    if run is None:
        return
    run.status = EvalStatus.DONE
    run.collection = collection
    run.aggregate = dict(aggregate)
    run.warnings = json.dumps(warnings) if warnings else None
    run.completed_questions = len(items)
    run.progress = 100
    for position, (question, chunk_ids, metrics, note) in enumerate(items):
        session.add(
            EvalItem(
                id=new_id(),
                run_id=run_id,
                position=position,
                question=question,
                retrieved_chunk_ids=list(chunk_ids),
                metrics=dict(metrics),
                note=note,
            )
        )


def fail_eval_run(session: Session, run_id: str, error: str) -> None:
    run = session.get(EvalRun, run_id)
    if run is not None:
        run.status = EvalStatus.FAILED
        run.error = error
