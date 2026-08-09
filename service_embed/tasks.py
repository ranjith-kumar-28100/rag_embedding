"""Celery tasks: index / reindex / evaluate.

The library does the work; these tasks only translate between Celery, the DB and
``Indexer`` / ``RagasEvaluator``. Per-file failures never raise out of a task, so
one bad JSONL cannot poison a batch — the file is marked FAILED and the job rolls
up to PARTIAL.
"""

from __future__ import annotations

import logging
from typing import Any

from celery import chord

from rag_embedding.models import EvalRequest

from .celery_app import celery_app
from .db import session_scope
from .deps import get_evaluator, get_indexer, get_store
from .models import FileStatus, IndexJob
from .repository import (
    fail_eval_run,
    fail_job,
    finish_eval_run,
    mark_eval_running,
    mark_file_result,
    mark_file_running,
    refresh_job_status,
    set_active_collection,
    set_point_count,
    update_file_progress,
)

logger = logging.getLogger("rag_embedding.service.tasks")

# Progress is written at most this often (in children) to keep a big file from
# turning into thousands of tiny SQLite writes.
PROGRESS_WRITE_EVERY = 200


def _progress_writer(file_id: str) -> Any:
    last_written = {"n": -1}

    def write(processed: int, total: int) -> None:
        if processed != total and processed - last_written["n"] < PROGRESS_WRITE_EVERY:
            return
        last_written["n"] = processed
        with session_scope() as session:
            update_file_progress(session, file_id, processed, total)

    return write


@celery_app.task(name="service_embed.index_file", bind=True)  # type: ignore[untyped-decorator]
def index_file_task(
    self: Any, job_id: str, file_id: str, collection: str, input_path: str
) -> str:
    """Index one chunk JSONL into ``collection``. Never raises."""
    from rag_embedding.chunks import count_lines

    try:
        indexer = get_indexer()
        indexer.ensure_collection(collection)
        with session_scope() as session:
            mark_file_running(session, file_id, total_children=count_lines(input_path))

        stats = indexer.index_file(collection, input_path, progress=_progress_writer(file_id))

        with session_scope() as session:
            mark_file_result(
                session,
                file_id,
                status=FileStatus.DONE,
                total_chunks=stats.total_chunks,
                children_indexed=stats.children_indexed,
                parents_stored=stats.parents_stored,
                skipped=stats.skipped,
                seconds=stats.seconds,
                warnings=stats.warnings,
            )
        return FileStatus.DONE.value
    except Exception as exc:  # noqa: BLE001 - a task must not crash the batch
        logger.exception("index_file_task failed job=%s file=%s", job_id, file_id)
        with session_scope() as session:
            mark_file_result(session, file_id, status=FileStatus.FAILED, error=str(exc))
        return FileStatus.FAILED.value


@celery_app.task(name="service_embed.finalize_index_job")  # type: ignore[untyped-decorator]
def finalize_index_job(_results: list[str], job_id: str) -> str:
    """Chord callback: roll up status, record point count, swap the alias if asked."""
    with session_scope() as session:
        refresh_job_status(session, job_id)
        job = session.get(IndexJob, job_id)
        if job is None:
            return "unknown"
        collection, activate, status = job.collection, job.activate_on_success, job.status

    store = get_store()
    try:
        count = store.count(collection)
    except Exception:  # noqa: BLE001 - counting must not fail the job
        logger.warning("could not count %s", collection, exc_info=True)
        count = 0

    with session_scope() as session:
        set_point_count(session, collection, count)

    # Only swap when the build actually succeeded — a PARTIAL reindex must not
    # take over serving traffic (§9: the old collection keeps serving).
    if activate and status.value == "done":
        store.set_alias(get_indexer().cfg.alias, collection)
        with session_scope() as session:
            set_active_collection(session, collection)

    return status.value


@celery_app.task(name="service_embed.reindex", bind=True)  # type: ignore[untyped-decorator]
def reindex_task(
    self: Any, job_id: str, file_id: str, source_collection: str, target_collection: str
) -> str:
    """Rebuild ``target`` from ``source``'s stored payloads (§9). Never raises."""
    try:
        indexer = get_indexer()
        with session_scope() as session:
            mark_file_running(
                session, file_id, total_children=get_store().count(source_collection)
            )

        stats = indexer.reindex_from(
            source_collection, target_collection, progress=_progress_writer(file_id)
        )

        with session_scope() as session:
            mark_file_result(
                session,
                file_id,
                status=FileStatus.DONE,
                total_chunks=stats.total_chunks,
                children_indexed=stats.children_indexed,
                skipped=stats.skipped,
                seconds=stats.seconds,
                warnings=stats.warnings,
            )
        return FileStatus.DONE.value
    except Exception as exc:  # noqa: BLE001
        logger.exception("reindex_task failed job=%s", job_id)
        with session_scope() as session:
            mark_file_result(session, file_id, status=FileStatus.FAILED, error=str(exc))
            fail_job(session, job_id, str(exc))
        return FileStatus.FAILED.value


@celery_app.task(name="service_embed.evaluate")  # type: ignore[untyped-decorator]
def evaluate_task(run_id: str, request_json: str, collection: str | None) -> str:
    """Run the eval harness against the active (or a named) collection."""
    try:
        with session_scope() as session:
            mark_eval_running(session, run_id)

        request = EvalRequest.model_validate_json(request_json)
        report = get_evaluator(collection).evaluate(request)

        with session_scope() as session:
            finish_eval_run(
                session,
                run_id,
                collection=report.collection,
                aggregate=report.aggregate,
                warnings=report.warnings,
                items=[
                    (item.question, item.retrieved_chunk_ids, item.metrics, item.note)
                    for item in report.items
                ],
            )
        return "done"
    except Exception as exc:  # noqa: BLE001
        logger.exception("evaluate_task failed run=%s", run_id)
        with session_scope() as session:
            fail_eval_run(session, run_id, str(exc))
        return "failed"


# --------------------------------------------------------------------------- #
# Dispatch helpers — the API calls these, and tests monkeypatch them.
# --------------------------------------------------------------------------- #
def enqueue_index_job(job_id: str, collection: str, file_tasks: list[tuple[str, str]]) -> None:
    """``file_tasks`` is a list of ``(file_id, input_path)``."""
    header = [
        index_file_task.s(job_id, file_id, collection, input_path)
        for file_id, input_path in file_tasks
    ]
    chord(header)(finalize_index_job.s(job_id))


def enqueue_reindex(
    job_id: str, file_id: str, source_collection: str, target_collection: str
) -> None:
    chord([reindex_task.s(job_id, file_id, source_collection, target_collection)])(
        finalize_index_job.s(job_id)
    )


def enqueue_evaluate(run_id: str, request: EvalRequest, collection: str | None) -> None:
    evaluate_task.delay(run_id, request.model_dump_json(), collection)
