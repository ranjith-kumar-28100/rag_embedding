"""FastAPI app (spec §11).

    POST /index                        upload chunk JSONL -> {index_job_id}
    GET  /index                        recent index/reindex jobs
    GET  /index/{id}                   progress (percent) + stats
    POST /search                       hybrid / dense / sparse retrieval
    POST /search/compare               dense vs sparse vs hybrid, same query
    POST /evaluate                     RAGAS run over a dataset
    GET  /evaluate, /evaluate/{id}     eval progress + metrics
    POST /reindex                      build a new collection, optional swap
    GET  /collections                  registry + which one the alias serves
    POST /collections/{name}/activate  manual alias swap / rollback
    GET  /health

Indexing / reindex / evaluate are async (Celery); search is synchronous. The API
only orchestrates: it saves uploads, writes rows, and dispatches — every embed
happens in a worker, so a 500 MB upload returns a job id immediately.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from rag_embedding.models import EvalQuestion, EvalRequest, SearchRequest, SearchResponse
from rag_embedding.store.qdrant_store import (
    CollectionNotFoundError,
    DimensionMismatchError,
)

from . import repository, tasks
from .config import get_settings
from .db import SessionLocal, init_db
from .deps import get_config, get_indexer, get_retriever, get_store
from .models import JobKind
from .schemas import (
    ActivateOut,
    CollectionOut,
    CollectionsOut,
    EvalCreatedOut,
    EvalRunOut,
    HealthOut,
    IndexJobCreatedOut,
    IndexJobOut,
    ReindexIn,
)

ALLOWED_SUFFIXES: tuple[str, ...] = (".jsonl", ".jsonl.gz", ".ndjson", ".json")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    init_db()
    yield


app = FastAPI(title="rag_embedding service", version="0.1.0", lifespan=lifespan)


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    cfg = get_config()
    qdrant_status, active = "down", None
    try:
        active = get_store().resolve_alias(cfg.alias)
        qdrant_status = "ok"
    except Exception as exc:  # noqa: BLE001 - health must report, not raise
        qdrant_status = f"error: {exc}"

    redis_status = "down"
    try:
        import redis

        redis.Redis.from_url(get_settings().redis_url).ping()
        redis_status = "ok"
    except Exception as exc:  # noqa: BLE001
        redis_status = f"error: {exc}"

    overall = "ok" if qdrant_status == "ok" and redis_status == "ok" else "degraded"
    return HealthOut(
        status=overall, qdrant=qdrant_status, redis=redis_status, active_collection=active
    )


# --------------------------------------------------------------------------- #
# Indexing (§7)
# --------------------------------------------------------------------------- #
def _target_collection(session: Session) -> tuple[str, int]:
    """Where new chunks go: the collection the alias serves, else a fresh v1.

    Re-using the serving collection is what makes ``POST /index`` incremental —
    uploading a second batch adds to the same index instead of shadowing it.
    """
    cfg = get_config()
    active = get_store().resolve_alias(cfg.alias)
    if active is not None:
        row = repository.get_collection(session, active)
        return active, (row.version if row else 1)
    version = repository.next_version(session)
    return cfg.physical_collection(version), version


@app.post("/index", response_model=IndexJobCreatedOut, status_code=202)
async def create_index_job(
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
) -> IndexJobCreatedOut:
    """Upload one or more stage-1 chunk files (.jsonl / .jsonl.gz)."""
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")

    cfg = get_config()
    settings = get_settings()
    max_bytes = cfg.max_upload_mb * 1024 * 1024

    job_id = repository.new_id()
    input_dir = settings.upload_dir / job_id
    input_dir.mkdir(parents=True, exist_ok=True)

    specs: list[repository.NewFile] = []
    for upload in files:
        name = Path(upload.filename or "chunks.jsonl").name
        if not name.lower().endswith(ALLOWED_SUFFIXES):
            raise HTTPException(
                status_code=415,
                detail=(
                    f"unsupported file {name!r}; expected stage-1 chunk JSONL "
                    f"({', '.join(ALLOWED_SUFFIXES)})"
                ),
            )
        data = await upload.read()
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"{name} exceeds the {cfg.max_upload_mb} MB limit"
            )
        dest = input_dir / name
        dest.write_bytes(data)
        specs.append(repository.NewFile(filename=name, input_path=str(dest)))

    collection, version = _target_collection(session)
    job = repository.create_index_job(session, job_id, collection, specs)
    file_tasks = [(f.id, f.input_path) for f in job.files]

    # Register the collection before any worker touches it, so /collections shows
    # the target while it is still being built.
    spec = get_indexer().collection_spec(collection)
    repository.upsert_collection(session, spec, version)
    if get_store().resolve_alias(cfg.alias) is None:
        # First-ever index: this collection will serve as soon as it exists.
        job.activate_on_success = True
        repository.set_active_collection(session, collection)
    session.commit()

    tasks.enqueue_index_job(job_id, collection, file_tasks)
    return IndexJobCreatedOut(
        index_job_id=job.id,
        status=job.status.value,
        collection=collection,
        total_files=job.total_files,
    )


@app.get("/index", response_model=list[IndexJobOut])
def list_index_jobs(
    limit: int = Query(50, ge=1, le=500), session: Session = Depends(get_session)
) -> list[IndexJobOut]:
    return [IndexJobOut.from_row(j) for j in repository.list_index_jobs(session, limit=limit)]


@app.get("/index/{job_id}", response_model=IndexJobOut)
def get_index_job(job_id: str, session: Session = Depends(get_session)) -> IndexJobOut:
    job = repository.get_index_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="index job not found")
    return IndexJobOut.from_row(job)


# --------------------------------------------------------------------------- #
# Search (§8)
# --------------------------------------------------------------------------- #
def _run_search(request: SearchRequest, collection: str | None) -> SearchResponse:
    try:
        return get_retriever(collection).search(request)
    except DimensionMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CollectionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"{exc} — index some chunks first (POST /index).",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, collection: str | None = Query(None)) -> SearchResponse:
    return _run_search(request, collection)


@app.post("/search/compare", response_model=dict[str, SearchResponse])
def search_compare(
    request: SearchRequest, collection: str | None = Query(None)
) -> dict[str, SearchResponse]:
    """Same query through all three modes — makes the hybrid benefit visible (§12)."""
    try:
        return get_retriever(collection).compare_modes(request)
    except DimensionMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Collections & reindex (§9, §11)
# --------------------------------------------------------------------------- #
@app.get("/collections", response_model=CollectionsOut)
def list_collections(session: Session = Depends(get_session)) -> CollectionsOut:
    cfg = get_config()
    try:
        active = get_store().resolve_alias(cfg.alias)
    except Exception:  # noqa: BLE001 - registry should still render if Qdrant is down
        active = None
    rows = repository.list_collections(session)
    return CollectionsOut(
        alias=cfg.alias,
        active_collection=active,
        collections=[CollectionOut.from_row(r, is_active=r.name == active) for r in rows],
    )


@app.post("/collections/{name}/activate", response_model=ActivateOut)
def activate_collection(name: str, session: Session = Depends(get_session)) -> ActivateOut:
    """Manual alias swap — also the rollback path after a bad reindex."""
    store = get_store()
    if not store.collection_exists(name):
        raise HTTPException(status_code=404, detail=f"collection {name!r} does not exist")
    cfg = get_config()
    store.set_alias(cfg.alias, name)
    repository.set_active_collection(session, name)
    session.commit()
    return ActivateOut(alias=cfg.alias, collection=name)


@app.post("/reindex", response_model=IndexJobCreatedOut, status_code=202)
def reindex(body: ReindexIn, session: Session = Depends(get_session)) -> IndexJobCreatedOut:
    """Build a new collection from the current one under the *current* config.

    Nothing is re-uploaded: child text lives in the Qdrant payload and parents in
    the docstore. The old collection keeps serving until the alias swaps (§9).
    """
    cfg = get_config()
    store = get_store()
    source = body.source_collection or store.resolve_alias(cfg.alias)
    if source is None:
        raise HTTPException(status_code=409, detail="nothing to reindex: no active collection")
    if not store.collection_exists(source):
        raise HTTPException(status_code=404, detail=f"collection {source!r} does not exist")

    version = repository.next_version(session)
    target = cfg.physical_collection(version)
    spec = get_indexer().collection_spec(target)
    repository.upsert_collection(session, spec, version)

    old_row = repository.get_collection(session, source)
    job_id = repository.new_id()
    job = repository.create_index_job(
        session,
        job_id,
        target,
        [repository.NewFile(filename=f"reindex from {source}", input_path=source)],
        kind=JobKind.REINDEX,
        source_collection=source,
        reason=body.reason,
        config_diff=_config_diff(old_row, spec),
        activate_on_success=body.activate_on_success,
    )
    file_id = job.files[0].id
    session.commit()

    tasks.enqueue_reindex(job_id, file_id, source, target)
    return IndexJobCreatedOut(
        index_job_id=job_id, status=job.status.value, collection=target, total_files=1
    )


def _config_diff(old: Any, new: Any) -> dict[str, Any]:
    """Old-vs-new snapshot stored on the job row for auditability (§9)."""
    fields = (
        "provider",
        "model_id",
        "dim",
        "hnsw_m",
        "hnsw_ef_construct",
        "quantization",
        "on_disk_vectors",
        "on_disk_payload",
        "sparse_enabled",
    )
    diff: dict[str, Any] = {}
    for field in fields:
        before = getattr(old, field, None)
        after = getattr(new, field, None)
        if before != after:
            diff[field] = {"from": before, "to": after}
    return diff


# --------------------------------------------------------------------------- #
# Evaluation (§10)
# --------------------------------------------------------------------------- #
@app.post("/evaluate", response_model=EvalCreatedOut, status_code=202)
async def create_eval_run(
    file: UploadFile | None = File(None),
    dataset: str | None = Form(None),
    mode: str = Form("hybrid"),
    top_k: int | None = Form(None),
    rerank: bool | None = Form(None),
    llm_metrics: bool = Form(False),
    collection: str | None = Form(None),
    session: Session = Depends(get_session),
) -> EvalCreatedOut:
    """Upload an eval set (JSON array or JSONL) and score retrieval against it."""
    raw = await file.read() if file is not None else (dataset or "").encode()
    questions = _parse_eval_dataset(raw, filename=file.filename if file else None)
    if not questions:
        raise HTTPException(status_code=400, detail="eval dataset is empty")

    cfg = get_config()
    request = EvalRequest(
        dataset=questions,
        top_k=top_k or cfg.top_k,
        mode=mode,
        rerank=rerank,
        llm_metrics=llm_metrics,
    )
    target = collection or get_store().resolve_alias(cfg.alias) or cfg.alias

    run_id = repository.new_id()
    repository.create_eval_run(
        session,
        run_id,
        collection=target,
        mode=request.mode,
        top_k=request.top_k or cfg.top_k,
        total_questions=len(questions),
        config_snapshot=_config_snapshot(),
    )
    session.commit()

    tasks.enqueue_evaluate(run_id, request, collection)
    return EvalCreatedOut(eval_run_id=run_id, status="queued", total_questions=len(questions))


@app.get("/evaluate", response_model=list[EvalRunOut])
def list_eval_runs(
    limit: int = Query(50, ge=1, le=500), session: Session = Depends(get_session)
) -> list[EvalRunOut]:
    return [
        EvalRunOut.from_row(r, include_items=False)
        for r in repository.list_eval_runs(session, limit=limit)
    ]


@app.get("/evaluate/{run_id}", response_model=EvalRunOut)
def get_eval_run(run_id: str, session: Session = Depends(get_session)) -> EvalRunOut:
    run = repository.get_eval_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="eval run not found")
    return EvalRunOut.from_row(run)


def _parse_eval_dataset(raw: bytes, *, filename: str | None) -> list[EvalQuestion]:
    """Accept a JSON array or JSONL, plain or gzipped."""
    if filename and filename.endswith(".gz"):
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8").strip()
    if not text:
        return []
    try:
        if text.startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"eval dataset is not valid JSON: {exc}") from exc
    return [EvalQuestion.model_validate(row) for row in rows]


def _config_snapshot() -> dict[str, Any]:
    """The knobs that change what retrieval returns — stored with every run."""
    cfg = get_config()
    return {
        "embedding_provider": cfg.embedding_provider,
        "local_embedding_model": cfg.local_embedding_model,
        "sparse_enabled": cfg.sparse_enabled,
        "fusion": cfg.fusion,
        "top_k": cfg.top_k,
        "child_prefetch_k": cfg.child_prefetch_k,
        "with_parents": cfg.with_parents,
        "hnsw_ef": cfg.hnsw_ef,
        "quantization": cfg.quantization,
        "rerank_enabled": cfg.rerank_enabled,
        "rerank_model": cfg.rerank_model if cfg.rerank_enabled else None,
        "rerank_candidates": cfg.rerank_candidates,
    }
