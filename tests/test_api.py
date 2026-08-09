"""API flow with Celery replaced by in-process task execution.

Redis and the broker are out of the picture: ``enqueue_*`` is monkeypatched to
run the *real* task bodies eagerly via ``.apply()``, so the DB writes, progress
updates, alias swap and eval persistence are all genuinely exercised — only the
transport is faked. Qdrant is in-process and the dense model is the fake.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from rag_embedding.models import EvalRequest

from conftest import FakeDenseEmbedder

SPEC_SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_chunks.jsonl"


@pytest.fixture
def client(
    service_db: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[TestClient]:
    monkeypatch.setenv("RAG_EMBED_COLLECTION_PREFIX", "test_rag")
    monkeypatch.setenv("RAG_EMBED_EMBEDDING_BATCH_SIZE", "2")
    monkeypatch.setenv("RAG_EMBED_TOP_K", "5")
    monkeypatch.setenv("RAG_EMBED_CHILD_PREFETCH_K", "20")

    from service_embed import api, deps, tasks

    deps.reset()
    shared_qdrant = QdrantClient(":memory:")
    monkeypatch.setattr(deps, "build_client", lambda _cfg: shared_qdrant)
    monkeypatch.setattr(deps, "build_dense_embedder", lambda _cfg: FakeDenseEmbedder())

    def fake_index(job_id: str, collection: str, file_tasks: list[tuple[str, str]]) -> None:
        results = [
            tasks.index_file_task.apply(args=[job_id, fid, collection, path]).get()
            for fid, path in file_tasks
        ]
        tasks.finalize_index_job.apply(args=[results, job_id]).get()

    def fake_reindex(job_id: str, file_id: str, source: str, target: str) -> None:
        result = tasks.reindex_task.apply(args=[job_id, file_id, source, target]).get()
        tasks.finalize_index_job.apply(args=[[result], job_id]).get()

    def fake_evaluate(run_id: str, request: EvalRequest, collection: str | None) -> None:
        tasks.evaluate_task.apply(args=[run_id, request.model_dump_json(), collection]).get()

    monkeypatch.setattr(tasks, "enqueue_index_job", fake_index)
    monkeypatch.setattr(tasks, "enqueue_reindex", fake_reindex)
    monkeypatch.setattr(tasks, "enqueue_evaluate", fake_evaluate)

    with TestClient(api.app) as test_client:
        yield test_client
    deps.reset()
    shared_qdrant.close()


def _index_sample(client: TestClient, name: str = "chunks.jsonl") -> dict[str, Any]:
    resp = client.post("/index", files=[("files", (name, SPEC_SAMPLE.read_bytes()))])
    assert resp.status_code == 202, resp.text
    return dict(resp.json())


# --------------------------------------------------------------------------- #
# Health & indexing
# --------------------------------------------------------------------------- #
def test_health_reports_each_dependency(client: TestClient) -> None:
    body = client.get("/health").json()
    assert set(body) == {"status", "qdrant", "redis", "active_collection", "reranker"}
    assert body["qdrant"] == "ok"  # in-process Qdrant is reachable


def test_health_reports_no_reranker_by_default(client: TestClient) -> None:
    """The UI reads this to decide whether offering `rerank: true` can work."""
    assert client.get("/health").json()["reranker"] is None


def test_health_reports_the_reranker_when_enabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from service_embed import deps

    monkeypatch.setenv("RAG_EMBED_RERANK_ENABLED", "true")
    deps.get_config.cache_clear()
    deps.get_reranker.cache_clear()
    assert client.get("/health").json()["reranker"] == "cross_encoder:BAAI/bge-reranker-base"
    deps.get_config.cache_clear()
    deps.get_reranker.cache_clear()


def test_index_upload_indexes_and_activates(client: TestClient) -> None:
    created = _index_sample(client)
    assert created["collection"] == "test_rag_v1"

    job = client.get(f"/index/{created['index_job_id']}").json()
    assert job["status"] == "done"
    assert job["progress"] == 100
    assert job["children_indexed"] == 4
    assert job["parents_stored"] == 1
    # Progress starts against the line count (5) and is corrected to the real
    # child count (4) as soon as the roles are split.
    assert job["files"][0]["total_children"] == 4
    assert job["files"][0]["processed_children"] == 4

    # first index takes over the alias, so search works immediately
    assert client.get("/health").json()["active_collection"] == "test_rag_v1"


def test_second_upload_adds_to_the_same_collection(client: TestClient) -> None:
    _index_sample(client)
    second = _index_sample(client, name="more.jsonl")
    assert second["collection"] == "test_rag_v1"
    registry = client.get("/collections").json()
    assert len(registry["collections"]) == 1


def test_gzipped_upload_accepted(client: TestClient) -> None:
    import gzip

    payload = gzip.compress(SPEC_SAMPLE.read_bytes())
    resp = client.post("/index", files=[("files", ("chunks.jsonl.gz", payload))])
    assert resp.status_code == 202
    job = client.get(f"/index/{resp.json()['index_job_id']}").json()
    assert job["children_indexed"] == 4


def test_unsupported_extension_rejected(client: TestClient) -> None:
    resp = client.post("/index", files=[("files", ("report.pdf", b"%PDF-1.4"))])
    assert resp.status_code == 415
    assert "chunk JSONL" in resp.json()["detail"]


def test_oversized_upload_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from service_embed import deps

    monkeypatch.setenv("RAG_EMBED_MAX_UPLOAD_MB", "0")
    deps.get_config.cache_clear()
    resp = client.post("/index", files=[("files", ("chunks.jsonl", b"x" * 1024))])
    assert resp.status_code == 413
    deps.get_config.cache_clear()


def test_malformed_jsonl_fails_the_file_not_the_request(client: TestClient) -> None:
    resp = client.post("/index", files=[("files", ("bad.jsonl", b"{not json\n"))])
    assert resp.status_code == 202
    job = client.get(f"/index/{resp.json()['index_job_id']}").json()
    assert job["status"] == "failed"
    assert job["files"][0]["error"]


def test_index_job_404(client: TestClient) -> None:
    assert client.get("/index/nope").status_code == 404


def test_index_job_listing(client: TestClient) -> None:
    _index_sample(client)
    listing = client.get("/index").json()
    assert len(listing) == 1
    assert listing[0]["kind"] == "index"


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def test_search_returns_hits_and_parents(client: TestClient) -> None:
    _index_sample(client)
    body = client.post(
        "/search", json={"query": "dense and sparse reciprocal rank fusion"}
    ).json()
    assert body["mode"] == "hybrid"
    assert body["hits"][0]["chunk_id"] == "a1b2c3"
    assert body["parents"][0]["chunk_id"] == "7b1e0parent"
    assert "search_ms" in body["timings"]


def test_search_modes(client: TestClient) -> None:
    _index_sample(client)
    for mode in ("dense", "sparse", "hybrid"):
        body = client.post("/search", json={"query": "revenue", "mode": mode}).json()
        assert body["mode"] == mode


def test_search_filters(client: TestClient) -> None:
    _index_sample(client)
    body = client.post(
        "/search",
        json={"query": "revenue", "filters": {"file_type": ["xlsx"]}, "with_parents": False},
    ).json()
    assert {h["payload"]["file_type"] for h in body["hits"]} == {"xlsx"}


def test_rerank_is_off_by_default_and_reported(client: TestClient) -> None:
    _index_sample(client)
    body = client.post("/search", json={"query": "revenue"}).json()
    assert body["reranked"] is False
    assert all(h["retrieval_score"] is None for h in body["hits"])


def test_requesting_rerank_without_one_configured_is_400(client: TestClient) -> None:
    _index_sample(client)
    resp = client.post("/search", json={"query": "revenue", "rerank": True})
    assert resp.status_code == 400
    assert "RAG_EMBED_RERANK_ENABLED" in resp.json()["detail"]


def test_rerank_reorders_and_widens_candidates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through the API with a stand-in cross-encoder."""
    from service_embed import deps

    from test_rerank import FakeReranker

    _index_sample(client)
    reranker = FakeReranker()
    deps.get_reranker.cache_clear()
    monkeypatch.setattr(deps, "build_reranker", lambda _cfg: reranker)
    monkeypatch.setenv("RAG_EMBED_RERANK_ENABLED", "true")
    deps.get_config.cache_clear()

    body = client.post("/search", json={"query": "quarterly revenue", "top_k": 2}).json()
    assert body["reranked"] is True
    assert "rerank_ms" in body["timings"]
    assert len(body["hits"]) == 2
    assert body["hits"][0]["retrieval_score"] is not None
    assert body["hits"][0]["source"].endswith("+rerank")
    # more candidates were scored than were returned
    assert reranker.calls[0][1] > 2

    deps.get_reranker.cache_clear()
    deps.get_config.cache_clear()


def test_eval_run_records_the_rerank_setting(client: TestClient) -> None:
    _index_sample(client)
    created = client.post(
        "/evaluate", data={"dataset": json.dumps(EVAL_SET), "rerank": "false"}
    ).json()
    run = client.get(f"/evaluate/{created['eval_run_id']}").json()
    assert run["status"] == "done"
    assert run["config_snapshot"]["rerank_enabled"] is False


def test_search_compare_returns_all_modes(client: TestClient) -> None:
    _index_sample(client)
    body = client.post("/search/compare", json={"query": "revenue"}).json()
    assert set(body) == {"dense", "sparse", "hybrid"}


def test_search_before_any_index_is_404(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "anything"})
    assert resp.status_code == 404
    assert "POST /index" in resp.json()["detail"]


def test_unknown_filter_field_rejected(client: TestClient) -> None:
    _index_sample(client)
    resp = client.post("/search", json={"query": "x", "filters": {"author": ["me"]}})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Collections & reindex
# --------------------------------------------------------------------------- #
def test_collections_registry(client: TestClient) -> None:
    _index_sample(client)
    body = client.get("/collections").json()
    assert body["alias"] == "test_rag"
    assert body["active_collection"] == "test_rag_v1"
    row = body["collections"][0]
    assert row["is_active"] is True
    assert row["model_id"] == "fake:fake-embedder"
    assert row["dim"] == 32
    assert row["point_count"] == 4
    assert row["quantization"] == "int8"


def test_reindex_builds_v2_without_swapping(client: TestClient) -> None:
    _index_sample(client)
    created = client.post("/reindex", json={"reason": "trying a bigger model"}).json()
    assert created["collection"] == "test_rag_v2"

    job = client.get(f"/index/{created['index_job_id']}").json()
    assert job["kind"] == "reindex"
    assert job["status"] == "done"
    assert job["reason"] == "trying a bigger model"

    # zero downtime: the alias still points at v1 until someone activates v2
    assert client.get("/collections").json()["active_collection"] == "test_rag_v1"


def test_reindex_can_auto_activate(client: TestClient) -> None:
    _index_sample(client)
    client.post("/reindex", json={"activate_on_success": True})
    assert client.get("/collections").json()["active_collection"] == "test_rag_v2"


def test_reindex_records_a_config_diff(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _index_sample(client)
    from service_embed import deps

    monkeypatch.setenv("RAG_EMBED_HNSW_M", "32")
    deps.reset()
    monkeypatch.setattr(deps, "build_dense_embedder", lambda _cfg: FakeDenseEmbedder())
    created = client.post("/reindex", json={"reason": "denser graph"}).json()
    job = client.get(f"/index/{created['index_job_id']}").json()
    assert job["config_diff"]["hnsw_m"] == {"from": 16, "to": 32}


def test_reindex_without_an_active_collection_is_409(client: TestClient) -> None:
    assert client.post("/reindex", json={}).status_code == 409


def test_manual_activate_and_rollback(client: TestClient) -> None:
    _index_sample(client)
    client.post("/reindex", json={"activate_on_success": True})
    assert client.get("/collections").json()["active_collection"] == "test_rag_v2"

    out = client.post("/collections/test_rag_v1/activate").json()
    assert out == {"alias": "test_rag", "collection": "test_rag_v1"}
    assert client.get("/collections").json()["active_collection"] == "test_rag_v1"


def test_activating_a_missing_collection_is_404(client: TestClient) -> None:
    assert client.post("/collections/test_rag_v9/activate").status_code == 404


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
EVAL_SET = [
    {
        "question": "how are dense and sparse signals combined?",
        "ground_truth_contexts": ["7b1e0parent"],
    },
    {"question": "what was Q2 revenue?", "ground_truth_contexts": ["t4b1e0"]},
]


def test_evaluate_scores_and_persists(client: TestClient) -> None:
    _index_sample(client)
    created = client.post(
        "/evaluate",
        files={"file": ("eval.json", json.dumps(EVAL_SET).encode())},
        data={"mode": "hybrid", "top_k": "5"},
    ).json()

    run = client.get(f"/evaluate/{created['eval_run_id']}").json()
    assert run["status"] == "done"
    assert run["total_questions"] == 2
    assert run["aggregate"]["hit_rate@5"] == 1.0
    assert len(run["items"]) == 2
    assert run["config_snapshot"]["fusion"] == "rrf"


def test_evaluate_accepts_jsonl(client: TestClient) -> None:
    _index_sample(client)
    payload = "\n".join(json.dumps(row) for row in EVAL_SET).encode()
    created = client.post(
        "/evaluate", files={"file": ("eval.jsonl", payload)}, data={"mode": "dense"}
    ).json()
    assert created["total_questions"] == 2


def test_evaluate_accepts_an_inline_dataset(client: TestClient) -> None:
    _index_sample(client)
    created = client.post("/evaluate", data={"dataset": json.dumps(EVAL_SET)}).json()
    assert created["total_questions"] == 2


def test_empty_eval_set_rejected(client: TestClient) -> None:
    _index_sample(client)
    assert client.post("/evaluate", data={"dataset": "[]"}).status_code == 400


def test_invalid_eval_json_rejected(client: TestClient) -> None:
    _index_sample(client)
    resp = client.post("/evaluate", files={"file": ("eval.json", b"{nope")})
    assert resp.status_code == 400


def test_eval_run_listing_omits_items(client: TestClient) -> None:
    _index_sample(client)
    client.post("/evaluate", data={"dataset": json.dumps(EVAL_SET)})
    listing = client.get("/evaluate").json()
    assert len(listing) == 1
    assert listing[0]["items"] == []
    assert listing[0]["aggregate"]


def test_eval_run_404(client: TestClient) -> None:
    assert client.get("/evaluate/nope").status_code == 404


def test_eval_can_target_a_candidate_collection(client: TestClient) -> None:
    """Score v2 before swapping the alias to it — the §12 before/after workflow."""
    _index_sample(client)
    client.post("/reindex", json={})
    created = client.post(
        "/evaluate",
        data={"dataset": json.dumps(EVAL_SET), "collection": "test_rag_v2"},
    ).json()
    run = client.get(f"/evaluate/{created['eval_run_id']}").json()
    assert run["collection"] == "test_rag_v2"
    assert run["status"] == "done"
