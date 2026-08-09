"""Opt-in round-trip against a REAL Qdrant container (spec §17).

    make qdrant                 # docker compose up -d qdrant
    make test-integration

Skipped unless ``RAG_EMBED_TEST_QDRANT_URL`` is set. These are the assertions the
in-process client cannot make: quantization and on-disk settings actually applied
to the collection, payload indexes actually created, and HNSW search returning
the right neighbour from a real index.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient

from rag_embedding.config import EmbeddingConfig
from rag_embedding.embedders.sparse_bm25 import Bm25Encoder
from rag_embedding.indexer import Indexer
from rag_embedding.models import Chunk, SearchFilters, SearchRequest
from rag_embedding.rerank import build_reranker
from rag_embedding.retrieval import Retriever
from rag_embedding.store.docstore import InMemoryParentDocStore
from rag_embedding.store.qdrant_store import QdrantStore

from conftest import (
    FakeDenseEmbedder,
    dense_params,
    integration_url,
    requires_qdrant,
    sparse_params,
)

pytestmark = [pytest.mark.integration, requires_qdrant]


@pytest.fixture
def live_prefix() -> str:
    return f"itest_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def live(live_prefix: str) -> Iterator[tuple[Indexer, Retriever, QdrantStore]]:
    url = integration_url()
    assert url is not None
    client = QdrantClient(url=url, timeout=60)
    cfg = EmbeddingConfig(
        qdrant_url=url, collection_prefix=live_prefix, embedding_batch_size=2, top_k=5,
        child_prefetch_k=20,
    )
    store = QdrantStore(client, cfg)
    dense, sparse, docstore = FakeDenseEmbedder(), Bm25Encoder(), InMemoryParentDocStore()
    indexer = Indexer(cfg=cfg, store=store, dense=dense, sparse=sparse, docstore=docstore)
    retriever = Retriever(
        cfg=cfg, store=store, dense=dense, sparse=sparse, docstore=docstore,
        collection=cfg.alias,
    )
    try:
        yield indexer, retriever, store
    finally:
        for name in (cfg.physical_collection(1), cfg.physical_collection(2)):
            if store.collection_exists(name):
                store.delete_collection(name)
        client.close()


def test_index_search_round_trip(
    live: tuple[Indexer, Retriever, QdrantStore], sample_chunks: list[Chunk], live_prefix: str
) -> None:
    indexer, retriever, store = live
    v1 = f"{live_prefix}_v1"

    indexer.ensure_collection(v1)
    stats = indexer.index_chunks(v1, sample_chunks)
    assert stats.children_indexed == 4
    indexer.activate(v1)

    response = retriever.search(SearchRequest(query="dense and sparse fusion"))
    assert response.hits[0].chunk_id == "c1"
    assert response.parents[0].chunk_id == "p1"

    lexical = retriever.search(SearchRequest(query="1001", mode="sparse", with_parents=False))
    assert [h.chunk_id for h in lexical.hits] == ["c4"]

    filtered = retriever.search(
        SearchRequest(
            query="revenue", filters=SearchFilters(doc_id=["doc2"]), with_parents=False
        )
    )
    assert {h.payload["doc_id"] for h in filtered.hits} == {"doc2"}


def test_storage_settings_are_really_applied(
    live: tuple[Indexer, Retriever, QdrantStore], sample_chunks: list[Chunk], live_prefix: str
) -> None:
    indexer, _retriever, store = live
    v1 = f"{live_prefix}_v1"
    indexer.ensure_collection(v1)
    indexer.index_chunks(v1, sample_chunks)

    info = store.client.get_collection(v1)
    dense = dense_params(info)
    assert dense.on_disk is True
    assert dense.hnsw_config.m == 16
    assert dense.hnsw_config.ef_construct == 128
    assert dense.quantization_config.scalar.type == "int8"
    assert dense.quantization_config.scalar.always_ram is True
    assert info.config.params.on_disk_payload is True
    assert sparse_params(info).modifier == "idf"
    assert set(info.payload_schema) >= {"doc_id", "file_type", "block_type", "section_path"}


def test_reindex_and_zero_downtime_alias_swap(
    live: tuple[Indexer, Retriever, QdrantStore], sample_chunks: list[Chunk], live_prefix: str
) -> None:
    indexer, retriever, store = live
    v1, v2 = f"{live_prefix}_v1", f"{live_prefix}_v2"

    indexer.ensure_collection(v1)
    indexer.index_chunks(v1, sample_chunks)
    indexer.activate(v1)

    indexer.reindex_from(v1, v2)
    # v1 still serves while v2 is built
    assert store.resolve_alias(live_prefix) == v1
    assert retriever.search(SearchRequest(query="fusion")).hits

    indexer.activate(v2)
    assert store.resolve_alias(live_prefix) == v2
    assert retriever.search(SearchRequest(query="fusion")).hits[0].chunk_id == "c1"


@pytest.mark.skipif(
    os.environ.get("RAG_EMBED_TEST_RERANK") != "1",
    reason="set RAG_EMBED_TEST_RERANK=1 to load the real cross-encoder (~280 MB download)",
)
def test_real_cross_encoder_scores_and_reorders(
    live: tuple[Indexer, Retriever, QdrantStore], sample_chunks: list[Chunk], live_prefix: str
) -> None:
    """The one test that actually loads BAAI/bge-reranker-base.

    Everything else fakes the scorer, because what needs asserting elsewhere is
    the wiring. This asserts the model loads offline-cacheably and produces a
    usable ordering.
    """
    indexer, _retriever, store = live
    v1 = f"{live_prefix}_v1"
    indexer.ensure_collection(v1)
    indexer.index_chunks(v1, sample_chunks)
    indexer.activate(v1)

    cfg = indexer.cfg.model_copy(update={"rerank_enabled": True, "rerank_candidates": 10})
    reranker = build_reranker(cfg)
    assert reranker is not None
    retriever = Retriever(
        cfg=cfg,
        store=store,
        dense=indexer.dense,
        sparse=indexer.sparse,
        docstore=indexer.docstore,
        collection=cfg.alias,
        reranker=reranker,
    )

    response = retriever.search(
        SearchRequest(query="what were the quarterly revenue figures?", top_k=3)
    )
    assert response.reranked is True
    assert response.timings["rerank_ms"] > 0
    assert response.hits[0].chunk_id == "c3"  # the revenue table
    assert all(h.retrieval_score is not None for h in response.hits)
    assert all(h.source.endswith("+rerank") for h in response.hits)
    scores = [h.score for h in response.hits]
    assert scores == sorted(scores, reverse=True)


def test_idempotent_upsert_against_a_real_index(
    live: tuple[Indexer, Retriever, QdrantStore], sample_chunks: list[Chunk], live_prefix: str
) -> None:
    indexer, _retriever, store = live
    v1 = f"{live_prefix}_v1"
    indexer.ensure_collection(v1)
    indexer.index_chunks(v1, sample_chunks)
    indexer.index_chunks(v1, sample_chunks)
    assert store.count(v1) == 4
