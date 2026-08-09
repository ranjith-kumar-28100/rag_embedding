"""Shared fixtures.

Every test is hermetic by default: a deterministic fake embedder (no model
download, no torch), an in-process Qdrant (``QdrantClient(":memory:")``, which
supports named vectors, sparse vectors, Query API fusion and aliases), and an
in-memory or throwaway-SQLite parent docstore.

The only tests that touch a real Qdrant are marked ``integration`` and skip
unless ``RAG_EMBED_TEST_QDRANT_URL`` is set.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Point the microservice at a throwaway DB/dirs BEFORE service modules import
# (service_embed.db builds its engine at import time).
os.environ.setdefault("RAG_EMBED_SVC_DATABASE_URL", "sqlite:///./data/test_rag_embedding.db")
os.environ.setdefault("RAG_EMBED_SVC_DATA_DIR", "./data/test")
os.environ.setdefault("RAG_EMBED_SVC_UPLOAD_DIR", "./data/test/uploads")

import pytest
from qdrant_client import QdrantClient

from rag_embedding.config import EmbeddingConfig
from rag_embedding.embedders.base import SparseVector
from rag_embedding.embedders.sparse_bm25 import Bm25Encoder, tokenize
from rag_embedding.indexer import Indexer
from rag_embedding.models import Chunk
from rag_embedding.retrieval import Retriever
from rag_embedding.store.docstore import InMemoryParentDocStore
from rag_embedding.store.qdrant_store import DENSE_VECTOR, SPARSE_VECTOR, QdrantStore

FIXTURE_COLLECTION = "test_rag_v1"


@pytest.fixture(autouse=True, scope="session")
def _ignore_dotenv() -> Iterator[None]:
    """Never read the developer's ``.env`` during tests.

    ``EmbeddingConfig`` loads ``.env`` by design, so a local file (e.g. from
    ``cp .env.example .env``) would otherwise leak into every config the suite
    builds and quietly change what is being asserted. Explicit env vars set by
    individual tests still work — this only removes the file.
    """
    original = EmbeddingConfig.model_config.get("env_file")
    EmbeddingConfig.model_config["env_file"] = None
    try:
        yield
    finally:
        EmbeddingConfig.model_config["env_file"] = original


class FakeDenseEmbedder:
    """Deterministic, content-derived, L2-normalized vectors.

    Built from a token bag rather than a hash of the whole string, so texts that
    *share words* land near each other. Retrieval tests need that: a pure hash
    would make every vector orthogonal and every ranking arbitrary.
    """

    def __init__(self, dim: int = 32, model_name: str = "fake-embedder") -> None:
        self._dim = dim
        self._model_name = model_name
        self.call_count = 0

    @property
    def model_id(self) -> str:
        return f"fake:{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in tokenize(text) or ["__empty__"]:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vector[digest[0] % self._dim] += 1.0
            vector[digest[1] % self._dim] += 0.5
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


@pytest.fixture
def fake_dense() -> FakeDenseEmbedder:
    return FakeDenseEmbedder()


@pytest.fixture
def bm25() -> Bm25Encoder:
    return Bm25Encoder()


@pytest.fixture
def cfg(tmp_path: Path) -> EmbeddingConfig:
    return EmbeddingConfig(
        collection_prefix="test_rag",
        embedding_batch_size=2,  # small, so batching is actually exercised
        top_k=5,
        child_prefetch_k=20,
        # Local-mode Qdrant ignores these, but keeping them realistic means the
        # spec objects the tests assert on are the ones production builds.
        quantization="int8",
    )


@pytest.fixture
def qdrant() -> Iterator[QdrantClient]:
    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture
def store(qdrant: QdrantClient, cfg: EmbeddingConfig) -> QdrantStore:
    return QdrantStore(qdrant, cfg)


@pytest.fixture
def docstore() -> InMemoryParentDocStore:
    return InMemoryParentDocStore()


@pytest.fixture
def indexer(
    cfg: EmbeddingConfig,
    store: QdrantStore,
    fake_dense: FakeDenseEmbedder,
    bm25: Bm25Encoder,
    docstore: InMemoryParentDocStore,
) -> Indexer:
    return Indexer(cfg=cfg, store=store, dense=fake_dense, sparse=bm25, docstore=docstore)


@pytest.fixture
def retriever(
    cfg: EmbeddingConfig,
    store: QdrantStore,
    fake_dense: FakeDenseEmbedder,
    bm25: Bm25Encoder,
    docstore: InMemoryParentDocStore,
) -> Retriever:
    return Retriever(
        cfg=cfg,
        store=store,
        dense=fake_dense,
        sparse=bm25,
        docstore=docstore,
        collection=FIXTURE_COLLECTION,
    )


# --------------------------------------------------------------------------- #
# Chunk fixtures
# --------------------------------------------------------------------------- #
def make_chunk(
    chunk_id: str,
    text: str,
    *,
    role: str = "child",
    parent_id: str | None = "p1",
    doc_id: str = "doc1",
    block_type: str = "text",
    file_type: str = "pdf",
    section_path: list[str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        parent_id=parent_id,
        role=role,
        block_type=block_type,
        text=text,
        section_path=section_path if section_path is not None else ["3. Architecture"],
        source_uri="file:///data/report.pdf",
        file_type=file_type,
        token_count=len(text.split()),
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """One parent plus four children spanning block types and documents."""
    return [
        make_chunk(
            "p1",
            "Retrieval fuses dense and sparse signals and returns parent context.",
            role="parent",
            parent_id=None,
        ),
        make_chunk(
            "c1",
            "The retrieval subsystem fuses dense and sparse BM25 signals with reciprocal rank fusion.",
        ),
        make_chunk("c2", "HNSW indexes approximate nearest neighbours over dense vectors."),
        make_chunk(
            "c3",
            "Quarterly revenue table: Q1 1.2M, Q2 1.5M",
            block_type="table",
            doc_id="doc2",
            file_type="xlsx",
            parent_id="p1",
            section_path=["Sheet1"],
        ),
        make_chunk(
            "c4",
            "invoice id=1001 amount=250.00 status=cleared",
            block_type="row",
            doc_id="doc3",
            file_type="csv",
            parent_id=None,
            section_path=["transactions"],
        ),
    ]


@pytest.fixture
def indexed(indexer: Indexer, sample_chunks: list[Chunk]) -> Indexer:
    """An indexer whose FIXTURE_COLLECTION already holds ``sample_chunks``."""
    indexer.ensure_collection(FIXTURE_COLLECTION)
    indexer.index_chunks(FIXTURE_COLLECTION, sample_chunks)
    return indexer


@pytest.fixture
def sparse_vec() -> SparseVector:
    return SparseVector(indices=[1, 5, 9], values=[0.5, 0.25, 0.75])


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
@pytest.fixture
def service_db() -> Iterator[None]:
    """Fresh service + docstore schema per test (drop + create on the engine)."""
    from rag_embedding.store.docstore import DocstoreBase
    from service_embed import models  # noqa: F401  (register tables)
    from service_embed.db import Base, engine

    for metadata in (Base.metadata, DocstoreBase.metadata):
        metadata.drop_all(engine)
        metadata.create_all(engine)
    yield
    for metadata in (Base.metadata, DocstoreBase.metadata):
        metadata.drop_all(engine)


# --------------------------------------------------------------------------- #
# Qdrant introspection helpers
#
# qdrant-client models the collection config with wide unions (a collection may
# have one unnamed vector or a dict of named ones, several quantization shapes,
# eight condition types). These narrow once, here, so the assertions in the
# tests stay about behaviour instead of about the client's type algebra.
# --------------------------------------------------------------------------- #
def dense_params(info: Any) -> Any:
    """VectorParams for the named dense vector of a live collection."""
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict), "expected named vectors"
    return vectors[DENSE_VECTOR]


def sparse_params(info: Any) -> Any:
    """SparseVectorParams for the named BM25 vector (None if sparse is off)."""
    sparse = info.config.params.sparse_vectors or {}
    return sparse.get(SPARSE_VECTOR)


def point_ids(points: list[Any]) -> list[str]:
    """chunk_ids of scored points, in rank order (payload is Optional client-side)."""
    return [str((p.payload or {})["chunk_id"]) for p in points]


def conditions(query_filter: Any) -> list[Any]:
    """The ``must`` conditions of a built filter."""
    assert query_filter is not None, "expected a filter, got None"
    return list(query_filter.must or [])


def integration_url() -> str | None:
    return os.environ.get("RAG_EMBED_TEST_QDRANT_URL")


requires_qdrant = pytest.mark.skipif(
    integration_url() is None,
    reason="set RAG_EMBED_TEST_QDRANT_URL to run integration tests against a real Qdrant",
)
