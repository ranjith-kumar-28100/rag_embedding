"""Process-wide singletons that wire the library into the service.

The heavy objects — the dense model, the Qdrant client — are built once per
process and reused, which is why model load never shows up in a task's timing.
Everything here is ``lru_cache``d rather than global-assigned so tests can clear
it (``reset()``) between cases.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient

from rag_embedding.config import EmbeddingConfig
from rag_embedding.embedders.base import DenseEmbedder, SparseEmbedder
from rag_embedding.embedders.factory import build_dense_embedder, build_sparse_embedder
from rag_embedding.eval.ragas_eval import RagasEvaluator
from rag_embedding.indexer import Indexer
from rag_embedding.offline import enforce_offline
from rag_embedding.rerank import Reranker, build_reranker
from rag_embedding.retrieval import Retriever
from rag_embedding.store.docstore import (
    ParentDocStore,
    QdrantParentDocStore,
    SqliteParentDocStore,
)
from rag_embedding.store.qdrant_store import QdrantStore, build_client

logger = logging.getLogger("rag_embedding.service.deps")

PARENTS_COLLECTION_SUFFIX = "__parents"


@lru_cache
def get_config() -> EmbeddingConfig:
    cfg = EmbeddingConfig()
    if cfg.offline:
        # Must run before sentence-transformers is first imported; building the
        # config is the earliest common point on every entrypoint.
        enforce_offline()
    return cfg


@lru_cache
def get_qdrant() -> QdrantClient:
    return build_client(get_config())


@lru_cache
def get_store() -> QdrantStore:
    return QdrantStore(get_qdrant(), get_config())


@lru_cache
def get_dense() -> DenseEmbedder:
    return build_dense_embedder(get_config())


@lru_cache
def get_sparse() -> SparseEmbedder | None:
    return build_sparse_embedder(get_config())


@lru_cache
def get_docstore() -> ParentDocStore:
    cfg = get_config()
    if cfg.docstore == "qdrant":
        store = QdrantParentDocStore(
            get_qdrant(), f"{cfg.collection_prefix}{PARENTS_COLLECTION_SUFFIX}"
        )
        store.create_schema()
        return store
    from .db import engine

    sqlite_store = SqliteParentDocStore(engine)
    return sqlite_store


@lru_cache
def get_reranker() -> Reranker | None:
    """The cross-encoder, or None when rerank_enabled=false. Loaded once per process."""
    return build_reranker(get_config())


@lru_cache
def get_indexer() -> Indexer:
    return Indexer(
        cfg=get_config(),
        store=get_store(),
        dense=get_dense(),
        sparse=get_sparse(),
        docstore=get_docstore(),
    )


def get_retriever(collection: str | None = None) -> Retriever:
    """Not cached: the target collection varies (alias by default, a specific
    version when comparing a candidate reindex before activating it)."""
    return Retriever(
        cfg=get_config(),
        store=get_store(),
        dense=get_dense(),
        sparse=get_sparse(),
        docstore=get_docstore(),
        collection=collection,
        reranker=get_reranker(),
    )


def get_evaluator(collection: str | None = None) -> RagasEvaluator:
    return RagasEvaluator(get_config(), get_retriever(collection))


def reset() -> None:
    """Drop every cached singleton (tests, and after a config change)."""
    for cached in (
        get_config,
        get_qdrant,
        get_store,
        get_dense,
        get_sparse,
        get_docstore,
        get_reranker,
        get_indexer,
    ):
        cached.cache_clear()
