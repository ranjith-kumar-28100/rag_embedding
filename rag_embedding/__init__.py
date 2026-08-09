"""rag_embedding — embedding, hybrid indexing & retrieval (stage 2).

chunk JSONL in -> embedded, hybrid-indexed, retrievable context out. It does not
parse or chunk documents (that is rag_ingestion, stage 1) and it does not
synthesize answers. The two stages share a wire contract, not code.
"""

from __future__ import annotations

from .config import EmbeddingConfig, HnswParams
from .chunks import ChunkParseError, iter_chunks, load_chunks, split_roles
from .indexer import Indexer, IndexStats
from .models import (
    BlockType,
    Chunk,
    ChunkRole,
    CollectionSpec,
    EvalQuestion,
    EvalReport,
    EvalRequest,
    ParentContext,
    SearchFilters,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from .offline import enforce_offline
from .rerank import CrossEncoderReranker, Reranker, build_reranker
from .retrieval import Retriever

__all__ = [
    "BlockType",
    "Chunk",
    "ChunkParseError",
    "ChunkRole",
    "CollectionSpec",
    "CrossEncoderReranker",
    "EmbeddingConfig",
    "EvalQuestion",
    "EvalReport",
    "EvalRequest",
    "HnswParams",
    "IndexStats",
    "Indexer",
    "ParentContext",
    "Reranker",
    "Retriever",
    "SearchFilters",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
    "build_reranker",
    "enforce_offline",
    "iter_chunks",
    "load_chunks",
    "split_roles",
]

__version__ = "0.1.0"
