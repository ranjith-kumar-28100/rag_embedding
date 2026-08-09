"""Dense and sparse embedding providers behind two Protocols."""

from __future__ import annotations

from .base import DenseEmbedder, EmbedderError, SparseEmbedder, SparseVector
from .dense_local import SentenceTransformerEmbedder
from .factory import build_dense_embedder, build_sparse_embedder
from .sparse_bm25 import Bm25Encoder, tokenize

__all__ = [
    "Bm25Encoder",
    "DenseEmbedder",
    "EmbedderError",
    "SentenceTransformerEmbedder",
    "SparseEmbedder",
    "SparseVector",
    "build_dense_embedder",
    "build_sparse_embedder",
    "tokenize",
]
