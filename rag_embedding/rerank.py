"""Cross-encoder reranking (spec §8.4) — deferred, wiring only.

The decision on record (§18) is to ship hybrid + parent lookup without a
reranker. This module exists so enabling one later is a config flip and a model
download, not a change to the retrieval path: ``Retriever`` already calls
``maybe_rerank`` between fusion and parent lookup.

``rerank_enabled=true`` fails loudly rather than silently returning unranked
results — a silent no-op would look like "the reranker didn't help".
"""

from __future__ import annotations

import logging

from .config import EmbeddingConfig
from .models import SearchHit

logger = logging.getLogger("rag_embedding.rerank")


class RerankerNotAvailableError(RuntimeError):
    """rerank_enabled=true but no reranker is wired in this build."""


def maybe_rerank(cfg: EmbeddingConfig, query: str, hits: list[SearchHit]) -> list[SearchHit]:
    """Pass-through unless reranking is enabled (in which case: not implemented)."""
    if not cfg.rerank_enabled:
        return hits
    raise RerankerNotAvailableError(
        f"rerank_enabled=true (model={cfg.rerank_model!r}) but reranking is deferred in v1. "
        "Set RAG_EMBED_RERANK_ENABLED=false, or implement CrossEncoderReranker here and "
        "prefetch the model into the model-cache volume."
    )
