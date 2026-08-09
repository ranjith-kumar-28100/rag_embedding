"""Provider dispatch — mirrors ``rag_ingestion.chunkers.embeddings.build_embeddings``.

One place decides which dense provider is live, so nothing downstream branches on
``embedding_provider``.
"""

from __future__ import annotations

from ..config import EmbeddingConfig
from .base import DenseEmbedder, EmbedderError, SparseEmbedder
from .dense_local import SentenceTransformerEmbedder
from .sparse_bm25 import Bm25Encoder

_LOCAL_ALIASES = frozenset({"sentence_transformers", "local", "st"})


def build_dense_embedder(cfg: EmbeddingConfig) -> DenseEmbedder:
    """Local sentence-transformers (default) or Azure OpenAI."""
    provider = cfg.embedding_provider.lower()
    if provider in _LOCAL_ALIASES:
        return SentenceTransformerEmbedder(
            cfg.local_embedding_model,
            device=cfg.local_embedding_device,
            batch_size=cfg.embedding_batch_size,
        )
    if provider == "azure":
        # Imported lazily so the optional `openai` dependency is only needed by
        # deployments that actually opted into egress.
        from .dense_azure import AzureOpenAIEmbedder

        return AzureOpenAIEmbedder(
            endpoint=cfg.azure_openai_endpoint,
            api_key=(
                cfg.azure_openai_api_key.get_secret_value() if cfg.azure_openai_api_key else None
            ),
            deployment=cfg.embedding_deployment,
            api_version=cfg.azure_openai_api_version,
            batch_size=cfg.embedding_batch_size,
        )
    raise EmbedderError(
        f"Unknown embedding_provider {cfg.embedding_provider!r}; "
        "expected 'sentence_transformers' or 'azure'."
    )


def build_sparse_embedder(cfg: EmbeddingConfig) -> SparseEmbedder | None:
    """The BM25 encoder, or ``None`` when the sparse branch is switched off."""
    if not cfg.sparse_enabled:
        return None
    return Bm25Encoder(
        k1=cfg.bm25_k1,
        b=cfg.bm25_b,
        avg_len=cfg.bm25_avg_len,
        tokenizer=cfg.bm25_tokenizer,
    )
