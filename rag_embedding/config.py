"""Embedding/indexing/retrieval configuration (spec §13).

Every tunable lives here. Embedders, the Qdrant store, retrieval and the eval
harness read from this object — no hardcoded thresholds downstream. Mirrors
``rag_ingestion.IngestionConfig`` in shape and in the air-gap guardrail.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Fusion = Literal["rrf", "dbsf"]
Quantization = Literal["none", "int8", "product"]
Docstore = Literal["sqlite", "qdrant"]
ChunkSource = Literal["upload", "path"]


class EmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_EMBED_",
        env_file=".env",
        extra="ignore",
    )

    # dense embeddings (§5.1) -------------------------------------------------
    # "sentence_transformers" -> fully local, no credentials (default)
    # "azure"                 -> Azure OpenAI embeddings (the only egress path)
    embedding_provider: str = "sentence_transformers"
    embedding_batch_size: int = 64

    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_embedding_device: str = "cpu"

    # azure — optional so local paths and unit tests run without credentials.
    # Validated lazily, only when the azure provider actually embeds.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    embedding_deployment: str = "text-embedding-3-large"

    # sparse / BM25 (§5.2) ----------------------------------------------------
    sparse_enabled: bool = True
    bm25_tokenizer: str = "default"  # default | whitespace
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    # BM25 length normalisation needs an average document length. Encoding is
    # streaming (documents arrive batch by batch), so we use a fixed expected
    # length rather than a corpus statistic — the same trade-off FastEmbed makes.
    # Qdrant supplies the IDF term at query time via Modifier.IDF.
    bm25_avg_len: float = 256.0

    # qdrant (§6) -------------------------------------------------------------
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_timeout: float = 60.0
    collection_prefix: str = "rag"

    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    hnsw_ef: int = 64  # query-time
    quantization: Quantization = "int8"
    on_disk_vectors: bool = True
    on_disk_payload: bool = True

    # retrieval (§8) ----------------------------------------------------------
    fusion: Fusion = "rrf"
    top_k: int = 10
    child_prefetch_k: int = 50
    with_parents: bool = True

    # rerank (§8.4) — local cross-encoder over the fused candidates.
    # Default off, and the cost is not marginal: one forward pass per candidate.
    # Measured on CPU, bge-reranker-base is ~140 ms at 10 candidates and ~610 ms
    # at 50, against ~10-20 ms for the fused search itself. Turn it on when
    # recall@k is already good but ordering is not, and size rerank_candidates
    # (or switch to ms-marco-MiniLM-L-6-v2, ~6x faster) against a latency budget.
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"  # MIT; ms-marco-MiniLM-L-6-v2 is lighter
    rerank_device: str = "cpu"
    rerank_batch_size: int = 32
    rerank_max_length: int = 512
    # How many fused hits get re-scored. Reranking can only reorder what it is
    # given, so this must exceed top_k or the whole step is a no-op.
    rerank_candidates: int = 50

    # parent docstore (§6.2) --------------------------------------------------
    docstore: Docstore = "sqlite"

    # evaluation (§10) --------------------------------------------------------
    ragas_llm_provider: str = "ollama"  # ollama | azure
    ragas_model: str = "mistral"
    ollama_base_url: str = "http://localhost:11434"
    ragas_judge_deployment: str = "gpt-4o-mini"  # azure only
    # LLM-judged metrics are opt-in: they are slow and, with a small local judge,
    # only indicative (§10.2). The deterministic metrics always run.
    ragas_llm_metrics_enabled: bool = False
    # Answer generation for faithfulness/answer_relevancy — deferred (§18).
    generation_enabled: bool = False

    # input (§7) --------------------------------------------------------------
    chunk_source: ChunkSource = "upload"  # upload (API multipart) | path (dev-only)
    chunk_path: str | None = None  # only used when chunk_source == "path"
    max_upload_mb: int = 500

    # air-gap: rejects any azure provider so no document text can egress.
    offline: bool = False

    # derived -----------------------------------------------------------------
    @property
    def alias(self) -> str:
        """The stable, client-facing collection name (a Qdrant alias)."""
        return self.collection_prefix

    def physical_collection(self, version: int) -> str:
        return f"{self.collection_prefix}_v{version}"

    @model_validator(mode="after")
    def _no_azure_when_offline(self) -> "EmbeddingConfig":
        """Fail fast if offline is requested but an Azure provider would egress data."""
        if self.offline:
            outbound = [
                name
                for name, value in (
                    ("embedding_provider", self.embedding_provider),
                    ("ragas_llm_provider", self.ragas_llm_provider),
                )
                if value.lower() == "azure"
            ]
            if outbound:
                raise ValueError(
                    f"offline=True but {', '.join(outbound)} = 'azure' would send document "
                    "data off-machine. Use the local providers (sentence_transformers / ollama)."
                )
        return self

    @model_validator(mode="after")
    def _sane_retrieval_sizes(self) -> "EmbeddingConfig":
        if self.child_prefetch_k < self.top_k:
            raise ValueError(
                f"child_prefetch_k ({self.child_prefetch_k}) must be >= top_k ({self.top_k}); "
                "fusion needs at least top_k candidates from each branch."
            )
        if self.rerank_enabled and self.rerank_candidates < self.top_k:
            raise ValueError(
                f"rerank_candidates ({self.rerank_candidates}) must be >= top_k ({self.top_k}); "
                "a reranker can only reorder the candidates it is handed."
            )
        return self


class HnswParams(BaseSettings):
    """HNSW/storage knobs captured on a collection (recorded in the registry)."""

    m: int = Field(default=16)
    ef_construct: int = Field(default=128)
    quantization: Quantization = "int8"
    on_disk_vectors: bool = True
    on_disk_payload: bool = True

    @classmethod
    def from_config(cls, cfg: EmbeddingConfig) -> "HnswParams":
        return cls(
            m=cfg.hnsw_m,
            ef_construct=cfg.hnsw_ef_construct,
            quantization=cfg.quantization,
            on_disk_vectors=cfg.on_disk_vectors,
            on_disk_payload=cfg.on_disk_payload,
        )
