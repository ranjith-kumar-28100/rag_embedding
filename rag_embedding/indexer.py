"""Indexing and reindexing pipeline (spec §7, §9).

Chunk JSONL in → parents to the docstore, children embedded (dense + BM25) and
upserted into Qdrant with the full payload. Reindex re-embeds the *stored*
payloads, so a new model or new HNSW settings never require stage 1 to re-emit
anything.

This lives in the library, not in the Celery task, so the whole indexing path is
testable without a broker: the task layer only supplies a progress callback and
writes DB rows.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from .chunks import (
    ChunkParseError,
    batched,
    chunk_from_payload,
    count_lines,
    iter_chunks,
    split_roles,
)
from .config import EmbeddingConfig
from .embedders.base import DenseEmbedder, SparseEmbedder
from .models import Chunk, ChunkRole, CollectionSpec
from .store.docstore import ParentDocStore
from .store.qdrant_store import QdrantStore

logger = logging.getLogger("rag_embedding.indexer")

# (processed_children, total_children) — the UI's percent comes straight from this.
ProgressCallback = Callable[[int, int], None]


class IndexStats(BaseModel):
    total_chunks: int = 0
    children_indexed: int = 0
    parents_stored: int = 0
    skipped: int = 0
    seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class Indexer:
    def __init__(
        self,
        *,
        cfg: EmbeddingConfig,
        store: QdrantStore,
        dense: DenseEmbedder,
        sparse: SparseEmbedder | None,
        docstore: ParentDocStore,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.docstore = docstore

    # -- collection spec ---------------------------------------------------- #
    def collection_spec(self, name: str) -> CollectionSpec:
        """Fingerprint the active embedder onto a collection definition (§7.1)."""
        return CollectionSpec(
            name=name,
            provider=self.cfg.embedding_provider,
            model_id=self.dense.model_id,
            dim=self.dense.dim,
            hnsw_m=self.cfg.hnsw_m,
            hnsw_ef_construct=self.cfg.hnsw_ef_construct,
            quantization=self.cfg.quantization,
            on_disk_vectors=self.cfg.on_disk_vectors,
            on_disk_payload=self.cfg.on_disk_payload,
            sparse_enabled=self.cfg.sparse_enabled,
        )

    def ensure_collection(self, name: str) -> CollectionSpec:
        spec = self.collection_spec(name)
        self.store.create_collection(spec)
        return spec

    # -- indexing (§7) ------------------------------------------------------ #
    def index_file(
        self,
        collection: str,
        path: str | Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> IndexStats:
        """Index one ``.jsonl`` / ``.jsonl.gz``.

        The file is walked twice: once to count lines (so progress has a
        denominator before any embedding starts) and once to index. Counting is
        cheap next to embedding, and a progress bar that can only count up is
        worth more than the saved pass.
        """
        total_lines = count_lines(path)
        stats = self.index_chunks(
            collection,
            iter_chunks(path),
            expected_total=total_lines,
            progress=progress,
        )
        if stats.total_chunks == 0:
            # Individual bad lines are tolerated, but a file that yields nothing
            # is a failure the operator has to see — not a job quietly marked done.
            raise ChunkParseError(
                f"{Path(path).name}: no valid chunks in {total_lines} line(s) — "
                "is this a stage-1 chunk JSONL?"
            )
        return stats

    def index_chunks(
        self,
        collection: str,
        chunks: Iterable[Chunk],
        *,
        expected_total: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> IndexStats:
        started = time.perf_counter()
        stats = IndexStats()

        split = split_roles(chunks)
        stats.total_chunks = split.total
        stats.skipped = split.skipped
        stats.warnings.extend(split.errors[:20])
        if expected_total is not None and split.total < expected_total:
            malformed = expected_total - split.total - split.skipped
            if malformed > 0:
                stats.warnings.append(f"{malformed} line(s) skipped as malformed JSON")

        # Parents first: a child can only resolve to a parent that already exists,
        # and a crash between the two should leave orphan parents (harmless) rather
        # than children pointing at nothing (a silent context hole).
        if split.parents:
            stats.parents_stored = self.docstore.upsert(split.parents)

        total_children = len(split.children)
        if progress is not None:
            progress(0, total_children)

        for batch in batched(split.children, self.cfg.embedding_batch_size):
            texts = [c.text for c in batch]
            dense_vectors = self.dense.embed_documents(texts)
            sparse_vectors = self.sparse.embed_documents(texts) if self.sparse else None
            stats.children_indexed += self.store.upsert_children(
                collection, batch, dense_vectors, sparse_vectors
            )
            if progress is not None:
                progress(stats.children_indexed, total_children)

        stats.seconds = round(time.perf_counter() - started, 3)
        logger.info(
            "indexed %s: %d children, %d parents in %.2fs",
            collection,
            stats.children_indexed,
            stats.parents_stored,
            stats.seconds,
        )
        return stats

    # -- reindex (§9) ------------------------------------------------------- #
    def reindex_from(
        self,
        source_collection: str,
        target_collection: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> IndexStats:
        """Rebuild ``target`` from ``source``'s stored payloads — no re-upload.

        Child text lives in the Qdrant payload, so a new embedding model or new
        HNSW/quantization settings only cost a re-embed. Parents are untouched:
        they are not embedded, and the docstore is shared across versions.
        """
        self.ensure_collection(target_collection)
        total = self.store.count(source_collection)
        if progress is not None:
            progress(0, total)

        stats = IndexStats(total_chunks=total)
        started = time.perf_counter()
        buffer: list[Chunk] = []

        def flush() -> None:
            if not buffer:
                return
            texts = [c.text for c in buffer]
            dense_vectors = self.dense.embed_documents(texts)
            sparse_vectors = self.sparse.embed_documents(texts) if self.sparse else None
            stats.children_indexed += self.store.upsert_children(
                target_collection, buffer, dense_vectors, sparse_vectors
            )
            buffer.clear()
            if progress is not None:
                progress(stats.children_indexed, total)

        for payload in self.store.scroll_payloads(source_collection):
            chunk = chunk_from_payload(payload)
            if chunk.role is not ChunkRole.CHILD or not chunk.text.strip():
                stats.skipped += 1
                continue
            buffer.append(chunk)
            if len(buffer) >= self.cfg.embedding_batch_size:
                flush()
        flush()

        stats.seconds = round(time.perf_counter() - started, 3)
        return stats

    def activate(self, collection: str, alias: str | None = None) -> str:
        """Point the serving alias at ``collection`` — the atomic swap (§9)."""
        target_alias = alias or self.cfg.alias
        self.store.set_alias(target_alias, collection)
        return target_alias
