"""Hybrid retrieval: fuse → (rerank) → parent lookup (spec §8).

``Retriever`` is the whole read path. It owns no state beyond its collaborators,
so a test can build one from a fake embedder, an in-memory docstore and an
in-process Qdrant and exercise every branch without a container.

Ordering of concerns, and why:
  1. dimension guard first — searching a collection built with another model
     returns plausible-looking nonsense, so it is a hard error, not a warning;
  2. fusion happens server-side in Qdrant (one round trip, one filter);
  3. parent lookup last — it is a keyed fetch, so it must not run over more
     children than survived ranking.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .config import EmbeddingConfig
from .embedders.base import DenseEmbedder, SparseEmbedder
from .models import (
    ParentContext,
    SearchHit,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from .rerank import maybe_rerank
from .store.docstore import ParentDocStore
from .store.qdrant_store import QdrantStore, build_filter

logger = logging.getLogger("rag_embedding.retrieval")


class _Timings:
    """Millisecond stopwatch for the response's ``timings`` block."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] = round((time.perf_counter() - start) * 1000.0, 2)


class Retriever:
    def __init__(
        self,
        *,
        cfg: EmbeddingConfig,
        store: QdrantStore,
        dense: DenseEmbedder,
        sparse: SparseEmbedder | None,
        docstore: ParentDocStore,
        collection: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.docstore = docstore
        # Default target is the alias, never a physical collection — that is what
        # makes the reindex swap invisible to callers (§9).
        self.collection = collection or cfg.alias

    # -- public ------------------------------------------------------------- #
    def search(self, request: SearchRequest) -> SearchResponse:
        timings = _Timings()
        top_k = request.top_k or self.cfg.top_k
        prefetch_k = max(request.prefetch_k or self.cfg.child_prefetch_k, top_k)
        with_parents = (
            self.cfg.with_parents if request.with_parents is None else request.with_parents
        )
        mode = self._effective_mode(request.mode)

        dense_vector: list[float] | None = None
        sparse_vector = None

        with timings.phase("embed_ms"):
            if mode in ("dense", "hybrid"):
                # Guard before embedding cost is spent, but after we know a dense
                # vector is actually needed.
                self.store.assert_dim(self.collection, self.dense.dim)
                dense_vector = self.dense.embed_query(request.query)
            if mode in ("sparse", "hybrid") and self.sparse is not None:
                sparse_vector = self.sparse.embed_query(request.query)

        with timings.phase("search_ms"):
            points = self.store.query(
                self.collection,
                mode=mode,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=top_k,
                prefetch_limit=prefetch_k,
                query_filter=build_filter(request.filters),
            )

        hits = [_hit_from_point(point, source=mode) for point in points]
        hits = maybe_rerank(self.cfg, request.query, hits)

        parents: list[ParentContext] = []
        if with_parents:
            with timings.phase("parents_ms"):
                parents = self.lookup_parents(hits)

        return SearchResponse(
            query=request.query,
            mode=mode,
            collection=self.collection,
            hits=hits,
            parents=parents,
            timings=timings.values,
        )

    def lookup_parents(self, hits: list[SearchHit]) -> list[ParentContext]:
        """Small-to-big (§8.2): dedupe parent ids, keep best child score, fetch once.

        Parents come back in *best-child order*, so the caller can feed them to a
        generator top-down. Children that have no parent (or whose parent is
        missing from the docstore) are not silently dropped — they are returned
        as a self-parent so the context set never loses a ranked hit.
        """
        if not hits:
            return []

        order: list[str] = []
        best_score: dict[str, float] = {}
        children: dict[str, list[SearchHit]] = {}
        for hit in hits:
            parent_id = hit.payload.get("parent_id") or hit.chunk_id
            if parent_id not in best_score:
                order.append(parent_id)
                best_score[parent_id] = hit.score
                children[parent_id] = []
            best_score[parent_id] = max(best_score[parent_id], hit.score)
            children[parent_id].append(hit)

        fetched = self.docstore.get_many(order)
        contexts: list[ParentContext] = []
        for parent_id in order:
            parent = fetched.get(parent_id)
            if parent is not None:
                contexts.append(
                    ParentContext(
                        chunk_id=parent.chunk_id,
                        doc_id=parent.doc_id,
                        text=parent.text,
                        score=best_score[parent_id],
                        payload=parent.to_payload(),
                        children=children[parent_id],
                    )
                )
                continue
            # No parent row: fall back to the child itself as its own context.
            fallback = children[parent_id][0]
            contexts.append(
                ParentContext(
                    chunk_id=fallback.chunk_id,
                    doc_id=str(fallback.payload.get("doc_id", "")),
                    text=fallback.text,
                    score=best_score[parent_id],
                    payload=fallback.payload,
                    children=children[parent_id],
                )
            )
        return contexts

    def compare_modes(self, request: SearchRequest) -> dict[str, SearchResponse]:
        """Run dense, sparse and hybrid for the same query (UI §12, raw vs fused).

        Three real queries rather than a client-side re-fusion, so what the UI
        shows is exactly what each mode would return in production.
        """
        out: dict[str, SearchResponse] = {}
        for mode in ("dense", "sparse", "hybrid"):
            if mode == "sparse" and self.sparse is None:
                continue
            out[mode] = self.search(
                request.model_copy(update={"mode": mode, "with_parents": False})
            )
        return out

    # -- internals ---------------------------------------------------------- #
    def _effective_mode(self, requested: SearchMode) -> SearchMode:
        """Degrade to dense when the sparse branch is switched off — loudly, once."""
        if self.sparse is not None:
            return requested
        if requested == "sparse":
            raise ValueError(
                "mode='sparse' requested but sparse_enabled=false; "
                "set RAG_EMBED_SPARSE_ENABLED=true and re-index."
            )
        if requested == "hybrid":
            logger.warning("sparse_enabled=false; serving 'hybrid' as dense-only")
            return "dense"
        return requested


def _hit_from_point(point: Any, *, source: str) -> SearchHit:
    payload: dict[str, Any] = dict(point.payload or {})
    return SearchHit(
        chunk_id=str(payload.get("chunk_id", point.id)),
        score=float(point.score),
        text=str(payload.get("text", "")),
        payload=payload,
        source=source,
    )
