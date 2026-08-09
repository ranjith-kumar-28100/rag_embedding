"""Qdrant collection lifecycle, upsert, hybrid query and aliases (spec §6, §8, §9).

The only module that knows Qdrant's API. Everything above it speaks
``Chunk`` / ``SearchHit``.

Storage posture (§6.3), all config-driven:
  * dense vectors ``on_disk`` with int8 scalar quantization kept ``always_ram`` —
    candidate scoring stays in memory at ~4x less of it, exact rescoring reads
    the full vector from disk;
  * ``on_disk_payload`` so payload size does not inflate RAM;
  * keyword payload indexes on the four filter keys, so a filtered search is an
    index intersection rather than a full scan;
  * deterministic uuid5 point ids, so re-indexing a chunk overwrites it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from qdrant_client import QdrantClient, models

from ..config import EmbeddingConfig
from ..embedders.base import SparseVector
from ..models import Chunk, CollectionSpec, SearchFilters, point_id_for

logger = logging.getLogger("rag_embedding.store.qdrant")

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

# Payload keys that get a keyword index (§6.1). section_path is a list, which
# Qdrant indexes element-wise — filtering by any ancestor heading just works.
INDEXED_KEYWORD_FIELDS: tuple[str, ...] = ("doc_id", "file_type", "block_type", "section_path")


class DimensionMismatchError(RuntimeError):
    """The live collection was built with a different embedding model/dim (§5.1)."""


class CollectionNotFoundError(RuntimeError):
    """The requested collection or alias does not resolve."""


def build_client(cfg: EmbeddingConfig) -> QdrantClient:
    return QdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None,
        timeout=int(cfg.qdrant_timeout),
    )


def _quantization_config(kind: str) -> models.QuantizationConfig | None:
    """int8 by default; ``product`` for extreme compression; ``none`` to disable."""
    if kind == "none":
        return None
    if kind == "int8":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                # Quantized vectors stay resident even with on-disk originals —
                # that combination is the whole point: cheap RAM scoring, exact
                # rescoring from disk.
                always_ram=True,
            )
        )
    if kind == "product":
        return models.ProductQuantization(
            product=models.ProductQuantizationConfig(
                compression=models.CompressionRatio.X16,
                always_ram=True,
            )
        )
    raise ValueError(f"unknown quantization {kind!r}; expected none|int8|product")


class QdrantStore:
    """Collection lifecycle + writes + queries for one Qdrant instance."""

    def __init__(self, client: QdrantClient, cfg: EmbeddingConfig) -> None:
        self.client = client
        self.cfg = cfg

    # -- lifecycle ---------------------------------------------------------- #
    def collection_exists(self, name: str) -> bool:
        return bool(self.client.collection_exists(name))

    def create_collection(self, spec: CollectionSpec) -> None:
        """Create a collection to spec. Idempotent — validates dim if it exists."""
        if self.collection_exists(spec.name):
            self.assert_dim(spec.name, spec.dim)
            return

        sparse_config = (
            {SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)}
            if spec.sparse_enabled
            else {}
        )
        self.client.create_collection(
            collection_name=spec.name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=spec.dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=spec.hnsw_m, ef_construct=spec.hnsw_ef_construct
                    ),
                    on_disk=spec.on_disk_vectors,
                    quantization_config=_quantization_config(spec.quantization),
                )
            },
            sparse_vectors_config=sparse_config,
            on_disk_payload=spec.on_disk_payload,
        )
        self._create_payload_indexes(spec.name)
        logger.info("created collection %s (dim=%d, model=%s)", spec.name, spec.dim, spec.model_id)

    def _create_payload_indexes(self, name: str) -> None:
        for field in INDEXED_KEYWORD_FIELDS:
            self.client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)

    def dense_dim(self, name: str) -> int:
        """Configured dense size of a live collection (accepts an alias)."""
        try:
            info = self.client.get_collection(name)
        except Exception as exc:  # noqa: BLE001 - client raises several types
            raise CollectionNotFoundError(f"collection or alias {name!r} not found") from exc
        params = info.config.params.vectors
        if isinstance(params, dict):
            dense = params.get(DENSE_VECTOR)
            if dense is None:
                raise CollectionNotFoundError(f"{name!r} has no {DENSE_VECTOR!r} named vector")
            return int(dense.size)
        if params is None:
            raise CollectionNotFoundError(f"{name!r} has no dense vector configured")
        return int(params.size)

    def assert_dim(self, name: str, dim: int) -> None:
        """Hard error on a cross-model query — never silently search the wrong space."""
        actual = self.dense_dim(name)
        if actual != dim:
            raise DimensionMismatchError(
                f"collection {name!r} stores {actual}-d vectors but the active embedder "
                f"produces {dim}-d. Re-index with the matching model (POST /reindex) instead "
                "of searching across embedding spaces."
            )

    def count(self, name: str) -> int:
        return int(self.client.count(name, exact=True).count)

    # -- aliases (§9) ------------------------------------------------------- #
    def set_alias(self, alias: str, collection: str) -> None:
        """Atomically repoint ``alias`` at ``collection`` — the zero-downtime swap."""
        actions: list[models.AliasOperations] = []
        if alias in self.list_aliases():
            actions.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
            )
        actions.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=actions)
        logger.info("alias %s -> %s", alias, collection)

    def list_aliases(self) -> dict[str, str]:
        """alias name -> collection name."""
        return {
            description.alias_name: description.collection_name
            for description in self.client.get_aliases().aliases
        }

    def resolve_alias(self, alias: str) -> str | None:
        return self.list_aliases().get(alias)

    # -- writes ------------------------------------------------------------- #
    def upsert_children(
        self,
        collection: str,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None,
    ) -> int:
        """Upsert one batch of children with both named vectors and full payload."""
        if not chunks:
            return 0
        if len(dense_vectors) != len(chunks):
            raise ValueError(
                f"got {len(dense_vectors)} dense vectors for {len(chunks)} chunks"
            )
        if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
            raise ValueError(
                f"got {len(sparse_vectors)} sparse vectors for {len(chunks)} chunks"
            )

        points: list[models.PointStruct] = []
        for idx, chunk in enumerate(chunks):
            vectors: dict[str, Any] = {DENSE_VECTOR: dense_vectors[idx]}
            if sparse_vectors is not None:
                sparse = sparse_vectors[idx]
                # An all-stopword chunk yields no terms; sending an empty sparse
                # vector is valid and simply makes it dense-only retrievable.
                vectors[SPARSE_VECTOR] = models.SparseVector(
                    indices=sparse.indices, values=sparse.values
                )
            points.append(
                models.PointStruct(
                    id=chunk.point_id, vector=vectors, payload=chunk.to_payload()
                )
            )
        self.client.upsert(collection_name=collection, points=points, wait=True)
        return len(points)

    def delete_by_doc(self, collection: str, doc_id: str) -> None:
        self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )

    # -- reads -------------------------------------------------------------- #
    def scroll_payloads(self, collection: str, batch_size: int = 256) -> Iterator[dict[str, Any]]:
        """Stream every payload — the reindex source (§9), no re-upload needed."""
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if point.payload:
                    yield dict(point.payload)
            if offset is None:
                return

    def retrieve_payloads(self, collection: str, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch payloads by chunk_id (via the deterministic point id)."""
        if not chunk_ids:
            return {}
        records = self.client.retrieve(
            collection_name=collection,
            ids=[point_id_for(cid) for cid in chunk_ids],
            with_payload=True,
            with_vectors=False,
        )
        out: dict[str, dict[str, Any]] = {}
        for record in records:
            if record.payload and "chunk_id" in record.payload:
                out[str(record.payload["chunk_id"])] = dict(record.payload)
        return out

    # -- queries (§8) ------------------------------------------------------- #
    def query(
        self,
        collection: str,
        *,
        mode: str,
        dense_vector: list[float] | None,
        sparse_vector: SparseVector | None,
        limit: int,
        prefetch_limit: int,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]:
        """Dense, sparse, or server-side-fused hybrid via the Query API."""
        params = models.SearchParams(hnsw_ef=self.cfg.hnsw_ef)

        if mode == "dense":
            if dense_vector is None:
                raise ValueError("dense mode needs a dense query vector")
            response = self.client.query_points(
                collection_name=collection,
                query=dense_vector,
                using=DENSE_VECTOR,
                limit=limit,
                query_filter=query_filter,
                search_params=params,
                with_payload=True,
            )
            return list(response.points)

        if mode == "sparse":
            if sparse_vector is None:
                raise ValueError("sparse mode needs a sparse query vector (sparse_enabled=false?)")
            response = self.client.query_points(
                collection_name=collection,
                query=models.SparseVector(
                    indices=sparse_vector.indices, values=sparse_vector.values
                ),
                using=SPARSE_VECTOR,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            return list(response.points)

        if mode != "hybrid":
            raise ValueError(f"unknown search mode {mode!r}; expected dense|sparse|hybrid")

        # Hybrid: both branches prefetch `prefetch_limit` candidates, Qdrant fuses
        # them server-side. Fusing in the store (not the client) means one round
        # trip and one place where the filter is applied.
        prefetch: list[models.Prefetch] = []
        if dense_vector is not None:
            prefetch.append(
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR,
                    limit=prefetch_limit,
                    filter=query_filter,
                    params=params,
                )
            )
        if sparse_vector is not None and sparse_vector.indices:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices, values=sparse_vector.values
                    ),
                    using=SPARSE_VECTOR,
                    limit=prefetch_limit,
                    filter=query_filter,
                )
            )
        if not prefetch:
            return []
        if len(prefetch) == 1:
            # Only one branch has anything to contribute (sparse disabled, or a
            # query of pure stopwords). Fusing one list is a no-op; skip it.
            single = prefetch[0]
            response = self.client.query_points(
                collection_name=collection,
                query=single.query,
                using=single.using,
                limit=limit,
                query_filter=query_filter,
                search_params=params if single.using == DENSE_VECTOR else None,
                with_payload=True,
            )
            return list(response.points)

        fusion = (
            models.Fusion.DBSF if self.cfg.fusion == "dbsf" else models.Fusion.RRF
        )
        response = self.client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=fusion),
            limit=limit,
            with_payload=True,
        )
        return list(response.points)


def build_filter(filters: SearchFilters) -> models.Filter | None:
    """Translate the API filter DTO into a Qdrant filter (§8.3).

    Within a field the values are ORed (``MatchAny``); across fields they are
    ANDed — "any of these docs, and any of these block types".
    """
    if filters.is_empty():
        return None
    conditions: list[models.Condition] = []
    for key, values in (
        ("doc_id", filters.doc_id),
        ("file_type", filters.file_type),
        ("block_type", filters.block_type),
        ("section_path", filters.section_path),
    ):
        if values:
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))
            )
    return models.Filter(must=conditions)
