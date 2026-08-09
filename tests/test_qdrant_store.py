"""Collection schema, idempotent upsert, filter translation, aliases, dim guard.

Runs against an in-process Qdrant (``:memory:``), which implements named vectors,
sparse vectors, Query API fusion, payload indexes and aliases — so these are real
store assertions, not mock assertions.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient, models

from rag_embedding.config import EmbeddingConfig
from rag_embedding.embedders.sparse_bm25 import Bm25Encoder
from rag_embedding.models import Chunk, CollectionSpec, SearchFilters
from rag_embedding.store.qdrant_store import (
    CollectionNotFoundError,
    DimensionMismatchError,
    QdrantStore,
    _quantization_config,
    build_filter,
)

from conftest import (
    FakeDenseEmbedder,
    conditions,
    dense_params,
    make_chunk,
    point_ids,
    sparse_params,
)


def _spec(name: str = "c1", dim: int = 32) -> CollectionSpec:
    return CollectionSpec(name=name, provider="fake", model_id="fake:m", dim=dim)


def test_creates_named_dense_and_sparse_vectors(store: QdrantStore, qdrant: QdrantClient) -> None:
    store.create_collection(_spec())
    info = qdrant.get_collection("c1")
    assert dense_params(info).size == 32
    assert dense_params(info).distance == models.Distance.COSINE
    assert sparse_params(info) is not None


def test_sparse_vector_declares_idf_modifier(store: QdrantStore, qdrant: QdrantClient) -> None:
    """Without Modifier.IDF the document-side weights would never be IDF-scaled."""
    store.create_collection(_spec())
    assert sparse_params(qdrant.get_collection("c1")).modifier == models.Modifier.IDF


def test_sparse_disabled_creates_dense_only(store: QdrantStore, qdrant: QdrantClient) -> None:
    store.create_collection(_spec().model_copy(update={"sparse_enabled": False}))
    assert sparse_params(qdrant.get_collection("c1")) is None


def test_create_is_idempotent(store: QdrantStore) -> None:
    store.create_collection(_spec())
    store.create_collection(_spec())  # must not raise
    assert store.collection_exists("c1")


def test_recreating_with_a_different_dim_is_a_hard_error(store: QdrantStore) -> None:
    store.create_collection(_spec(dim=32))
    with pytest.raises(DimensionMismatchError, match="384-d"):
        store.create_collection(_spec(dim=384))


def test_assert_dim_message_names_the_fix(store: QdrantStore) -> None:
    store.create_collection(_spec(dim=32))
    with pytest.raises(DimensionMismatchError, match="reindex"):
        store.assert_dim("c1", 768)


def test_missing_collection_raises_not_found(store: QdrantStore) -> None:
    with pytest.raises(CollectionNotFoundError):
        store.dense_dim("nope")


def test_quantization_config_variants() -> None:
    assert _quantization_config("none") is None
    assert isinstance(_quantization_config("int8"), models.ScalarQuantization)
    assert isinstance(_quantization_config("product"), models.ProductQuantization)
    with pytest.raises(ValueError, match="unknown quantization"):
        _quantization_config("binary")


def test_int8_quantization_keeps_scored_vectors_in_ram() -> None:
    quant = _quantization_config("int8")
    assert isinstance(quant, models.ScalarQuantization)
    assert quant.scalar.always_ram is True


# --------------------------------------------------------------------------- #
# Upsert
# --------------------------------------------------------------------------- #
def _upsert(
    store: QdrantStore, chunks: list[Chunk], dense: FakeDenseEmbedder, bm25: Bm25Encoder
) -> int:
    texts = [c.text for c in chunks]
    return store.upsert_children(
        "c1", chunks, dense.embed_documents(texts), bm25.embed_documents(texts)
    )


def test_upsert_is_idempotent_on_chunk_id(
    store: QdrantStore, fake_dense: FakeDenseEmbedder, bm25: Bm25Encoder
) -> None:
    store.create_collection(_spec())
    chunks = [make_chunk("c1", "dense retrieval"), make_chunk("c2", "sparse retrieval")]
    _upsert(store, chunks, fake_dense, bm25)
    _upsert(store, chunks, fake_dense, bm25)
    assert store.count("c1") == 2  # re-indexing overwrites, never duplicates


def test_upsert_updates_payload_in_place(
    store: QdrantStore, fake_dense: FakeDenseEmbedder, bm25: Bm25Encoder
) -> None:
    store.create_collection(_spec())
    _upsert(store, [make_chunk("c1", "original text")], fake_dense, bm25)
    _upsert(store, [make_chunk("c1", "revised text")], fake_dense, bm25)
    assert store.retrieve_payloads("c1", ["c1"])["c1"]["text"] == "revised text"


def test_upsert_rejects_mismatched_vector_counts(
    store: QdrantStore, fake_dense: FakeDenseEmbedder
) -> None:
    store.create_collection(_spec())
    chunks = [make_chunk("c1", "a"), make_chunk("c2", "b")]
    with pytest.raises(ValueError, match="dense vectors"):
        store.upsert_children("c1", chunks, [[0.0] * 32], None)


def test_upsert_empty_batch_is_a_noop(store: QdrantStore) -> None:
    store.create_collection(_spec())
    assert store.upsert_children("c1", [], [], None) == 0


def test_scroll_payloads_streams_everything(
    store: QdrantStore, fake_dense: FakeDenseEmbedder, bm25: Bm25Encoder
) -> None:
    store.create_collection(_spec())
    chunks = [make_chunk(f"c{i}", f"chunk number {i} about retrieval") for i in range(25)]
    _upsert(store, chunks, fake_dense, bm25)
    scrolled = list(store.scroll_payloads("c1", batch_size=4))
    assert {p["chunk_id"] for p in scrolled} == {c.chunk_id for c in chunks}


def test_delete_by_doc(
    store: QdrantStore, fake_dense: FakeDenseEmbedder, bm25: Bm25Encoder
) -> None:
    store.create_collection(_spec())
    _upsert(
        store,
        [make_chunk("a", "one", doc_id="d1"), make_chunk("b", "two", doc_id="d2")],
        fake_dense,
        bm25,
    )
    store.delete_by_doc("c1", "d1")
    assert store.count("c1") == 1


# --------------------------------------------------------------------------- #
# Aliases (§9)
# --------------------------------------------------------------------------- #
def test_alias_create_and_resolve(store: QdrantStore) -> None:
    store.create_collection(_spec("v1"))
    store.set_alias("rag", "v1")
    assert store.resolve_alias("rag") == "v1"


def test_alias_swap_is_atomic_replacement(store: QdrantStore) -> None:
    store.create_collection(_spec("v1"))
    store.create_collection(_spec("v2"))
    store.set_alias("rag", "v1")
    store.set_alias("rag", "v2")
    assert store.resolve_alias("rag") == "v2"
    assert list(store.list_aliases()) == ["rag"]  # no stale duplicate alias


def test_unset_alias_resolves_to_none(store: QdrantStore) -> None:
    assert store.resolve_alias("rag") is None


def test_queries_work_through_an_alias(
    store: QdrantStore, fake_dense: FakeDenseEmbedder, bm25: Bm25Encoder
) -> None:
    store.create_collection(_spec())
    _upsert(store, [make_chunk("c1", "dense retrieval fusion")], fake_dense, bm25)
    store.set_alias("rag", "c1")
    points = store.query(
        "rag",
        mode="dense",
        dense_vector=fake_dense.embed_query("retrieval"),
        sparse_vector=None,
        limit=5,
        prefetch_limit=10,
        query_filter=None,
    )
    assert point_ids(points) == ["c1"]


# --------------------------------------------------------------------------- #
# Filters (§8.3)
# --------------------------------------------------------------------------- #
def test_empty_filter_is_none() -> None:
    assert build_filter(SearchFilters()) is None


def test_filter_ors_within_a_field_and_ands_across() -> None:
    must = conditions(build_filter(SearchFilters(doc_id=["d1", "d2"], file_type=["pdf"])))
    assert {c.key for c in must} == {"doc_id", "file_type"}
    doc_condition = next(c for c in must if c.key == "doc_id")
    assert doc_condition.match.any == ["d1", "d2"]


def test_section_path_filter_matches_any_ancestor(
    store: QdrantStore, fake_dense: FakeDenseEmbedder, bm25: Bm25Encoder
) -> None:
    """section_path is a list, so filtering by a parent heading works."""
    store.create_collection(_spec())
    _upsert(
        store,
        [make_chunk("c1", "nested content", section_path=["3. Architecture", "3.2 Retrieval"])],
        fake_dense,
        bm25,
    )
    points = store.query(
        "c1",
        mode="dense",
        dense_vector=fake_dense.embed_query("content"),
        sparse_vector=None,
        limit=5,
        prefetch_limit=10,
        query_filter=build_filter(SearchFilters(section_path=["3. Architecture"])),
    )
    assert len(points) == 1


def test_filter_excludes_non_matching(
    store: QdrantStore, fake_dense: FakeDenseEmbedder, bm25: Bm25Encoder
) -> None:
    store.create_collection(_spec())
    _upsert(
        store,
        [make_chunk("a", "alpha text", doc_id="d1"), make_chunk("b", "alpha text", doc_id="d2")],
        fake_dense,
        bm25,
    )
    points = store.query(
        "c1",
        mode="hybrid",
        dense_vector=fake_dense.embed_query("alpha"),
        sparse_vector=bm25.embed_query("alpha"),
        limit=5,
        prefetch_limit=10,
        query_filter=build_filter(SearchFilters(doc_id=["d2"])),
    )
    assert point_ids(points) == ["b"]


# --------------------------------------------------------------------------- #
# Query modes
# --------------------------------------------------------------------------- #
def test_unknown_mode_rejected(store: QdrantStore) -> None:
    store.create_collection(_spec())
    with pytest.raises(ValueError, match="unknown search mode"):
        store.query(
            "c1",
            mode="magic",
            dense_vector=[0.0] * 32,
            sparse_vector=None,
            limit=1,
            prefetch_limit=1,
            query_filter=None,
        )


def test_dense_mode_requires_a_vector(store: QdrantStore) -> None:
    store.create_collection(_spec())
    with pytest.raises(ValueError, match="dense query vector"):
        store.query(
            "c1",
            mode="dense",
            dense_vector=None,
            sparse_vector=None,
            limit=1,
            prefetch_limit=1,
            query_filter=None,
        )


def test_hybrid_with_no_usable_branch_returns_empty(store: QdrantStore) -> None:
    store.create_collection(_spec())
    assert (
        store.query(
            "c1",
            mode="hybrid",
            dense_vector=None,
            sparse_vector=None,
            limit=5,
            prefetch_limit=10,
            query_filter=None,
        )
        == []
    )


def test_dbsf_fusion_is_configurable(qdrant: QdrantClient, fake_dense: FakeDenseEmbedder) -> None:
    store = QdrantStore(qdrant, EmbeddingConfig(fusion="dbsf", collection_prefix="test_rag"))
    bm25 = Bm25Encoder()
    store.create_collection(_spec())
    _upsert(store, [make_chunk("c1", "dense sparse fusion")], fake_dense, bm25)
    points = store.query(
        "c1",
        mode="hybrid",
        dense_vector=fake_dense.embed_query("fusion"),
        sparse_vector=bm25.embed_query("fusion"),
        limit=5,
        prefetch_limit=10,
        query_filter=None,
    )
    assert len(points) == 1
