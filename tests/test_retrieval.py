"""Hybrid retrieval: mode behaviour, fusion, parent lookup, guards."""

from __future__ import annotations

import pytest

from rag_embedding.config import EmbeddingConfig
from rag_embedding.indexer import Indexer
from rag_embedding.models import Chunk, SearchFilters, SearchHit, SearchRequest
from rag_embedding.rerank import RerankerNotAvailableError
from rag_embedding.retrieval import Retriever
from rag_embedding.store.docstore import InMemoryParentDocStore
from rag_embedding.store.qdrant_store import DimensionMismatchError, QdrantStore

from conftest import FIXTURE_COLLECTION, FakeDenseEmbedder, make_chunk


def test_hybrid_search_returns_ranked_children(indexed: Indexer, retriever: Retriever) -> None:
    response = retriever.search(SearchRequest(query="dense and sparse fusion"))
    assert response.mode == "hybrid"
    assert response.collection == FIXTURE_COLLECTION
    assert response.hits
    assert response.hits[0].chunk_id == "c1"
    scores = [h.score for h in response.hits]
    assert scores == sorted(scores, reverse=True)


def test_parent_lookup_returns_parent_text(indexed: Indexer, retriever: Retriever) -> None:
    response = retriever.search(SearchRequest(query="dense and sparse fusion"))
    assert response.parents
    parent = response.parents[0]
    assert parent.chunk_id == "p1"
    assert "returns parent context" in parent.text
    assert [c.chunk_id for c in parent.children][0] == "c1"


def test_parents_are_deduped_and_keep_the_best_child_score(
    retriever: Retriever, docstore: InMemoryParentDocStore
) -> None:
    docstore.upsert([make_chunk("p1", "parent body", role="parent", parent_id=None)])
    hits = [
        SearchHit(chunk_id="c1", score=0.9, text="a", payload={"parent_id": "p1"}, source="hybrid"),
        SearchHit(chunk_id="c2", score=0.5, text="b", payload={"parent_id": "p1"}, source="hybrid"),
    ]
    parents = retriever.lookup_parents(hits)
    assert len(parents) == 1
    assert parents[0].score == 0.9
    assert len(parents[0].children) == 2


def test_child_without_a_parent_becomes_its_own_context(retriever: Retriever) -> None:
    """A ranked hit must never vanish just because it has no parent row."""
    hits = [SearchHit(chunk_id="orphan", score=0.7, text="body", payload={}, source="hybrid")]
    parents = retriever.lookup_parents(hits)
    assert [p.chunk_id for p in parents] == ["orphan"]
    assert parents[0].text == "body"


def test_with_parents_false_skips_the_lookup(indexed: Indexer, retriever: Retriever) -> None:
    response = retriever.search(SearchRequest(query="retrieval", with_parents=False))
    assert response.parents == []
    assert "parents_ms" not in response.timings


def test_top_k_is_respected(indexed: Indexer, retriever: Retriever) -> None:
    assert len(retriever.search(SearchRequest(query="retrieval", top_k=2)).hits) <= 2


def test_timings_are_reported(indexed: Indexer, retriever: Retriever) -> None:
    timings = retriever.search(SearchRequest(query="retrieval")).timings
    assert {"embed_ms", "search_ms", "parents_ms"} <= set(timings)
    assert all(v >= 0 for v in timings.values())


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def test_sparse_mode_finds_a_rare_literal_token(indexed: Indexer, retriever: Retriever) -> None:
    """The lexical branch is what catches ids the dense model blurs."""
    response = retriever.search(SearchRequest(query="1001", mode="sparse", with_parents=False))
    assert [h.chunk_id for h in response.hits] == ["c4"]


def test_dense_mode_runs_without_the_sparse_branch(
    indexed: Indexer, retriever: Retriever
) -> None:
    response = retriever.search(SearchRequest(query="approximate neighbours", mode="dense"))
    assert response.mode == "dense"
    assert response.hits[0].chunk_id == "c2"


def test_hybrid_covers_the_union_of_both_branches(
    indexed: Indexer, retriever: Retriever
) -> None:
    dense_ids = {
        h.chunk_id
        for h in retriever.search(
            SearchRequest(query="invoice 1001 cleared", mode="dense", with_parents=False)
        ).hits
    }
    sparse_ids = {
        h.chunk_id
        for h in retriever.search(
            SearchRequest(query="invoice 1001 cleared", mode="sparse", with_parents=False)
        ).hits
    }
    hybrid_ids = {
        h.chunk_id
        for h in retriever.search(
            SearchRequest(query="invoice 1001 cleared", mode="hybrid", with_parents=False)
        ).hits
    }
    assert (dense_ids | sparse_ids) >= hybrid_ids
    assert hybrid_ids >= (dense_ids & sparse_ids)
    assert hybrid_ids


def test_compare_modes_runs_all_three(indexed: Indexer, retriever: Retriever) -> None:
    results = retriever.compare_modes(SearchRequest(query="retrieval fusion"))
    assert set(results) == {"dense", "sparse", "hybrid"}
    assert all(r.parents == [] for r in results.values())


def test_sparse_mode_rejected_when_sparse_disabled(
    cfg: EmbeddingConfig,
    store: QdrantStore,
    fake_dense: FakeDenseEmbedder,
    docstore: InMemoryParentDocStore,
) -> None:
    retriever = Retriever(
        cfg=cfg, store=store, dense=fake_dense, sparse=None, docstore=docstore,
        collection=FIXTURE_COLLECTION,
    )
    with pytest.raises(ValueError, match="sparse_enabled"):
        retriever.search(SearchRequest(query="x", mode="sparse"))


def test_hybrid_degrades_to_dense_when_sparse_disabled(
    cfg: EmbeddingConfig,
    store: QdrantStore,
    fake_dense: FakeDenseEmbedder,
    docstore: InMemoryParentDocStore,
    sample_chunks: list[Chunk],
) -> None:
    indexer = Indexer(cfg=cfg, store=store, dense=fake_dense, sparse=None, docstore=docstore)
    indexer.ensure_collection(FIXTURE_COLLECTION)
    indexer.index_chunks(FIXTURE_COLLECTION, sample_chunks)
    retriever = Retriever(
        cfg=cfg, store=store, dense=fake_dense, sparse=None, docstore=docstore,
        collection=FIXTURE_COLLECTION,
    )
    response = retriever.search(SearchRequest(query="retrieval", mode="hybrid"))
    assert response.mode == "dense"
    assert response.hits


# --------------------------------------------------------------------------- #
# Filters and guards
# --------------------------------------------------------------------------- #
def test_filter_scopes_retrieval_to_one_document(
    indexed: Indexer, retriever: Retriever
) -> None:
    response = retriever.search(
        SearchRequest(
            query="revenue retrieval", filters=SearchFilters(doc_id=["doc2"]), with_parents=False
        )
    )
    assert {h.payload["doc_id"] for h in response.hits} == {"doc2"}


def test_block_type_filter(indexed: Indexer, retriever: Retriever) -> None:
    response = retriever.search(
        SearchRequest(
            query="revenue", filters=SearchFilters(block_type=["table"]), with_parents=False
        )
    )
    assert all(h.payload["block_type"] == "table" for h in response.hits)


def test_dimension_mismatch_is_a_hard_error(
    indexed: Indexer,
    cfg: EmbeddingConfig,
    store: QdrantStore,
    docstore: InMemoryParentDocStore,
    bm25: object,
) -> None:
    """Never silently search a collection built with a different model."""
    other_model = Retriever(
        cfg=cfg,
        store=store,
        dense=FakeDenseEmbedder(dim=64, model_name="other"),
        sparse=bm25,  # type: ignore[arg-type]
        docstore=docstore,
        collection=FIXTURE_COLLECTION,
    )
    with pytest.raises(DimensionMismatchError, match="32-d"):
        other_model.search(SearchRequest(query="anything"))


def test_rerank_enabled_fails_loudly(
    indexed: Indexer,
    store: QdrantStore,
    fake_dense: FakeDenseEmbedder,
    bm25: object,
    docstore: InMemoryParentDocStore,
) -> None:
    """A stub that silently no-ops would look like 'reranking didn't help'."""
    cfg = EmbeddingConfig(collection_prefix="test_rag", rerank_enabled=True)
    retriever = Retriever(
        cfg=cfg, store=store, dense=fake_dense, sparse=bm25,  # type: ignore[arg-type]
        docstore=docstore, collection=FIXTURE_COLLECTION,
    )
    with pytest.raises(RerankerNotAvailableError, match="deferred"):
        retriever.search(SearchRequest(query="retrieval"))


def test_retriever_defaults_to_the_alias(
    cfg: EmbeddingConfig,
    store: QdrantStore,
    fake_dense: FakeDenseEmbedder,
    bm25: object,
    docstore: InMemoryParentDocStore,
) -> None:
    retriever = Retriever(
        cfg=cfg, store=store, dense=fake_dense, sparse=bm25,  # type: ignore[arg-type]
        docstore=docstore,
    )
    assert retriever.collection == cfg.alias == "test_rag"
