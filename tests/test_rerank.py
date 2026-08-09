"""Cross-encoder reranking: scoring, ordering, candidate widening, config guards.

The real ``CrossEncoder`` is never loaded here — a fake scorer stands in, because
what needs asserting is the *wiring* (how many candidates are fetched, which score
ends up where, what the fallbacks are), not BAAI's model weights.
"""

from __future__ import annotations

import pytest

from rag_embedding.config import EmbeddingConfig
from rag_embedding.indexer import Indexer
from rag_embedding.models import SearchHit, SearchRequest
from rag_embedding.rerank import (
    CrossEncoderReranker,
    Reranker,
    RerankerUnavailableError,
    _apply_scores,
    build_reranker,
)
from rag_embedding.retrieval import Retriever
from rag_embedding.store.docstore import InMemoryParentDocStore
from rag_embedding.store.qdrant_store import QdrantStore

from conftest import FIXTURE_COLLECTION, FakeDenseEmbedder


class FakeReranker:
    """Scores by how many query tokens a chunk contains — deterministic and
    deliberately *different* from the fused order, so a test can tell whether
    reranking actually took effect."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    @property
    def model_id(self) -> str:
        return "fake:reranker"

    def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]:
        self.calls.append((query, len(hits)))
        terms = set(query.lower().split())
        scores = [float(sum(t in hit.text.lower() for t in terms)) for hit in hits]
        return _apply_scores(hits, scores, top_n=top_n)


def hit(chunk_id: str, score: float, text: str = "") -> SearchHit:
    return SearchHit(chunk_id=chunk_id, score=score, text=text or chunk_id, source="hybrid")


# --------------------------------------------------------------------------- #
# Score application (pure)
# --------------------------------------------------------------------------- #
def test_apply_scores_reorders_by_the_new_score() -> None:
    hits = [hit("a", 0.9), hit("b", 0.5), hit("c", 0.1)]
    out = _apply_scores(hits, [0.1, 9.0, 3.0], top_n=3)
    assert [h.chunk_id for h in out] == ["b", "c", "a"]


def test_apply_scores_preserves_the_fused_score() -> None:
    """The two scales are different; both are reported rather than conflated."""
    out = _apply_scores([hit("a", 0.9)], [-4.2], top_n=1)
    assert out[0].score == -4.2
    assert out[0].retrieval_score == 0.9
    assert out[0].source == "hybrid+rerank"


def test_apply_scores_truncates_to_top_n() -> None:
    hits = [hit(str(i), 1.0 / (i + 1)) for i in range(10)]
    assert len(_apply_scores(hits, [float(i) for i in range(10)], top_n=3)) == 3


def test_apply_scores_handles_negative_logits() -> None:
    """Cross-encoders emit unbounded logits; ordering must not assume positivity."""
    out = _apply_scores([hit("a", 0.5), hit("b", 0.4)], [-8.0, -1.0], top_n=2)
    assert [h.chunk_id for h in out] == ["b", "a"]


def test_apply_scores_is_idempotent_on_retrieval_score() -> None:
    """Reranking twice must not overwrite the original fused score."""
    once = _apply_scores([hit("a", 0.9)], [2.0], top_n=1)
    twice = _apply_scores(once, [5.0], top_n=1)
    assert twice[0].retrieval_score == 0.9


def test_apply_scores_empty() -> None:
    assert _apply_scores([], [], top_n=5) == []


def test_rerank_of_no_hits_is_empty() -> None:
    assert FakeReranker().rerank("q", [], top_n=5) == []


# --------------------------------------------------------------------------- #
# Factory & config
# --------------------------------------------------------------------------- #
def test_disabled_by_default() -> None:
    assert build_reranker(EmbeddingConfig()) is None


def test_factory_builds_the_configured_cross_encoder() -> None:
    reranker = build_reranker(EmbeddingConfig(rerank_enabled=True))
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_id == "cross_encoder:BAAI/bge-reranker-base"


def test_factory_passes_config_through() -> None:
    cfg = EmbeddingConfig(
        rerank_enabled=True,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_device="cpu",
        rerank_batch_size=8,
    )
    reranker = build_reranker(cfg)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker._batch_size == 8
    assert reranker.model_id.endswith("ms-marco-MiniLM-L-6-v2")


def test_construction_does_not_load_the_model() -> None:
    """Import/construct must stay free — workers build this at startup."""
    assert CrossEncoderReranker("definitely/not-a-real-model")._model is None


def test_missing_model_fails_with_actionable_guidance() -> None:
    reranker = CrossEncoderReranker("definitely/not-a-real-model-xyz")
    with pytest.raises(RerankerUnavailableError, match="prefetch"):
        reranker.rerank("q", [hit("a", 1.0)], top_n=1)


def test_candidates_must_cover_top_k() -> None:
    with pytest.raises(ValueError, match="rerank_candidates"):
        EmbeddingConfig(rerank_enabled=True, rerank_candidates=5, top_k=10)


def test_candidates_unchecked_when_disabled() -> None:
    assert EmbeddingConfig(rerank_candidates=5, top_k=10).rerank_enabled is False


def test_protocol_is_satisfied() -> None:
    assert isinstance(CrossEncoderReranker("m"), Reranker)
    assert isinstance(FakeReranker(), Reranker)


# --------------------------------------------------------------------------- #
# Retrieval integration
# --------------------------------------------------------------------------- #
@pytest.fixture
def rerank_cfg() -> EmbeddingConfig:
    return EmbeddingConfig(
        collection_prefix="test_rag",
        embedding_batch_size=2,
        top_k=2,
        child_prefetch_k=20,
        rerank_enabled=True,
        rerank_candidates=10,
    )


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def rr_retriever(
    rerank_cfg: EmbeddingConfig,
    store: QdrantStore,
    fake_dense: FakeDenseEmbedder,
    bm25: object,
    docstore: InMemoryParentDocStore,
    fake_reranker: FakeReranker,
) -> Retriever:
    return Retriever(
        cfg=rerank_cfg,
        store=store,
        dense=fake_dense,
        sparse=bm25,  # type: ignore[arg-type]
        docstore=docstore,
        collection=FIXTURE_COLLECTION,
        reranker=fake_reranker,
    )


def test_reranker_sees_more_candidates_than_top_k(
    indexed: Indexer, rr_retriever: Retriever, fake_reranker: FakeReranker
) -> None:
    """The whole point: reranking can only reorder what it is handed."""
    response = rr_retriever.search(SearchRequest(query="dense sparse fusion"))
    _query, candidate_count = fake_reranker.calls[0]
    assert candidate_count == 4  # every indexed child, > top_k of 2
    assert len(response.hits) == 2  # truncated back to top_k


def test_response_reports_that_it_was_reranked(
    indexed: Indexer, rr_retriever: Retriever
) -> None:
    response = rr_retriever.search(SearchRequest(query="retrieval"))
    assert response.reranked is True
    assert "rerank_ms" in response.timings
    assert all(h.source.endswith("+rerank") for h in response.hits)
    assert all(h.retrieval_score is not None for h in response.hits)


def test_rerank_actually_changes_the_order(
    indexed: Indexer, rr_retriever: Retriever
) -> None:
    query = "revenue quarterly"
    fused = rr_retriever.search(SearchRequest(query=query, rerank=False, with_parents=False))
    reranked = rr_retriever.search(SearchRequest(query=query, rerank=True, with_parents=False))
    assert reranked.hits[0].chunk_id == "c3"  # the chunk containing both terms
    assert reranked.reranked is True and fused.reranked is False


def test_per_request_off_switch(indexed: Indexer, rr_retriever: Retriever) -> None:
    response = rr_retriever.search(SearchRequest(query="retrieval", rerank=False))
    assert response.reranked is False
    assert "rerank_ms" not in response.timings
    assert len(response.hits) == 2  # still truncated to top_k without a reranker


def test_parents_come_from_the_reranked_children(
    indexed: Indexer, rr_retriever: Retriever
) -> None:
    """Parent lookup must run after reranking, not on the fused order."""
    response = rr_retriever.search(SearchRequest(query="revenue quarterly"))
    assert response.parents
    assert response.parents[0].children[0].chunk_id == response.hits[0].chunk_id


def test_requesting_rerank_without_one_configured_is_an_error(
    indexed: Indexer, retriever: Retriever
) -> None:
    """Silently returning unranked hits would read as 'reranking didn't help'."""
    with pytest.raises(ValueError, match="no reranker is configured"):
        retriever.search(SearchRequest(query="retrieval", rerank=True))


def test_retriever_builds_a_reranker_from_config(
    rerank_cfg: EmbeddingConfig,
    store: QdrantStore,
    fake_dense: FakeDenseEmbedder,
    bm25: object,
    docstore: InMemoryParentDocStore,
) -> None:
    """A library caller flipping the config should not need a second wiring step."""
    retriever = Retriever(
        cfg=rerank_cfg,
        store=store,
        dense=fake_dense,
        sparse=bm25,  # type: ignore[arg-type]
        docstore=docstore,
    )
    assert isinstance(retriever.reranker, CrossEncoderReranker)
