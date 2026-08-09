"""Dense provider dispatch and the Azure client's credential guard.

No test here loads a real model: the local provider is checked for *wiring*
(lazy load, identity, config plumbing), and the actual encoder quality is not
this repo's to assert.
"""

from __future__ import annotations

import pytest

from rag_embedding.config import EmbeddingConfig
from rag_embedding.embedders.base import DenseEmbedder, EmbedderError, SparseEmbedder
from rag_embedding.embedders.dense_azure import AzureOpenAIEmbedder, _l2_normalize
from rag_embedding.embedders.dense_local import SentenceTransformerEmbedder
from rag_embedding.embedders.factory import build_dense_embedder
from rag_embedding.embedders.sparse_bm25 import Bm25Encoder

from conftest import FakeDenseEmbedder


def test_default_provider_is_local() -> None:
    embedder = build_dense_embedder(EmbeddingConfig())
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder.model_id == "sentence_transformers:sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.parametrize("alias", ["sentence_transformers", "local", "st", "SENTENCE_TRANSFORMERS"])
def test_local_aliases(alias: str) -> None:
    assert isinstance(
        build_dense_embedder(EmbeddingConfig(embedding_provider=alias)),
        SentenceTransformerEmbedder,
    )


def test_unknown_provider_rejected() -> None:
    with pytest.raises(EmbedderError, match="Unknown embedding_provider"):
        build_dense_embedder(EmbeddingConfig(embedding_provider="cohere"))


def test_local_embedder_does_not_load_the_model_on_construction() -> None:
    """Construction must be free — Celery imports this module at startup."""
    embedder = SentenceTransformerEmbedder("definitely/not-a-real-model")
    assert embedder._model is None


def test_azure_requires_credentials() -> None:
    with pytest.raises(EmbedderError, match="azure_openai_endpoint"):
        build_dense_embedder(EmbeddingConfig(embedding_provider="azure"))


def test_azure_identity_and_known_dim() -> None:
    embedder = AzureOpenAIEmbedder(
        endpoint="https://example.openai.azure.com/",
        api_key="k",
        deployment="text-embedding-3-large",
        api_version="2024-10-21",
    )
    assert embedder.model_id == "azure:text-embedding-3-large"
    assert embedder.dim == 3072  # known, so no network probe is needed


def test_azure_config_plumbs_secret_through() -> None:
    cfg = EmbeddingConfig(
        embedding_provider="azure",
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_api_key="secret",
        embedding_deployment="text-embedding-3-small",
    )
    embedder = build_dense_embedder(cfg)
    assert embedder.model_id == "azure:text-embedding-3-small"
    assert embedder.dim == 1536


def test_l2_normalize() -> None:
    assert _l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    assert _l2_normalize([0.0, 0.0]) == [0.0, 0.0]  # no division by zero


def test_protocols_are_satisfied() -> None:
    """The fake used across the suite and the real classes share one shape."""
    assert isinstance(SentenceTransformerEmbedder("m"), DenseEmbedder)
    assert isinstance(Bm25Encoder(), SparseEmbedder)


def test_fake_dense_is_normalized_and_deterministic(fake_dense: FakeDenseEmbedder) -> None:
    a = fake_dense.embed_query("dense sparse fusion")
    b = fake_dense.embed_query("dense sparse fusion")
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0)
    assert isinstance(fake_dense, DenseEmbedder)
