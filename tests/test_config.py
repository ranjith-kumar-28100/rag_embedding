"""Config defaults, the air-gap guardrail, and offline env enforcement."""

from __future__ import annotations

import os

import pytest

from rag_embedding.config import EmbeddingConfig
from rag_embedding.offline import _OFFLINE_ENV, enforce_offline


def test_defaults_are_local_and_online_capable() -> None:
    cfg = EmbeddingConfig()
    assert cfg.embedding_provider == "sentence_transformers"
    assert cfg.ragas_llm_provider == "ollama"
    assert cfg.sparse_enabled is True
    assert cfg.docstore == "sqlite"
    assert cfg.fusion == "rrf"
    assert cfg.offline is False
    assert cfg.rerank_enabled is False


def test_offline_rejects_azure_embeddings() -> None:
    with pytest.raises(ValueError, match="off-machine"):
        EmbeddingConfig(offline=True, embedding_provider="azure")


def test_offline_rejects_azure_judge() -> None:
    with pytest.raises(ValueError, match="off-machine"):
        EmbeddingConfig(offline=True, ragas_llm_provider="azure")


def test_offline_with_local_providers_ok() -> None:
    assert EmbeddingConfig(offline=True).offline is True


def test_prefetch_k_must_cover_top_k() -> None:
    with pytest.raises(ValueError, match="child_prefetch_k"):
        EmbeddingConfig(top_k=50, child_prefetch_k=10)


def test_alias_and_physical_names() -> None:
    cfg = EmbeddingConfig(collection_prefix="acme")
    assert cfg.alias == "acme"
    assert cfg.physical_collection(3) == "acme_v3"


def test_enforce_offline_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    enforce_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_enforce_offline_respects_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    enforce_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "0"
