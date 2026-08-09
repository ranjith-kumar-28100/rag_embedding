"""Azure OpenAI dense embeddings — the only path that sends text off-machine.

Opt-in: ``embedding_provider=azure``, and rejected outright when ``offline=true``
(``EmbeddingConfig`` validates that at load time, so this module is unreachable
in an air-gapped deployment).
"""

from __future__ import annotations

import logging
from typing import Any

from .base import EmbedderError

logger = logging.getLogger("rag_embedding.embedders.dense_azure")

# Known output sizes, so `dim` does not cost a network round-trip before the
# collection is created. Unknown deployments fall back to probing with one call.
_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


class AzureOpenAIEmbedder:
    """Batched Azure OpenAI embeddings, L2-normalized for cosine + int8 parity."""

    def __init__(
        self,
        *,
        endpoint: str | None,
        api_key: str | None,
        deployment: str,
        api_version: str,
        batch_size: int = 64,
    ) -> None:
        if not endpoint or not api_key:
            raise EmbedderError(
                "Azure embeddings require azure_openai_endpoint and azure_openai_api_key. "
                "Set RAG_EMBED_AZURE_OPENAI_ENDPOINT / RAG_EMBED_AZURE_OPENAI_API_KEY, or use "
                "RAG_EMBED_EMBEDDING_PROVIDER=sentence_transformers to stay local."
            )
        self._endpoint = endpoint
        self._api_key = api_key
        self._deployment = deployment
        self._api_version = api_version
        self._batch_size = batch_size
        self._client: Any = None
        self._dim: int | None = _KNOWN_DIMS.get(deployment)

    @property
    def model_id(self) -> str:
        return f"azure:{self._deployment}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Unknown deployment name: one tiny probe call settles it.
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError as exc:
                raise EmbedderError(
                    "the azure provider needs the optional dependency: "
                    'uv pip install -e ".[azure]"'
                ) from exc
            self._client = AzureOpenAI(
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
                api_version=self._api_version,
            )
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = client.embeddings.create(model=self._deployment, input=batch)
            # The API preserves input order, but sort by index rather than trust it.
            for item in sorted(response.data, key=lambda d: d.index):
                out.append(_l2_normalize([float(x) for x in item.embedding]))
        if self._dim is None and out:
            self._dim = len(out[0])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]
