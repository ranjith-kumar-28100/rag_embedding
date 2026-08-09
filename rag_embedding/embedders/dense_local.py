"""Local dense embeddings via sentence-transformers — no external calls.

Default provider. The model loads lazily on first use (so importing this module
costs nothing) and is reused for the life of the process — Celery workers build
one embedder per process, which is why model load never shows up per task.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import EmbedderError

logger = logging.getLogger("rag_embedding.embedders.dense_local")


class SentenceTransformerEmbedder:
    """L2-normalized sentence-transformers embeddings.

    Normalization is not cosmetic: the collection is created with cosine
    distance, and int8 scalar quantization (§6.1) buckets each dimension against
    a fixed range — unnormalized vectors of varying magnitude quantize far worse.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._dim: int | None = None

    # -- identity ----------------------------------------------------------- #
    @property
    def model_id(self) -> str:
        return f"sentence_transformers:{self._model_name}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._get_model().get_sentence_embedding_dimension())
        return self._dim

    # -- model -------------------------------------------------------------- #
    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - install-time failure
                raise EmbedderError(
                    "sentence-transformers is not installed; it is a core dependency "
                    "of the default (local) embedding provider."
                ) from exc
            logger.info("loading local embedding model %s on %s", self._model_name, self._device)
            try:
                self._model = SentenceTransformer(self._model_name, device=self._device)
            except Exception as exc:  # noqa: BLE001 - surface offline cache misses clearly
                raise EmbedderError(
                    f"could not load {self._model_name!r}. When running offline the model must "
                    "already be in the model cache — run `make prefetch` once while online."
                ) from exc
        return self._model

    # -- embedding ---------------------------------------------------------- #
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
