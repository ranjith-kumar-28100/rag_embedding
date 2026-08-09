"""Cross-encoder reranking (spec §8.4) — local, open-source, offline-cacheable.

Bi-encoders (the dense branch) embed the query and the document *independently*,
so the vector for a chunk is fixed before anyone asks a question. A cross-encoder
reads the query and the chunk **together** and scores the pair directly, which is
strictly more informative and correspondingly slower — O(candidates) forward
passes per query instead of one. That trade is why it runs on the fused top-N
rather than on the corpus.

Default model: ``BAAI/bge-reranker-base`` (MIT). ``cross-encoder/ms-marco-MiniLM-L-6-v2``
(Apache-2.0) is a lighter drop-in if latency matters more than accuracy. Both run
through sentence-transformers' ``CrossEncoder``, already a core dependency, so
enabling reranking adds no packages — only a one-time model download into the
model-cache volume (``make prefetch`` picks it up when ``rerank_enabled=true``).

Scoring note: cross-encoder outputs are unbounded logits on a different scale from
RRF's reciprocal ranks. The reranked score therefore *replaces* ``score`` and the
fused value is preserved as ``retrieval_score``, so a caller can always see both
and never has to guess which scale a number is on.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from .config import EmbeddingConfig
from .models import SearchHit

logger = logging.getLogger("rag_embedding.rerank")


@runtime_checkable
class Reranker(Protocol):
    """Re-scores query/chunk pairs. Implementations must be order-independent."""

    @property
    def model_id(self) -> str: ...

    def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]: ...


class RerankerUnavailableError(RuntimeError):
    """rerank_enabled=true but the cross-encoder could not be loaded."""


class CrossEncoderReranker:
    """Local cross-encoder reranking via sentence-transformers.

    The model loads lazily on first use and is reused for the process, matching
    the dense embedder's lifecycle — a Celery worker or API process pays the load
    once, not per request.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any = None

    @property
    def model_id(self) -> str:
        return f"cross_encoder:{self._model_name}"

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - core dependency
                raise RerankerUnavailableError(
                    "sentence-transformers is required for reranking."
                ) from exc
            logger.info("loading reranker %s on %s", self._model_name, self._device)
            try:
                self._model = CrossEncoder(
                    self._model_name, device=self._device, max_length=self._max_length
                )
            except Exception as exc:  # noqa: BLE001 - surface offline cache misses clearly
                raise RerankerUnavailableError(
                    f"could not load reranker {self._model_name!r}. When running offline the "
                    "model must already be in the model cache — set "
                    "RAG_EMBED_RERANK_ENABLED=true and run `make prefetch` once while online."
                ) from exc
        return self._model

    def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]:
        """Score every (query, chunk) pair and return the best ``top_n``."""
        if not hits:
            return []
        scores = self._get_model().predict(
            [(query, hit.text) for hit in hits],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return _apply_scores(hits, [float(s) for s in scores], top_n=top_n)


def _apply_scores(hits: list[SearchHit], scores: list[float], *, top_n: int) -> list[SearchHit]:
    """Attach rerank scores, sort, truncate. Pure — so ordering is unit-testable."""
    rescored = [
        hit.model_copy(
            update={
                "score": score,
                # Keep the fused score rather than overwrite it: the two are on
                # different scales and comparing them is how you tell whether
                # reranking actually changed anything.
                "retrieval_score": hit.retrieval_score if hit.retrieval_score is not None else hit.score,
                "source": f"{hit.source}+rerank",
            }
        )
        for hit, score in zip(hits, scores, strict=True)
    ]
    rescored.sort(key=lambda h: h.score, reverse=True)
    return rescored[:top_n]


def build_reranker(cfg: EmbeddingConfig) -> Reranker | None:
    """The configured cross-encoder, or ``None`` when reranking is switched off."""
    if not cfg.rerank_enabled:
        return None
    return CrossEncoderReranker(
        cfg.rerank_model,
        device=cfg.rerank_device,
        batch_size=cfg.rerank_batch_size,
        max_length=cfg.rerank_max_length,
    )
