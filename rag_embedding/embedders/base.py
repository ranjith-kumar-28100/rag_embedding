"""Embedder protocols — the seam every provider implements.

Keeping these as ``Protocol`` (not base classes) is what lets tests swap in a
deterministic fake embedder with no inheritance and no model download, and what
keeps the library provider-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SparseVector(BaseModel):
    """A BM25 vector in Qdrant's sparse form: parallel indices/values arrays."""

    indices: list[int]
    values: list[float]

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.indices, self.values, strict=True))


@runtime_checkable
class DenseEmbedder(Protocol):
    """Semantic vectors. Implementations must return L2-normalized rows (§5.1)."""

    @property
    def model_id(self) -> str:
        """Fingerprint recorded on the collection; a mismatch blocks search."""
        ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Lexical vectors. Documents and queries are weighted differently (§5.2)."""

    def embed_documents(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


class EmbedderError(RuntimeError):
    """Provider misconfiguration or a failed model load."""
