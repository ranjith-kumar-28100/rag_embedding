"""Chunk JSONL loader and parent/child split (spec §3, §7.2).

The only place that touches the stage-1 wire format. Reads ``.jsonl`` and
``.jsonl.gz`` streams, tolerates blank lines, and reports malformed lines
without aborting the file — a single bad line should not lose an ingest run.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from pydantic import ValidationError

from .models import Chunk, ChunkRole

logger = logging.getLogger("rag_embedding.chunks")


class ChunkParseError(ValueError):
    """Raised when a JSONL stream yields no usable chunks at all."""


@dataclass
class SplitChunks:
    """Children (to embed) and parents (to the docstore), plus reject counts."""

    children: list[Chunk] = field(default_factory=list)
    parents: list[Chunk] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.children) + len(self.parents)


def _open_text(path: Path) -> IO[str]:
    """Open plain or gzipped JSONL transparently."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_chunks(path: str | Path, *, max_errors_logged: int = 20) -> Iterator[Chunk]:
    """Yield ``Chunk`` objects from a ``.jsonl`` / ``.jsonl.gz`` file.

    Malformed lines are logged and skipped. Streaming (not a list) so a
    multi-hundred-MB upload never has to sit in memory at once.
    """
    p = Path(path)
    errors = 0
    with _open_text(p) as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield Chunk.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                errors += 1
                if errors <= max_errors_logged:
                    logger.warning("%s:%d skipped malformed chunk: %s", p.name, lineno, exc)


def load_chunks(path: str | Path) -> list[Chunk]:
    """Eagerly load a chunk file. Raises if it produced nothing usable."""
    chunks = list(iter_chunks(path))
    if not chunks:
        raise ChunkParseError(f"{path}: no valid chunks found (empty or malformed JSONL)")
    return chunks


def split_roles(chunks: Iterable[Chunk]) -> SplitChunks:
    """Partition by ``role``: children get embedded, parents go to the docstore.

    A child with an empty ``text`` is dropped — embedding whitespace produces a
    vector that matches everything weakly and pollutes recall.
    """
    out = SplitChunks()
    for chunk in chunks:
        if chunk.role is ChunkRole.PARENT:
            out.parents.append(chunk)
        elif not chunk.text.strip():
            out.skipped += 1
            out.errors.append(f"child {chunk.chunk_id} has empty text")
        else:
            out.children.append(chunk)
    return out


def count_lines(path: str | Path) -> int:
    """Cheap non-blank line count — used to size progress before parsing (§7.3)."""
    p = Path(path)
    total = 0
    with _open_text(p) as fh:
        for line in fh:
            if line.strip():
                total += 1
    return total


def batched(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    """Yield fixed-size batches (the dense embedder's unit of work)."""
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    """Rebuild a ``Chunk`` from a Qdrant payload — the reindex path (§9).

    Reindexing re-embeds existing text rather than asking stage 1 to re-emit it,
    so the payload written by :meth:`Chunk.to_payload` must round-trip.
    """
    row_start, row_end = payload.get("row_start"), payload.get("row_end")
    return Chunk(
        chunk_id=str(payload["chunk_id"]),
        doc_id=str(payload.get("doc_id", "")),
        parent_id=payload.get("parent_id"),
        role=ChunkRole(payload.get("role", "child")),
        block_type=payload.get("block_type", "text"),
        text=str(payload.get("text", "")),
        section_path=list(payload.get("section_path") or []),
        source_uri=str(payload.get("source_uri", "")),
        file_type=str(payload.get("file_type", "")),
        page=payload.get("page"),
        slide_number=payload.get("slide_number"),
        sheet_name=payload.get("sheet_name"),
        row_range=(row_start, row_end) if row_start is not None and row_end is not None else None,
        token_count=int(payload.get("token_count") or 0),
        metadata=dict(payload.get("metadata") or {}),
    )
