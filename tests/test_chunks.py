"""Chunk JSONL loading, the parent/child split, and payload round-tripping."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from rag_embedding.chunks import (
    ChunkParseError,
    batched,
    chunk_from_payload,
    count_lines,
    load_chunks,
    split_roles,
)
from rag_embedding.models import Chunk, ChunkRole, point_id_for

SPEC_SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_chunks.jsonl"


def test_loads_the_spec_appendix_sample() -> None:
    chunks = load_chunks(SPEC_SAMPLE)
    assert len(chunks) == 5
    roles = [c.role for c in chunks]
    assert roles.count(ChunkRole.PARENT) == 1
    assert roles.count(ChunkRole.CHILD) == 4
    row_chunk = next(c for c in chunks if c.chunk_id == "r0w123")
    assert row_chunk.row_range == (1, 10)
    assert row_chunk.metadata == {"routing": "row_chunking"}


def test_reads_gzip(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(SPEC_SAMPLE.read_text())
    assert len(load_chunks(path)) == 5


def test_malformed_lines_are_skipped_not_fatal(tmp_path: Path) -> None:
    good = SPEC_SAMPLE.read_text().splitlines()[0]
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join([good, "{not json", "", '{"chunk_id": "missing-fields"}', good]))
    chunks = load_chunks(path)
    assert len(chunks) == 2  # the two good lines; the rest logged and dropped


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(ChunkParseError):
        load_chunks(path)


def test_count_lines_ignores_blanks(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n')
    assert count_lines(path) == 2


def test_unknown_fields_are_ignored() -> None:
    """Stage 1 can add fields without breaking this service (§3 contract rule)."""
    raw = json.loads(SPEC_SAMPLE.read_text().splitlines()[1])
    raw["some_future_field"] = {"nested": True}
    chunk = Chunk.model_validate(raw)
    assert chunk.chunk_id == "a1b2c3"


def test_split_drops_empty_children(sample_chunks: list[Chunk]) -> None:
    blank = sample_chunks[1].model_copy(update={"chunk_id": "blank", "text": "   "})
    split = split_roles([*sample_chunks, blank])
    assert len(split.parents) == 1
    assert len(split.children) == 4
    assert split.skipped == 1
    assert "blank" in split.errors[0]


def test_point_id_is_deterministic_and_distinct() -> None:
    assert point_id_for("abc") == point_id_for("abc")
    assert point_id_for("abc") != point_id_for("abd")


def test_payload_round_trips(sample_chunks: list[Chunk]) -> None:
    original = sample_chunks[3]  # the table chunk, non-default doc/file/section
    restored = chunk_from_payload(original.to_payload())
    assert restored == original


def test_payload_flattens_metadata_and_joins_section_path() -> None:
    chunk = Chunk(
        chunk_id="x",
        doc_id="d",
        parent_id=None,
        role=ChunkRole.CHILD,
        block_type="text",
        text="t",
        section_path=["A", "B"],
        source_uri="file:///x",
        file_type="pdf",
        token_count=1,
        metadata={"tags": ["a", "b"], "n": 3, "obj": {"k": "v"}},
    )
    payload = chunk.to_payload()
    assert payload["section_path"] == ["A", "B"]
    assert payload["section_path_text"] == "A > B"
    assert payload["metadata"]["tags"] == "a > b"
    assert payload["metadata"]["n"] == 3
    assert isinstance(payload["metadata"]["obj"], str)


def test_batched_covers_every_item(sample_chunks: list[Chunk]) -> None:
    batches = list(batched(sample_chunks, 2))
    assert [len(b) for b in batches] == [2, 2, 1]
    assert sum(len(b) for b in batches) == len(sample_chunks)


def test_batched_rejects_zero(sample_chunks: list[Chunk]) -> None:
    with pytest.raises(ValueError, match="batch size"):
        list(batched(sample_chunks, 0))
