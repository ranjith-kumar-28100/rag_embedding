"""Indexing pipeline: role split, batching, progress, idempotence, reindex."""

from __future__ import annotations

from pathlib import Path

from rag_embedding.config import EmbeddingConfig
from rag_embedding.indexer import Indexer
from rag_embedding.models import Chunk
from rag_embedding.store.docstore import InMemoryParentDocStore
from rag_embedding.store.qdrant_store import QdrantStore

from conftest import FIXTURE_COLLECTION, FakeDenseEmbedder, make_chunk

SPEC_SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_chunks.jsonl"


def test_only_children_are_embedded(
    indexer: Indexer, sample_chunks: list[Chunk], store: QdrantStore,
    docstore: InMemoryParentDocStore,
) -> None:
    indexer.ensure_collection(FIXTURE_COLLECTION)
    stats = indexer.index_chunks(FIXTURE_COLLECTION, sample_chunks)
    assert stats.children_indexed == 4
    assert stats.parents_stored == 1
    assert store.count(FIXTURE_COLLECTION) == 4  # the parent is not a point
    assert docstore.count() == 1


def test_collection_spec_records_the_model_fingerprint(indexer: Indexer) -> None:
    spec = indexer.collection_spec("x")
    assert spec.model_id == "fake:fake-embedder"
    assert spec.dim == 32
    assert spec.provider == "sentence_transformers"
    assert spec.quantization == "int8"
    assert spec.sparse_enabled is True


def test_embedding_is_batched(
    indexer: Indexer, sample_chunks: list[Chunk], fake_dense: FakeDenseEmbedder
) -> None:
    """batch_size=2 over 4 children => 2 calls, not 1 and not 4."""
    indexer.ensure_collection(FIXTURE_COLLECTION)
    indexer.index_chunks(FIXTURE_COLLECTION, sample_chunks)
    assert fake_dense.call_count == 2


def test_progress_counts_children_only(
    indexer: Indexer, sample_chunks: list[Chunk]
) -> None:
    seen: list[tuple[int, int]] = []
    indexer.ensure_collection(FIXTURE_COLLECTION)
    indexer.index_chunks(FIXTURE_COLLECTION, sample_chunks, progress=lambda p, t: seen.append((p, t)))
    assert seen[0] == (0, 4)
    assert seen[-1] == (4, 4)
    assert [p for p, _ in seen] == sorted(p for p, _ in seen)  # monotonic


def test_reindexing_the_same_chunks_does_not_duplicate(
    indexer: Indexer, sample_chunks: list[Chunk], store: QdrantStore
) -> None:
    indexer.ensure_collection(FIXTURE_COLLECTION)
    indexer.index_chunks(FIXTURE_COLLECTION, sample_chunks)
    indexer.index_chunks(FIXTURE_COLLECTION, sample_chunks)
    assert store.count(FIXTURE_COLLECTION) == 4


def test_index_file_reads_the_spec_sample(indexer: Indexer, store: QdrantStore) -> None:
    indexer.ensure_collection(FIXTURE_COLLECTION)
    stats = indexer.index_file(FIXTURE_COLLECTION, SPEC_SAMPLE)
    assert stats.total_chunks == 5
    assert stats.children_indexed == 4
    assert stats.parents_stored == 1
    assert stats.seconds >= 0


def test_malformed_lines_surface_as_a_warning(indexer: Indexer, tmp_path: Path) -> None:
    good = SPEC_SAMPLE.read_text().splitlines()[1]
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join([good, "{broken", good.replace("a1b2c3", "a1b2c4")]))
    indexer.ensure_collection(FIXTURE_COLLECTION)
    stats = indexer.index_file(FIXTURE_COLLECTION, path)
    assert stats.children_indexed == 2
    assert any("malformed" in w for w in stats.warnings)


def test_empty_children_are_skipped_with_a_warning(indexer: Indexer) -> None:
    indexer.ensure_collection(FIXTURE_COLLECTION)
    stats = indexer.index_chunks(
        FIXTURE_COLLECTION, [make_chunk("blank", "   "), make_chunk("ok", "real content")]
    )
    assert stats.children_indexed == 1
    assert stats.skipped == 1


# --------------------------------------------------------------------------- #
# Reindex (§9)
# --------------------------------------------------------------------------- #
def test_reindex_rebuilds_from_stored_payloads(
    indexed: Indexer, store: QdrantStore
) -> None:
    """No re-upload: the child text is read back out of the old collection."""
    stats = indexed.reindex_from(FIXTURE_COLLECTION, "test_rag_v2")
    assert stats.children_indexed == 4
    assert store.count("test_rag_v2") == 4
    assert (
        store.retrieve_payloads("test_rag_v2", ["c1"])["c1"]["text"]
        == store.retrieve_payloads(FIXTURE_COLLECTION, ["c1"])["c1"]["text"]
    )


def test_reindex_to_a_new_dimension(
    indexed: Indexer,
    cfg: EmbeddingConfig,
    store: QdrantStore,
    bm25: object,
    docstore: InMemoryParentDocStore,
) -> None:
    """The point of reindexing: a different model, hence a different dim."""
    new_indexer = Indexer(
        cfg=cfg,
        store=store,
        dense=FakeDenseEmbedder(dim=64, model_name="bigger"),
        sparse=bm25,  # type: ignore[arg-type]
        docstore=docstore,
    )
    new_indexer.reindex_from(FIXTURE_COLLECTION, "test_rag_v2")
    assert store.dense_dim("test_rag_v2") == 64
    assert store.dense_dim(FIXTURE_COLLECTION) == 32  # old one untouched


def test_reindex_reports_progress(indexed: Indexer) -> None:
    seen: list[tuple[int, int]] = []
    indexed.reindex_from("test_rag_v1", "test_rag_v2", progress=lambda p, t: seen.append((p, t)))
    assert seen[0] == (0, 4)
    assert seen[-1][0] == 4


def test_activate_points_the_alias(indexed: Indexer, store: QdrantStore) -> None:
    indexed.reindex_from(FIXTURE_COLLECTION, "test_rag_v2")
    alias = indexed.activate("test_rag_v2")
    assert alias == "test_rag"
    assert store.resolve_alias("test_rag") == "test_rag_v2"


def test_old_collection_still_serves_until_the_swap(
    indexed: Indexer, store: QdrantStore
) -> None:
    indexed.activate(FIXTURE_COLLECTION)
    indexed.reindex_from(FIXTURE_COLLECTION, "test_rag_v2")
    assert store.resolve_alias("test_rag") == FIXTURE_COLLECTION  # zero downtime
    indexed.activate("test_rag_v2")
    assert store.resolve_alias("test_rag") == "test_rag_v2"
