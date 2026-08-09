"""Parent docstore backends: SQLite (default), Qdrant payload-only, in-memory."""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import create_engine

from rag_embedding.models import Chunk
from rag_embedding.store.docstore import (
    InMemoryParentDocStore,
    ParentDocStore,
    QdrantParentDocStore,
    SqliteParentDocStore,
)

from conftest import make_chunk


def _parents() -> list[Chunk]:
    return [
        make_chunk("p1", "First parent body", role="parent", parent_id=None),
        make_chunk("p2", "Second parent body", role="parent", parent_id=None, doc_id="doc2"),
    ]


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SqliteParentDocStore:
    store = SqliteParentDocStore(create_engine(f"sqlite:///{tmp_path / 'docs.db'}"))
    store.create_schema()
    return store


@pytest.fixture
def qdrant_store(qdrant: QdrantClient) -> QdrantParentDocStore:
    store = QdrantParentDocStore(qdrant, "parents")
    store.create_schema()
    return store


@pytest.fixture(params=["sqlite", "qdrant", "memory"])
def any_store(
    request: pytest.FixtureRequest,
    sqlite_store: SqliteParentDocStore,
    qdrant_store: QdrantParentDocStore,
) -> ParentDocStore:
    backends: dict[str, ParentDocStore] = {
        "sqlite": sqlite_store,
        "qdrant": qdrant_store,
        "memory": InMemoryParentDocStore(),
    }
    return backends[request.param]


def test_upsert_then_get(any_store: ParentDocStore) -> None:
    assert any_store.upsert(_parents()) == 2
    fetched = any_store.get_many(["p1", "p2"])
    assert set(fetched) == {"p1", "p2"}
    assert fetched["p1"].text == "First parent body"
    assert fetched["p2"].doc_id == "doc2"


def test_upsert_is_idempotent(any_store: ParentDocStore) -> None:
    any_store.upsert(_parents())
    any_store.upsert(_parents())
    assert any_store.count() == 2


def test_upsert_updates_existing_text(any_store: ParentDocStore) -> None:
    any_store.upsert(_parents())
    revised = _parents()[0].model_copy(update={"text": "Revised body"})
    any_store.upsert([revised])
    assert any_store.get_many(["p1"])["p1"].text == "Revised body"
    assert any_store.count() == 2


def test_missing_ids_are_simply_absent(any_store: ParentDocStore) -> None:
    any_store.upsert(_parents())
    assert set(any_store.get_many(["p1", "nope"])) == {"p1"}


def test_empty_inputs(any_store: ParentDocStore) -> None:
    assert any_store.upsert([]) == 0
    assert any_store.get_many([]) == {}


def test_round_trip_preserves_provenance(any_store: ParentDocStore) -> None:
    parent = make_chunk(
        "p3",
        "table parent",
        role="parent",
        parent_id=None,
        block_type="table",
        file_type="xlsx",
        section_path=["Sheet1", "Revenue"],
    )
    any_store.upsert([parent])
    restored = any_store.get_many(["p3"])["p3"]
    assert restored.section_path == ["Sheet1", "Revenue"]
    assert restored.file_type == "xlsx"
    assert restored.block_type.value == "table"


def test_all_backends_satisfy_the_protocol(
    sqlite_store: SqliteParentDocStore, qdrant_store: QdrantParentDocStore
) -> None:
    for store in (sqlite_store, qdrant_store, InMemoryParentDocStore()):
        assert isinstance(store, ParentDocStore)
