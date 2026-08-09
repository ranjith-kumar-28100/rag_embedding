"""Parent docstore (spec §6.2) — parents are stored, never embedded.

Small-to-big retrieval ranks *children* and returns *parents* as context, so the
parent text has to live somewhere retrievable by id. Three backends behind one
Protocol:

  * ``SqliteParentDocStore`` — the default (§18). SQLAlchemy, so the same URL
    swap that moves the service to Postgres moves the docstore with it, and
    parent writes are transactional alongside job state.
  * ``QdrantParentDocStore`` — a payload-only ``<prefix>__parents`` collection
    for deployments that would rather run one datastore than two.
  * ``InMemoryParentDocStore`` — tests.

Parents are keyed by ``chunk_id`` and are *version independent*: a reindex builds
a new vector collection but reuses the same parents, which is what makes reindex
cheap and re-upload unnecessary.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from qdrant_client import QdrantClient, models
from sqlalchemy import JSON, String, Text, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ..models import Chunk, ChunkRole, point_id_for


@runtime_checkable
class ParentDocStore(Protocol):
    def upsert(self, parents: list[Chunk]) -> int: ...

    def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]: ...

    def count(self) -> int: ...


# --------------------------------------------------------------------------- #
# SQLite / SQLAlchemy (default)
# --------------------------------------------------------------------------- #
class DocstoreBase(DeclarativeBase):
    """Separate declarative base: the library owns this table, not the service.

    The service's Alembic env lists both metadatas, so migrations still cover it.
    """


class ParentChunk(DocstoreBase):
    __tablename__ = "parent_chunks"

    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class SqliteParentDocStore:
    """Transactional parent store on any SQLAlchemy engine (SQLite by default)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)

    def create_schema(self) -> None:
        """Dev/test convenience. Production goes through Alembic."""
        DocstoreBase.metadata.create_all(self._engine)

    def upsert(self, parents: list[Chunk]) -> int:
        if not parents:
            return 0
        with self._session_factory() as session, session.begin():
            existing = {
                row.chunk_id
                for row in session.execute(
                    select(ParentChunk).where(
                        ParentChunk.chunk_id.in_([p.chunk_id for p in parents])
                    )
                ).scalars()
            }
            for parent in parents:
                payload = parent.to_payload()
                if parent.chunk_id in existing:
                    row = session.get(ParentChunk, parent.chunk_id)
                    if row is not None:
                        row.doc_id = parent.doc_id
                        row.text = parent.text
                        row.payload = payload
                else:
                    session.add(
                        ParentChunk(
                            chunk_id=parent.chunk_id,
                            doc_id=parent.doc_id,
                            text=parent.text,
                            payload=payload,
                        )
                    )
        return len(parents)

    def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        with self._session_factory() as session:
            rows = session.execute(
                select(ParentChunk).where(ParentChunk.chunk_id.in_(chunk_ids))
            ).scalars()
            return {row.chunk_id: _chunk_from_row(row) for row in rows}

    def count(self) -> int:
        with self._session_factory() as session:
            return len(list(session.execute(select(ParentChunk.chunk_id)).scalars()))


def _chunk_from_row(row: ParentChunk) -> Chunk:
    from ..chunks import chunk_from_payload

    payload = dict(row.payload or {})
    payload.setdefault("chunk_id", row.chunk_id)
    payload.setdefault("doc_id", row.doc_id)
    payload.setdefault("text", row.text)
    payload.setdefault("role", ChunkRole.PARENT.value)
    return chunk_from_payload(payload)


# --------------------------------------------------------------------------- #
# Qdrant payload-only collection (opt-in)
# --------------------------------------------------------------------------- #
class QdrantParentDocStore:
    """Parents in a vector-less Qdrant collection — one datastore, same ops.

    Qdrant requires *some* vector config, so this uses a 1-dimensional dummy
    vector that is never queried: lookups go through ``retrieve`` by point id.
    """

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    def create_schema(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={},  # payload-only: no index, no vector storage
                on_disk_payload=True,
            )
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def upsert(self, parents: list[Chunk]) -> int:
        if not parents:
            return 0
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(id=p.point_id, vector={}, payload=p.to_payload())
                for p in parents
            ],
            wait=True,
        )
        return len(parents)

    def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        from ..chunks import chunk_from_payload

        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[point_id_for(cid) for cid in chunk_ids],
            with_payload=True,
            with_vectors=False,
        )
        out: dict[str, Chunk] = {}
        for record in records:
            if record.payload and "chunk_id" in record.payload:
                chunk = chunk_from_payload(dict(record.payload))
                out[chunk.chunk_id] = chunk
        return out

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)


# --------------------------------------------------------------------------- #
# In-memory (tests)
# --------------------------------------------------------------------------- #
class InMemoryParentDocStore:
    def __init__(self) -> None:
        self._rows: dict[str, Chunk] = {}

    def create_schema(self) -> None:
        return None

    def upsert(self, parents: list[Chunk]) -> int:
        for parent in parents:
            self._rows[parent.chunk_id] = parent
        return len(parents)

    def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        return {cid: self._rows[cid] for cid in chunk_ids if cid in self._rows}

    def count(self) -> int:
        return len(self._rows)
