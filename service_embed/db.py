"""SQLAlchemy engine, session factory, and declarative base.

Sync SQLAlchemy 2.0 (typed ``Mapped[...]``). Celery workers and FastAPI request
handlers both use ``session_scope()``. SQLite runs in WAL mode so the API and
several workers can read/write concurrently without ``database is locked``.

The library's parent docstore (``rag_embedding.store.docstore``) rides on this
same engine — one file, one transaction boundary for job state and parents.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings

_settings = get_settings()

_is_sqlite = _settings.database_url.startswith("sqlite")
_connect_args: dict[str, Any] = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(
    _settings.database_url,
    connect_args=_connect_args,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


class Base(DeclarativeBase):
    pass


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    """Enable WAL + FK enforcement + a busy timeout for concurrent SQLite access."""
    if not _is_sqlite:
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def init_db() -> None:
    """Create tables for a fresh dev DB. Production uses `alembic upgrade head`."""
    from rag_embedding.store.docstore import DocstoreBase

    from . import models  # noqa: F401  (register mappers before create_all)

    Base.metadata.create_all(engine)
    DocstoreBase.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session: commit on success, rollback on error, always close."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
