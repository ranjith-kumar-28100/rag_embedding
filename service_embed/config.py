"""Service infrastructure settings (env prefix ``RAG_EMBED_SVC_``).

Only infra lives here — broker URL, DB URL, filesystem locations, the UI's API
base. Everything about *embedding, indexing and retrieval* stays in
``rag_embedding.EmbeddingConfig`` (prefix ``RAG_EMBED_``), so the two concerns
never bleed into each other. Same split as stage 1's RAG_SVC_ / RAG_INGEST_.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_EMBED_SVC_",
        env_file=".env",
        extra="ignore",
    )

    # message queue / result backend ------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # persistence -------------------------------------------------------------
    # SQLAlchemy URL. Default is a local SQLite file (which also holds the parent
    # docstore); swap for Postgres without touching the ORM models.
    database_url: str = "sqlite:///./data/rag_embedding.db"

    # filesystem --------------------------------------------------------------
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")

    # UI -> API ----------------------------------------------------------------
    api_base_url: str = "http://localhost:8001"

    def ensure_dirs(self) -> None:
        for directory in (self.data_dir, self.upload_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> ServiceSettings:
    settings = ServiceSettings()
    settings.ensure_dirs()
    return settings
