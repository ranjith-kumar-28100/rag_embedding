from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Both metadatas: the service owns jobs/collections/eval, the library owns the
# parent docstore table. They share one database, so they share one migration
# history.
from rag_embedding.store.docstore import DocstoreBase
from service_embed import models  # noqa: F401  (registers mappers)
from service_embed.config import get_settings
from service_embed.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Keep Alembic pointed at the same DB the service uses.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = [Base.metadata, DocstoreBase.metadata]


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,  # SQLite needs batch mode for ALTER
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
