"""Air-gapped / offline enforcement.

Mirrors ``rag_ingestion.offline``. When enabled, local model libraries are told
never to reach the network: a missing model *fails loudly* instead of being
silently downloaded, and telemetry is off. Qdrant is a local container, BM25 is
pure arithmetic, so with the default providers nothing leaves the machine.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("rag_embedding.offline")

# key -> value forced on. setdefault-style so an explicit env override wins.
_OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",  # huggingface_hub: no hub calls
    "TRANSFORMERS_OFFLINE": "1",  # transformers: no hub calls
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "LANGCHAIN_TRACING_V2": "false",  # no LangSmith tracing
    "LANGCHAIN_TRACING": "false",
    "ANONYMIZED_TELEMETRY": "False",  # various libs' analytics opt-out
}


def enforce_offline() -> None:
    """Force offline/no-telemetry env for local model libraries.

    Must run before the first import of transformers/sentence-transformers, which
    happens lazily on first embed — so calling this when building an embedder or
    a retriever is early enough. For the strongest guarantee, also export these
    in the process environment before launching Python (Docker Compose does).
    """
    applied: list[str] = []
    for key, value in _OFFLINE_ENV.items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    if applied:
        logger.info("offline mode: set %s", ", ".join(applied))
