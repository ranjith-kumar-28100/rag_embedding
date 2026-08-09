"""rag_embedding microservice: FastAPI + Celery + SQLite around the library.

A thin async wrapper. It adds job orchestration, a queue for backpressure under
heavy indexing, progress tracking, the collection registry and eval history — it
never re-implements embedding, indexing or retrieval. With the default providers
all processing stays on the machine.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
