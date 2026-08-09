"""Celery application — Redis broker + result backend.

``acks_late`` + ``prefetch_multiplier=1`` give graceful backpressure: a worker
takes exactly one file at a time and acks only when done, so a crash re-queues
the work instead of losing it. Scale throughput with more worker processes.
"""

from __future__ import annotations

from celery import Celery

from .config import get_settings

_settings = get_settings()

celery_app = Celery(
    "rag_embedding",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
    include=["service_embed.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86_400,  # keep results 1 day
)
