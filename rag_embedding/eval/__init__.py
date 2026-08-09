"""Retrieval evaluation: deterministic metrics + an optional LLM-judged tier."""

from __future__ import annotations

from .metrics import (
    RetrievedContext,
    aggregate,
    compute_all,
    context_precision,
    context_recall,
    hit_rate,
    reciprocal_rank,
)
from .ragas_eval import RagasEvaluator

__all__ = [
    "RagasEvaluator",
    "RetrievedContext",
    "aggregate",
    "compute_all",
    "context_precision",
    "context_recall",
    "hit_rate",
    "reciprocal_rank",
]
