"""Deterministic retrieval metrics (spec §10.2, reliable tier).

These need no LLM, so they are the metrics you can actually trust offline: given
``ground_truth_contexts`` they are arithmetic over ranks, and they mean the same
thing on every run. The LLM-judged tier lives in ``ragas_eval`` and is opt-in.

Matching a retrieved context against a reference is the one judgement call here.
References may be written either as chunk ids (exact) or as literal context text
(from a doc, rarely byte-identical to the chunk). So: exact id match first, then
substring containment either way, then a token overlap coefficient — the last of
which is why ``match_threshold`` is a parameter and not a constant.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

DEFAULT_MATCH_THRESHOLD = 0.8


def normalize(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def overlap_coefficient(a: str, b: str) -> float:
    """|A∩B| / min(|A|,|B|) — tolerant of a reference much shorter than the chunk."""
    tokens_a, tokens_b = set(a.split()), set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


@dataclass(frozen=True)
class RetrievedContext:
    chunk_id: str
    text: str


def matches(
    reference: str,
    context: RetrievedContext,
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> bool:
    """Does this retrieved context satisfy this reference?"""
    if reference == context.chunk_id:
        return True
    ref_norm, ctx_norm = normalize(reference), normalize(context.text)
    if not ref_norm or not ctx_norm:
        return False
    if ref_norm in ctx_norm or ctx_norm in ref_norm:
        return True
    return overlap_coefficient(ref_norm, ctx_norm) >= threshold


def first_relevant_rank(
    references: Sequence[str],
    contexts: Sequence[RetrievedContext],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> int | None:
    """1-based rank of the first retrieved context matching any reference."""
    for rank, context in enumerate(contexts, start=1):
        if any(matches(ref, context, threshold=threshold) for ref in references):
            return rank
    return None


def hit_rate(
    references: Sequence[str],
    contexts: Sequence[RetrievedContext],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float:
    """1.0 if anything relevant made the top-k at all."""
    return 1.0 if first_relevant_rank(references, contexts, threshold=threshold) else 0.0


def reciprocal_rank(
    references: Sequence[str],
    contexts: Sequence[RetrievedContext],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float:
    """1/rank of the first relevant context — rewards ranking it *first*, not just retrieving it."""
    rank = first_relevant_rank(references, contexts, threshold=threshold)
    return 1.0 / rank if rank else 0.0


def context_precision(
    references: Sequence[str],
    contexts: Sequence[RetrievedContext],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float:
    """Fraction of retrieved contexts that are relevant — how much noise came back."""
    if not contexts:
        return 0.0
    relevant = sum(
        1
        for context in contexts
        if any(matches(ref, context, threshold=threshold) for ref in references)
    )
    return relevant / len(contexts)


def context_recall(
    references: Sequence[str],
    contexts: Sequence[RetrievedContext],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float:
    """Fraction of references that were retrieved — how much was missed."""
    if not references:
        return 0.0
    found = sum(
        1
        for ref in references
        if any(matches(ref, context, threshold=threshold) for context in contexts)
    )
    return found / len(references)


def compute_all(
    references: Sequence[str],
    contexts: Sequence[RetrievedContext],
    *,
    top_k: int,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, float]:
    """Every reference-based metric for one question."""
    return {
        f"hit_rate@{top_k}": hit_rate(references, contexts, threshold=threshold),
        "mrr": reciprocal_rank(references, contexts, threshold=threshold),
        f"context_precision@{top_k}": context_precision(references, contexts, threshold=threshold),
        "context_recall": context_recall(references, contexts, threshold=threshold),
    }


def aggregate(per_item: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across questions that reported it."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in per_item:
        for key, value in item.items():
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: round(totals[key] / counts[key], 4) for key in sorted(totals)}
