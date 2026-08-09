"""Deterministic retrieval metrics and the eval harness's honesty about tiers."""

from __future__ import annotations

import pytest

from rag_embedding.config import EmbeddingConfig
from rag_embedding.eval.metrics import (
    RetrievedContext,
    aggregate,
    compute_all,
    context_precision,
    context_recall,
    first_relevant_rank,
    hit_rate,
    matches,
    normalize,
    overlap_coefficient,
    reciprocal_rank,
)
from rag_embedding.eval.ragas_eval import RagasEvaluator
from rag_embedding.indexer import Indexer
from rag_embedding.models import EvalQuestion, EvalRequest
from rag_embedding.retrieval import Retriever


def ctx(chunk_id: str, text: str = "") -> RetrievedContext:
    return RetrievedContext(chunk_id=chunk_id, text=text or chunk_id)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def test_normalize_strips_punctuation_and_case() -> None:
    assert normalize("The  Retrieval, Subsystem!") == "the retrieval subsystem"


def test_overlap_coefficient() -> None:
    assert overlap_coefficient("a b c", "a b") == 1.0
    assert overlap_coefficient("a b c d", "a x") == 0.5
    assert overlap_coefficient("", "a") == 0.0


def test_reference_matches_by_chunk_id() -> None:
    assert matches("c1", ctx("c1", "totally different text"))


def test_reference_matches_by_substring() -> None:
    assert matches("fuses dense and sparse", ctx("c1", "The subsystem fuses dense and sparse."))


def test_reference_matches_by_token_overlap() -> None:
    assert matches("dense sparse fusion signals", ctx("c1", "dense sparse fusion of signals"))


def test_unrelated_text_does_not_match() -> None:
    assert not matches("quarterly revenue table", ctx("c1", "HNSW graph traversal parameters"))


def test_threshold_is_tunable() -> None:
    reference, context = "alpha beta gamma delta", ctx("c1", "alpha beta epsilon zeta")
    assert not matches(reference, context, threshold=0.9)
    assert matches(reference, context, threshold=0.4)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_first_relevant_rank_is_one_based() -> None:
    assert first_relevant_rank(["c3"], [ctx("c1"), ctx("c2"), ctx("c3")]) == 3
    assert first_relevant_rank(["nope"], [ctx("c1")]) is None


def test_hit_rate() -> None:
    assert hit_rate(["c2"], [ctx("c1"), ctx("c2")]) == 1.0
    assert hit_rate(["c9"], [ctx("c1"), ctx("c2")]) == 0.0


def test_mrr_rewards_ranking_first() -> None:
    assert reciprocal_rank(["c1"], [ctx("c1"), ctx("c2")]) == 1.0
    assert reciprocal_rank(["c2"], [ctx("c1"), ctx("c2")]) == 0.5
    assert reciprocal_rank(["c9"], [ctx("c1")]) == 0.0


def test_context_precision_measures_noise() -> None:
    assert context_precision(["c1", "c2"], [ctx("c1"), ctx("c2")]) == 1.0
    assert context_precision(["c1"], [ctx("c1"), ctx("c2"), ctx("c3"), ctx("c4")]) == 0.25
    assert context_precision(["c1"], []) == 0.0


def test_context_recall_measures_misses() -> None:
    assert context_recall(["c1", "c2"], [ctx("c1")]) == 0.5
    assert context_recall(["c1", "c2"], [ctx("c1"), ctx("c2")]) == 1.0
    assert context_recall([], [ctx("c1")]) == 0.0


def test_compute_all_keys_include_k() -> None:
    metrics = compute_all(["c1"], [ctx("c1")], top_k=5)
    assert set(metrics) == {"hit_rate@5", "mrr", "context_precision@5", "context_recall"}


def test_aggregate_means_each_metric() -> None:
    assert aggregate([{"mrr": 1.0}, {"mrr": 0.0}]) == {"mrr": 0.5}


def test_aggregate_of_nothing_is_empty() -> None:
    assert aggregate([]) == {}


def test_aggregate_handles_metrics_missing_from_some_items() -> None:
    """A metric is averaged over the items that reported it, not over all items."""
    assert aggregate([{"a": 1.0, "b": 0.0}, {"a": 0.0}]) == {"a": 0.5, "b": 0.0}


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def test_evaluator_scores_a_referenced_question(
    indexed: Indexer, retriever: Retriever, cfg: EmbeddingConfig
) -> None:
    evaluator = RagasEvaluator(cfg, retriever)
    report = evaluator.evaluate(
        EvalRequest(
            dataset=[
                EvalQuestion(question="dense and sparse fusion", ground_truth_contexts=["p1"])
            ],
            with_parents=True,
        )
    )
    assert report.collection == retriever.collection
    assert report.aggregate["hit_rate@5"] == 1.0
    assert report.aggregate["mrr"] == 1.0
    assert report.items[0].retrieved_chunk_ids[0] == "p1"


def test_unreferenced_questions_are_noted_not_scored(
    indexed: Indexer, retriever: Retriever, cfg: EmbeddingConfig
) -> None:
    report = RagasEvaluator(cfg, retriever).evaluate(
        EvalRequest(dataset=[EvalQuestion(question="anything at all")])
    )
    assert report.aggregate == {}
    assert report.items[0].metrics == {}
    assert "no ground_truth_contexts" in (report.items[0].note or "")
    assert any("ground_truth_contexts" in w for w in report.warnings)


def test_children_are_scored_when_parents_are_off(
    indexed: Indexer, retriever: Retriever, cfg: EmbeddingConfig
) -> None:
    report = RagasEvaluator(cfg, retriever).evaluate(
        EvalRequest(
            dataset=[EvalQuestion(question="dense and sparse fusion", ground_truth_contexts=["c1"])],
            with_parents=False,
        )
    )
    assert report.items[0].retrieved_chunk_ids[0] == "c1"
    assert report.aggregate["mrr"] == 1.0


def test_reference_answer_without_generation_is_flagged(
    indexed: Indexer, retriever: Retriever, cfg: EmbeddingConfig
) -> None:
    """Answer-based metrics are deferred — say so rather than omitting silently."""
    report = RagasEvaluator(cfg, retriever).evaluate(
        EvalRequest(
            dataset=[
                EvalQuestion(
                    question="dense fusion",
                    ground_truth_contexts=["c1"],
                    reference_answer="It fuses dense and sparse.",
                )
            ]
        )
    )
    assert any("faithfulness" in w for w in report.warnings)


def test_llm_tier_degrades_to_a_warning_when_unavailable(
    indexed: Indexer, retriever: Retriever, cfg: EmbeddingConfig
) -> None:
    """A missing judge must never lose the deterministic results."""
    report = RagasEvaluator(cfg, retriever).evaluate(
        EvalRequest(
            dataset=[EvalQuestion(question="dense fusion", ground_truth_contexts=["p1"])],
            llm_metrics=True,
        )
    )
    assert report.aggregate["hit_rate@5"] == 1.0  # still there
    assert report.warnings  # and the judge's absence is reported


def test_mode_is_carried_into_the_report(
    indexed: Indexer, retriever: Retriever, cfg: EmbeddingConfig
) -> None:
    report = RagasEvaluator(cfg, retriever).evaluate(
        EvalRequest(dataset=[EvalQuestion(question="1001")], mode="sparse", top_k=3)
    )
    assert report.mode == "sparse"
    assert report.top_k == 3


@pytest.mark.parametrize("k", [1, 3, 10])
def test_top_k_appears_in_metric_names(
    indexed: Indexer, retriever: Retriever, cfg: EmbeddingConfig, k: int
) -> None:
    report = RagasEvaluator(cfg, retriever).evaluate(
        EvalRequest(
            dataset=[EvalQuestion(question="dense fusion", ground_truth_contexts=["c1"])],
            top_k=k,
        )
    )
    assert f"hit_rate@{k}" in report.aggregate
