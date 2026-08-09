"""RAGAS evaluation harness (spec §10).

Two tiers, deliberately separated because they are not equally trustworthy:

  * **Deterministic** (``metrics.py``): hit-rate@k, MRR, context_precision@k,
    context_recall. Pure arithmetic over ranks. These always run when the eval
    item carries ``ground_truth_contexts``, and they are the signal to steer by
    offline.
  * **LLM-judged** (RAGAS proper): ``context_relevancy`` style metrics, needing a
    judge LLM. Opt-in per request. With a small local judge (Mistral 7B via
    Ollama) these are *indicative, not authoritative* — that is a property of the
    judge, not of the harness, so the report says so in ``warnings`` rather than
    dressing the numbers up.

Answer-based metrics (faithfulness, answer_relevancy) need generation, which is
deferred (§18); requesting them without ``generation_enabled`` is reported as a
skipped metric rather than silently omitted.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import EmbeddingConfig
from ..models import (
    EvalItemResult,
    EvalReport,
    EvalRequest,
    SearchRequest,
    SearchResponse,
)
from ..retrieval import Retriever
from .metrics import RetrievedContext, aggregate, compute_all

logger = logging.getLogger("rag_embedding.eval")

JUDGE_CAVEAT = (
    "LLM-judged metrics come from a local judge and are indicative, not authoritative. "
    "Steer by hit_rate@k / MRR / context_recall, which need no LLM."
)


class RagasEvaluator:
    def __init__(self, cfg: EmbeddingConfig, retriever: Retriever) -> None:
        self.cfg = cfg
        self.retriever = retriever

    def evaluate(self, request: EvalRequest) -> EvalReport:
        top_k = request.top_k or self.cfg.top_k
        with_parents = (
            self.cfg.with_parents if request.with_parents is None else request.with_parents
        )

        items: list[EvalItemResult] = []
        scored: list[dict[str, float]] = []
        warnings: list[str] = []
        unreferenced = 0

        for question in request.dataset:
            response = self.retriever.search(
                SearchRequest(
                    query=question.question,
                    top_k=top_k,
                    mode=request.mode,
                    with_parents=with_parents,
                    rerank=request.rerank,
                )
            )
            contexts = _contexts_for(response, with_parents=with_parents)
            result = EvalItemResult(
                question=question.question,
                retrieved_chunk_ids=[c.chunk_id for c in contexts],
            )
            if question.ground_truth_contexts:
                result.metrics = compute_all(
                    question.ground_truth_contexts, contexts, top_k=top_k
                )
                scored.append(result.metrics)
            else:
                unreferenced += 1
                result.note = "no ground_truth_contexts — reference-based metrics skipped"
            items.append(result)

        if unreferenced:
            warnings.append(
                f"{unreferenced}/{len(request.dataset)} question(s) had no ground_truth_contexts; "
                "only retrieval traces were recorded for them."
            )

        report = EvalReport(
            collection=self.retriever.collection,
            mode=request.mode,
            top_k=top_k,
            aggregate=aggregate(scored),
            items=items,
            warnings=warnings,
        )

        if request.llm_metrics or self.cfg.ragas_llm_metrics_enabled:
            self._add_llm_metrics(request, report)
        if not self.cfg.generation_enabled and any(
            q.reference_answer for q in request.dataset
        ):
            report.warnings.append(
                "reference_answer supplied but generation is disabled (§18): faithfulness / "
                "answer_relevancy skipped. Set RAG_EMBED_GENERATION_ENABLED=true to enable."
            )
        return report

    # -- LLM-judged tier ---------------------------------------------------- #
    def _add_llm_metrics(self, request: EvalRequest, report: EvalReport) -> None:
        """Run RAGAS's judged metrics if the optional dependency is installed."""
        try:
            judged = self._run_ragas(request, report)
        except _RagasUnavailable as exc:
            report.warnings.append(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - a failed judge must not lose the run
            logger.exception("RAGAS LLM metrics failed")
            report.warnings.append(f"LLM-judged metrics failed and were skipped: {exc}")
            return

        report.aggregate.update(judged)
        report.warnings.append(JUDGE_CAVEAT)

    def _run_ragas(self, request: EvalRequest, report: EvalReport) -> dict[str, float]:
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import LLMContextRecall, ResponseRelevancy  # noqa: F401
        except ImportError as exc:
            raise _RagasUnavailable(
                'LLM-judged metrics need the optional extra: uv pip install -e ".[eval]" '
                "— deterministic metrics were still computed."
            ) from exc

        judge = self._build_judge()
        if judge is None:
            raise _RagasUnavailable(
                f"ragas_llm_provider={self.cfg.ragas_llm_provider!r} could not be built; "
                "LLM-judged metrics skipped."
            )

        # Only reference-bearing questions can be judged for recall.
        rows = [
            {
                "user_input": item.question,
                "retrieved_contexts": item.retrieved_chunk_ids,
                "reference": " ".join(question.ground_truth_contexts),
            }
            for item, question in zip(report.items, request.dataset, strict=True)
            if question.ground_truth_contexts
        ]
        if not rows:
            raise _RagasUnavailable(
                "no question carried ground_truth_contexts; nothing for the judge to score."
            )

        from ragas import EvaluationDataset

        result = ragas_evaluate(
            dataset=EvaluationDataset.from_list(rows),
            metrics=[LLMContextRecall()],
            llm=judge,
        )
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        return {f"llm_{key}": round(float(value), 4) for key, value in scores.items()}

    def _build_judge(self) -> Any:
        provider = self.cfg.ragas_llm_provider.lower()
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            from ragas.llms import LangchainLLMWrapper

            return LangchainLLMWrapper(
                ChatOllama(model=self.cfg.ragas_model, base_url=self.cfg.ollama_base_url)
            )
        if provider == "azure":
            if self.cfg.offline:  # defence in depth; config already rejects this
                raise _RagasUnavailable("azure judge requested while offline=true")
            from langchain_openai import AzureChatOpenAI
            from ragas.llms import LangchainLLMWrapper

            return LangchainLLMWrapper(
                AzureChatOpenAI(
                    azure_endpoint=self.cfg.azure_openai_endpoint,
                    api_key=self.cfg.azure_openai_api_key,
                    azure_deployment=self.cfg.ragas_judge_deployment,
                    api_version=self.cfg.azure_openai_api_version,
                )
            )
        return None


class _RagasUnavailable(RuntimeError):
    """The judged tier could not run — recorded as a warning, never fatal."""


def _contexts_for(response: SearchResponse, *, with_parents: bool) -> list[RetrievedContext]:
    """Score what the caller would actually feed a generator: parents if enabled."""
    if with_parents and response.parents:
        return [RetrievedContext(chunk_id=p.chunk_id, text=p.text) for p in response.parents]
    return [RetrievedContext(chunk_id=h.chunk_id, text=h.text) for h in response.hits]
