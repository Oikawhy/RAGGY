from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.generation.llm import FakeLLMClient, LLMClient
from app.generation.prompt import build_llm_prompt
from app.generation.validator import compute_confidence, decide_fallback_reason, detect_llm_insufficiency
from app.models import AskRequest, AskResponse, SourceRef
from app.pipeline.context import ContextItem
from app.pipeline.question_analyzer import analyze_question


Retriever = Callable[[str], list[ContextItem]]


class PipelineOrchestrator:
    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        context_chunks: list[ContextItem] | None = None,
        llm_client: LLMClient | None = None,
        calculation_supported: bool = True,
        trace_writer: Any | None = None,
    ) -> None:
        self.retriever = retriever or (lambda question: context_chunks or [])
        self.llm_client = llm_client or FakeLLMClient(
            {"answer": "У базі знань недостатньо даних для точної відповіді.", "fallback_reason": None, "used_sections": []}
        )
        self.calculation_supported = calculation_supported
        self.trace_writer = trace_writer

    def ask(self, request: AskRequest) -> AskResponse:
        started = perf_counter()
        trace_id = f"trace-{uuid4()}"
        analysis = analyze_question(request.question)
        context = self.retriever(request.question)

        if analysis.requires_calculation and not self.calculation_supported:
            return self._fallback_response(
                trace_id=trace_id,
                started=started,
                reason="У базі знань недостатньо даних для виконання розрахунку.",
            )

        if not context:
            return self._fallback_response(
                trace_id=trace_id,
                started=started,
                reason="У базі знань недостатньо релевантного контексту для точної відповіді.",
            )

        prompt = build_llm_prompt(context, request.question)
        llm_response = self.llm_client.generate(prompt)
        llm_insufficient = detect_llm_insufficiency(llm_response.answer)
        confidence = compute_confidence(
            top_rrf_score=max(item.score for item in context),
            supporting_chunks=len(context),
            vector_lexical_overlap=1.0,
            has_disclaimer=any("недостатньо" in item.content.casefold() or "does not contain" in item.content.casefold() for item in context),
            llm_signals_insufficiency=llm_insufficient,
        )
        fallback_reason = decide_fallback_reason(confidence, llm_response.fallback_reason)
        response = AskResponse(
            answer=llm_response.answer,
            sources=[
                SourceRef(
                    section=item.section,
                    chunk=item.content,
                    score=item.score,
                    chunk_id=item.chunk_id,
                    content_hash=item.content_hash,
                    preview=item.content[:160],
                )
                for item in context
            ],
            confidence=confidence,
            fallback_reason=fallback_reason,
            trace_id=trace_id,
            latency_ms=self._latency_ms(started),
        )
        self._trace(request, response)
        return response

    def _fallback_response(self, *, trace_id: str, started: float, reason: str) -> AskResponse:
        response = AskResponse(
            answer=reason,
            sources=[],
            confidence="low",
            fallback_reason=reason,
            trace_id=trace_id,
            latency_ms=self._latency_ms(started),
        )
        self._trace(None, response)
        return response

    def _latency_ms(self, started: float) -> int:
        return max(0, round((perf_counter() - started) * 1000))

    def _trace(self, request: AskRequest | None, response: AskResponse) -> None:
        if self.trace_writer is None:
            return
        self.trace_writer.write_sync(
            {
                "trace_id": response.trace_id,
                "question": request.question if request else None,
                "pipeline_steps": [],
                "response": response.model_dump(),
                "total_latency_ms": response.latency_ms,
                "error": None,
            }
        )
