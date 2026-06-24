from __future__ import annotations

import asyncio
import hashlib
import json
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.cache.redis_cache import (
    build_answer_cache_key,
    build_idempotency_key,
    build_query_embedding_cache_key,
    build_rate_limit_key,
)
from app.generation.prompt import build_llm_prompt
from app.generation.validator import compute_confidence, decide_fallback_reason, detect_llm_insufficiency
from app.models import AskRequest, AskResponse, SourceRef
from app.pipeline.context import ContextItem, assemble_context
from app.pipeline.question_analyzer import analyze_question
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.neighbor import expand_with_neighbors
from app.retrieval.models import FusedChunk


class AsyncPipelineOrchestrator:
    def __init__(
        self,
        *,
        embedding_backend,
        vector_retriever,
        lexical_retriever,
        llm_client,
        corpus_hash: str,
        rule_chunks: list[dict],
        pool=None,
        rrf_k: int = 60,
        max_chunks: int = 7,
        max_context_tokens: int = 3000,
        neighbor_window: int = 1,
        trace_writer: Any | None = None,
        cache: Any | None = None,
        circuit_breaker: Any | None = None,
        reranker: Any | None = None,
        rate_limit_rpm: int = 60,
        embedding_max_concurrency: int = 5,
        prompt_version: str = "p1",
        llm_model: str = "gpt-5.4-mini",
        retrieval_config_hash: str = "default",
        embedding_model: str = "BAAI/bge-m3",
        embedding_config_hash: str = "default",
    ) -> None:
        self.embedding_backend = embedding_backend
        self.vector_retriever = vector_retriever
        self.lexical_retriever = lexical_retriever
        self.llm_client = llm_client
        self.corpus_hash = corpus_hash
        self.rule_chunks = rule_chunks
        self.pool = pool
        self.rrf_k = rrf_k
        self.max_chunks = max_chunks
        self.max_context_tokens = max_context_tokens
        self.neighbor_window = neighbor_window
        self.trace_writer = trace_writer
        self.cache = cache
        self.circuit_breaker = circuit_breaker
        self.reranker = reranker
        self.rate_limit_rpm = rate_limit_rpm
        self.embedding_semaphore = asyncio.Semaphore(embedding_max_concurrency)
        self.prompt_version = prompt_version
        self.llm_model = llm_model
        self.retrieval_config_hash = retrieval_config_hash
        self.embedding_model = embedding_model
        self.embedding_config_hash = embedding_config_hash

    async def ask(self, request: AskRequest) -> AskResponse:
        started = perf_counter()
        trace_id = f"trace-{uuid4()}"
        analysis = analyze_question(request.question)
        pipeline_steps: list[dict] = []
        question_hash = hashlib.sha256(request.question.encode("utf-8")).hexdigest()[:16]
        idempotency_key = build_idempotency_key(request_id=request.request_id) if request.request_id else None

        # Decision 13.4: rate limiting (per-client when client_id provided, global fallback)
        if self.cache and self.rate_limit_rpm > 0:
            rl_subject = request.client_id or "global"
            rl_key = build_rate_limit_key(subject=rl_subject, window="60s")
            count = await self.cache.incr(rl_key, ttl_seconds=60)
            if count > self.rate_limit_rpm:
                return await self._fallback_response(
                    request=request,
                    trace_id=trace_id,
                    started=started,
                    reason="Перевищено ліміт запитів. Спробуйте через хвилину.",
                    pipeline_steps=[{"step": "rate_limit", "status": "rejected"}],
                )

        # Decision 13.3: idempotency guard
        if self.cache and idempotency_key:
            is_new = await self.cache.set_nx(idempotency_key, trace_id, ttl_seconds=30)
            if not is_new:
                # Duplicate in-flight request — wait briefly for answer cache
                await asyncio.sleep(0.5)
                cached = await self.cache.get(self._build_cache_key(request.question))
                if cached:
                    try:
                        return AskResponse.model_validate_json(cached)
                    except Exception:
                        pass

        # Decision 13.1: check answer cache
        cache_key = self._build_cache_key(request.question)
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                try:
                    response = AskResponse.model_validate_json(cached)
                    await self._trace(request, response, [{"step": "cache_hit", "duration_ms": self._latency_ms(started), "status": "success"}])
                    return response
                except Exception:
                    pass  # corrupted cache entry, re-compute

        # Decision 13.2: embedding cache (skip BGE-M3 for repeated questions)
        t_embed = perf_counter()
        embed_cache_key = build_query_embedding_cache_key(
            embedding_model=self.embedding_model,
            embedding_config_hash=self.embedding_config_hash,
            question_hash=question_hash,
        )
        query_embedding = None

        if self.cache:
            cached_emb = await self.cache.get(embed_cache_key)
            if cached_emb:
                try:
                    query_embedding = json.loads(cached_emb)
                except Exception:
                    pass

        if query_embedding is None:
            try:
                if self.circuit_breaker:
                    self.circuit_breaker.raise_if_open()
                async with self.embedding_semaphore:
                    query_embedding = await asyncio.to_thread(self.embedding_backend.embed_query, request.question)
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                # Cache the embedding
                if self.cache and query_embedding is not None:
                    try:
                        await self.cache.set(embed_cache_key, json.dumps(query_embedding))
                    except Exception:
                        pass  # fail-open
            except Exception:
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                query_embedding = None
        pipeline_steps.append({"step": "embedding", "duration_ms": self._latency_ms(t_embed), "status": "success" if query_embedding else "fallback"})

        # Dual retrieval
        t_retrieval = perf_counter()
        vector_task = None
        if query_embedding is not None:
            vector_task = asyncio.create_task(
                self.vector_retriever.search(query_embedding, self.corpus_hash, top_k=20)
            )
        lexical_task = asyncio.create_task(
            self.lexical_retriever.search(request.question, self.corpus_hash, top_k=20)
        )

        vector_results = (await vector_task) if vector_task else []
        lexical_results = await lexical_task
        pipeline_steps.append({"step": "retrieval", "duration_ms": self._latency_ms(t_retrieval), "status": "success", "details": {"vector_results": len(vector_results), "lexical_results": len(lexical_results)}})

        # RRF fusion
        fused = reciprocal_rank_fusion(vector_results, lexical_results, top_k=self.max_chunks * 2, k=self.rrf_k)

        if not fused:
            return await self._fallback_response(
                request=request,
                trace_id=trace_id,
                started=started,
                reason="У базі знань недостатньо релевантного контексту для точної відповіді.",
                pipeline_steps=pipeline_steps,
            )

        # Post-fusion quality filters
        fused = self._apply_quality_filters(fused)

        if not fused:
            return await self._fallback_response(
                request=request,
                trace_id=trace_id,
                started=started,
                reason="У базі знань недостатньо релевантного контексту для точної відповіді.",
                pipeline_steps=pipeline_steps,
            )

        # Second-stage reranking: improve top-context precision after cheap candidate generation.
        if self.reranker is not None:
            t_rerank = perf_counter()
            rrf_order = fused
            try:
                fused = self.reranker.rerank(request.question, fused, top_k=self.max_chunks)
                pipeline_steps.append(
                    {
                        "step": "rerank",
                        "duration_ms": self._latency_ms(t_rerank),
                        "status": "success",
                        "details": {"candidates": len(rrf_order), "returned": len(fused)},
                    }
                )
            except Exception as exc:
                fused = rrf_order[: self.max_chunks]
                pipeline_steps.append(
                    {
                        "step": "rerank",
                        "duration_ms": self._latency_ms(t_rerank),
                        "status": "fallback",
                        "details": {"error": str(exc)},
                    }
                )

        # Decision 12: pre-generation fallback for calculation questions
        if analysis.requires_calculation:
            import re as _re
            _CALC_CONTEXT_PATTERNS = (
                _re.compile(r"\d+[\s,.]?\d*\s*(?:грн|%|відсот)", _re.IGNORECASE),  # currency/percentage
                _re.compile(r"індекс\w*\s*[\d.,]+", _re.IGNORECASE),              # index values
                _re.compile(r"базов\w+\s+місяц", _re.IGNORECASE),                 # base month
                _re.compile(r"формул\w*|коефіцієнт", _re.IGNORECASE),             # formula/coefficient
                _re.compile(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}"),                    # dates with numbers
                _re.compile(r"\d+\s*[×x*·]\s*\d+", _re.IGNORECASE),              # multiplication
            )
            numeric_signal_count = sum(
                1 for chunk in fused
                for pat in _CALC_CONTEXT_PATTERNS
                if pat.search(chunk.content)
            )
            if numeric_signal_count < 2:
                return await self._fallback_response(
                    request=request,
                    trace_id=trace_id,
                    started=started,
                    reason="Для виконання розрахунку потрібні числові дані (базовий місяць, індекс, сума), яких немає у базі знань.",
                    pipeline_steps=pipeline_steps,
                )

        # Context augmentation (Decision 9): neighbor expansion + token budget
        t_augment = perf_counter()
        neighbors = []
        if self.pool and self.neighbor_window > 0:
            chunk_ids = [chunk.chunk_id for chunk in fused]
            neighbor_rows = await expand_with_neighbors(
                self.pool, chunk_ids, self.neighbor_window, self.corpus_hash
            )
            neighbors = [
                FusedChunk(
                    chunk_id=row["chunk_id"],
                    section=row["section"],
                    content=row["content"],
                    rrf_score=0.0,
                    section_order=row["section_order"],
                    chunk_index=row["chunk_index"],
                    content_hash=row["content_hash"],
                )
                for row in neighbor_rows
            ]

        assembled = assemble_context(
            chunks=fused,
            neighbors=neighbors,
            max_chunks=self.max_chunks,
            max_context_tokens=self.max_context_tokens,
        )
        context = assembled.items
        pipeline_steps.append({"step": "context_augmentation", "duration_ms": self._latency_ms(t_augment), "status": "success", "details": {"chunks_before": len(fused), "chunks_after": len(context), "total_tokens": assembled.token_count}})

        # Prompt + LLM generation (Decision 12: post-generation fallback)
        t_llm = perf_counter()
        prompt = build_llm_prompt(context, request.question, rule_chunks=self.rule_chunks)
        try:
            llm_response = await self.llm_client.generate(prompt)
            pipeline_steps.append({"step": "llm_generation", "duration_ms": self._latency_ms(t_llm), "status": "success"})
        except Exception:
            pipeline_steps.append({"step": "llm_generation", "duration_ms": self._latency_ms(t_llm), "status": "error"})
            return await self._fallback_response(
                request=request,
                trace_id=trace_id,
                started=started,
                reason="Сервіс генерації тимчасово недоступний. Спробуйте пізніше.",
                pipeline_steps=pipeline_steps,
            )

        # Confidence scoring
        llm_insufficient = detect_llm_insufficiency(llm_response.answer)
        vector_ids = {r.chunk_id for r in vector_results}
        lexical_ids = {r.chunk_id for r in lexical_results}
        overlap = len(vector_ids & lexical_ids) / max(len(vector_ids | lexical_ids), 1)
        confidence = compute_confidence(
            top_rrf_score=max(chunk.rrf_score for chunk in fused),
            supporting_chunks=len(fused),
            vector_lexical_overlap=overlap,
            has_disclaimer=any(
                "недостатньо" in item.content.casefold() or "does not contain" in item.content.casefold()
                for item in context
            ),
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
        await self._trace(request, response, pipeline_steps)

        # Decision 13.1: cache the response
        if self.cache:
            try:
                await self.cache.set(cache_key, response.model_dump_json())
            except Exception:
                pass  # fail-open: cache write failure is non-fatal
            # Decision 13.3: release idempotency lock
            if idempotency_key:
                try:
                    await self.cache.delete(idempotency_key)
                except Exception:
                    pass

        return response

    def _apply_quality_filters(self, fused: list) -> list:
        if not fused:
            return fused

        # 1. Score-gap cutoff: drop chunks where score drops >30% from top
        top_score = fused[0].rrf_score
        score_threshold = top_score * 0.70
        fused = [chunk for chunk in fused if chunk.rrf_score >= score_threshold]

        # 2. Section coherence: keep only on-topic unless question spans sections
        if fused:
            top_section = fused[0].section
            # Check if top-3 chunks naturally span multiple sections
            top3_sections = {chunk.section for chunk in fused[:3]}
            on_topic = [chunk for chunk in fused if chunk.section == top_section]
            if len(top3_sections) >= 2:
                # Multi-section question: allow 1 off-topic if score ≥95% of top
                off_topic = [
                    chunk for chunk in fused
                    if chunk.section != top_section and chunk.rrf_score >= fused[0].rrf_score * 0.95
                ]
                fused = on_topic + off_topic[:1]
            else:
                # Single-section question: on-topic only
                fused = on_topic

        # 3. Disclaimer filter: remove pure-disclaimer chunks from context
        # (they add noise, not knowledge)
        DISCLAIMER_PREFIXES = (
            "У цій базі знань",
            "AI-консультант не має",
        )
        fused = [
            chunk for chunk in fused
            if not chunk.content.startswith(DISCLAIMER_PREFIXES)
        ]

        # 4. Candidate cap. If reranking is enabled, keep a wider candidate set for
        # the second-stage scorer; otherwise keep the original final context cap.
        cap = self.max_chunks * 2 if self.reranker is not None else self.max_chunks
        return fused[:cap]

    async def _fallback_response(
        self, *, request: AskRequest | None = None, trace_id: str, started: float, reason: str,
        pipeline_steps: list[dict] | None = None,
    ) -> AskResponse:
        response = AskResponse(
            answer=reason,
            sources=[],
            confidence="low",
            fallback_reason=reason,
            trace_id=trace_id,
            latency_ms=self._latency_ms(started),
        )
        await self._trace(request, response, pipeline_steps)
        return response

    def _build_cache_key(self, question: str) -> str:
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        return build_answer_cache_key(
            corpus_hash=self.corpus_hash,
            prompt_version=self.prompt_version,
            llm_model=self.llm_model,
            retrieval_config_hash=self.retrieval_config_hash,
            question_hash=question_hash,
        )

    def _latency_ms(self, started: float) -> int:
        return max(0, round((perf_counter() - started) * 1000))

    async def _trace(self, request: AskRequest | None, response: AskResponse, pipeline_steps: list[dict] | None = None) -> None:
        if self.trace_writer is None:
            return
        await self.trace_writer.enqueue(
            {
                "trace_id": response.trace_id,
                "question": request.question if request else None,
                "pipeline_steps": pipeline_steps or [],
                "response": response.model_dump(),
                "total_latency_ms": response.latency_ms,
                "error": None,
            }
        )
