from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any
from uuid import uuid4

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.cache.redis_cache import AsyncRedisCache
from app.config import Settings
from app.db.pool import create_pool
from app.db.repository import KnowledgeRepository
from app.generation.llm import OpenAILLMClient
from app.ingestion.embedder import EmbeddingCircuitBreaker, build_embedding_backend
from app.models import AskRequest, AskResponse
from app.observability.tracer import AsyncTraceWriter
from app.pipeline.async_orchestrator import AsyncPipelineOrchestrator
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.runtime import build_runtime_state
from app.retrieval.lexical import PostgresLexicalRetriever
from app.retrieval.reranker import build_reranker
from app.retrieval.vector import PostgresVectorRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()

    # Database pool
    pool = await create_pool(settings.database_url, min_size=2, max_size=10)
    app.state.pool = pool

    # Redis
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    cache = AsyncRedisCache(
        redis_client,
        ttl_seconds=settings.cache_ttl_seconds,
        fail_open=settings.redis_control_plane_failure_mode == "fail_open",
    )
    app.state.cache = cache

    # Embedding backend
    embedding_backend = build_embedding_backend(
        settings.embedding_backend, settings.embedding_service_url, settings.embedding_timeout_seconds
    )
    circuit_breaker = EmbeddingCircuitBreaker(
        threshold=settings.embedding_circuit_breaker_threshold,
        cooldown_seconds=settings.embedding_circuit_breaker_cooldown,
    )

    # Build runtime state from DB
    async with pool.acquire() as connection:
        repo = KnowledgeRepository(connection)
        runtime = await build_runtime_state(repo)

    # Retrievers (pool-per-request pattern)
    vector_retriever = PostgresVectorRetriever(pool)
    lexical_retriever = PostgresLexicalRetriever(pool, runtime.bm25_index)
    reranker = build_reranker(
        settings.reranker_enabled,
        settings.reranker_backend,
        settings.reranker_model,
    )

    # LLM client
    llm_client = OpenAILLMClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_concurrency=settings.llm_max_concurrency,
    )

    # Trace writer
    trace_writer = AsyncTraceWriter(
        trace_dir=settings.trace_dir,
        max_queue_size=settings.trace_queue_size,
        enqueue_timeout_ms=settings.trace_enqueue_timeout_ms,
        retention_days=settings.trace_retention_days,
    )
    await trace_writer.start()
    trace_writer.cleanup_old_traces()

    # Orchestrator
    orchestrator = AsyncPipelineOrchestrator(
        embedding_backend=embedding_backend,
        vector_retriever=vector_retriever,
        lexical_retriever=lexical_retriever,
        llm_client=llm_client,
        corpus_hash=runtime.corpus_hash,
        rule_chunks=runtime.rule_chunks,
        pool=pool,
        rrf_k=settings.rrf_k,
        max_chunks=settings.max_chunks,
        max_context_tokens=settings.max_context_tokens,
        neighbor_window=settings.neighbor_window,
        trace_writer=trace_writer,
        cache=cache,
        circuit_breaker=circuit_breaker,
        reranker=reranker,
        embedding_max_concurrency=settings.embedding_max_concurrency,
        prompt_version=settings.prompt_version,
        llm_model=settings.llm_model,
        retrieval_config_hash=settings.retrieval_config_hash,
        embedding_model=settings.embedding_model,
        embedding_config_hash=settings.embedding_config_hash,
    )
    app.state.orchestrator = orchestrator
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.trace_writer = trace_writer

    yield

    # Shutdown
    await trace_writer.stop()
    await redis_client.aclose()
    await pool.close()


def create_app(pipeline_deps: dict[str, Any] | None = None) -> FastAPI:
    """Create the FastAPI app. When pipeline_deps is None (production), use lifespan."""
    if pipeline_deps is None:
        # Unit test mode: no lifespan, sync orchestrator
        app = FastAPI(title="AI Consultant RAG API", version="0.1.0")
        orchestrator = PipelineOrchestrator()
    else:
        app = FastAPI(title="AI Consultant RAG API", version="0.1.0")
        orchestrator = PipelineOrchestrator(**pipeline_deps)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "trace_id": f"trace-{uuid4()}",
                "details": exc.errors(),
            },
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        return orchestrator.ask(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "degraded",
            "checks": {
                "postgres": "not_configured_in_unit_runtime",
                "redis": "not_configured_in_unit_runtime",
                "llm": "dependency_injected",
            },
        }

    return app


def create_production_app() -> FastAPI:
    """Create the production FastAPI app with full lifespan."""
    app = FastAPI(title="AI Consultant RAG API", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "trace_id": f"trace-{uuid4()}",
                "details": exc.errors(),
            },
        )

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest) -> AskResponse:
        return await app.state.orchestrator.ask(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        checks: dict[str, Any] = {}
        pg_ok = False

        # PostgreSQL + pgvector check
        try:
            db_started = perf_counter()
            async with app.state.pool.acquire() as conn:
                from app.db.pool import check_database_health
                db_health = await check_database_health(conn)
                latency_ms = max(0, round((perf_counter() - db_started) * 1000))
                checks["postgres"] = {"status": db_health.status, "latency_ms": latency_ms}
                checks["pgvector"] = {"status": "up" if db_health.pgvector else "down", "chunks_indexed": db_health.chunks_indexed}
                pg_ok = db_health.status == "up" and db_health.chunks_indexed > 0
        except Exception as exc:
            checks["postgres"] = {"status": "down", "error": str(exc)}
            checks["pgvector"] = {"status": "unknown"}

        # Redis check
        redis_ok = False
        try:
            redis_ok = await app.state.cache.ping()
        except Exception:
            pass
        checks["redis"] = {"status": "up" if redis_ok else "down"}

        # Embedding model check (reflects circuit breaker state)
        embedding_backend = getattr(app.state, "settings", None) and app.state.settings.embedding_backend
        orchestrator = getattr(app.state, "orchestrator", None)
        cb = getattr(orchestrator, "circuit_breaker", None) if orchestrator else None
        embedding_up = True
        if cb:
            try:
                cb.raise_if_open()
            except Exception:
                embedding_up = False
        checks["embedding_model"] = {
            "status": "up" if embedding_up else "degraded",
            "provider": embedding_backend or "unknown",
        }

        # Knowledge base check
        runtime = getattr(app.state, "runtime", None)
        if runtime:
            active_chunks = len(runtime.retrievable_chunks) + len(runtime.rule_chunks)
            checks["knowledge_base"] = {
                "status": "indexed" if active_chunks > 0 else "empty",
                "version": runtime.knowledge_base_version,
                "corpus_hash": runtime.corpus_hash,
                "active_chunks": active_chunks,
            }
        else:
            checks["knowledge_base"] = {"status": "not_loaded"}

        # Overall status: healthy / degraded / unhealthy
        if not pg_ok or checks.get("knowledge_base", {}).get("status") != "indexed":
            overall = "unhealthy"
        elif not redis_ok or not embedding_up:
            overall = "degraded"
        else:
            overall = "healthy"

        return {"status": overall, "checks": checks}

    return app
