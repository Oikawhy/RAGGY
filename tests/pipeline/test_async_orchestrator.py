import pytest

from app.pipeline.async_orchestrator import AsyncPipelineOrchestrator
from app.generation.llm import LLMResponse
from app.models import AskRequest
from app.retrieval.models import RetrievedChunk


class FakeEmbedder:
    def embed_query(self, text):
        return [0.1] * 1024


class FakeVectorRetriever:
    async def search(self, query_embedding, corpus_hash, top_k):
        return [
            RetrievedChunk(chunk_id="v1", section="1. Щорічна відпустка",
                           content="Працівник може використати відпустку після 6 місяців",
                           score=0.8, rank=1, section_order=1, chunk_index=0, content_hash="h1")
        ]


class FakeLexicalRetriever:
    async def search(self, query, corpus_hash, top_k):
        return [
            RetrievedChunk(chunk_id="l1", section="1. Щорічна відпустка",
                           content="Відпустка надається після 6 місяців",
                           score=0.5, rank=1, section_order=1, chunk_index=1, content_hash="h2")
        ]


class ReversingReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, question, chunks, top_k):
        self.calls.append((question, [chunk.chunk_id for chunk in chunks], top_k))
        return list(reversed(chunks))[:top_k]


class FailingReranker:
    def rerank(self, question, chunks, top_k):
        raise RuntimeError("reranker unavailable")


class FakeAsyncLLM:
    async def generate(self, prompt):
        return LLMResponse(answer="Так, після 6 місяців", fallback_reason=None, used_sections=["1. Щорічна відпустка"])


class RecordingSemaphore:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    async def __aenter__(self):
        self.entered += 1

    async def __aexit__(self, exc_type, exc, tb):
        self.exited += 1
        return False


class FakeCache:
    def __init__(self):
        self.get_keys = []
        self.set_keys = []
        self.set_nx_keys = []
        self.incr_keys = []
        self.delete_keys = []

    async def get(self, key):
        self.get_keys.append(key)
        return None

    async def set(self, key, value):
        self.set_keys.append(key)

    async def set_nx(self, key, value, ttl_seconds=None):
        self.set_nx_keys.append(key)
        return True

    async def incr(self, key, ttl_seconds=None):
        self.incr_keys.append(key)
        return 1

    async def delete(self, key):
        self.delete_keys.append(key)


@pytest.mark.anyio
async def test_async_orchestrator_processes_full_pipeline():
    orchestrator = AsyncPipelineOrchestrator(
        embedding_backend=FakeEmbedder(),
        vector_retriever=FakeVectorRetriever(),
        lexical_retriever=FakeLexicalRetriever(),
        llm_client=FakeAsyncLLM(),
        corpus_hash="corp1",
        rule_chunks=[],
        rrf_k=60,
        max_chunks=8,
    )
    response = await orchestrator.ask(AskRequest(question="Коли працівник може отримати відпустку?"))
    assert "місяців" in response.answer
    assert response.confidence in {"high", "medium", "low"}
    assert len(response.sources) >= 1


@pytest.mark.anyio
async def test_async_orchestrator_uses_architecture_cache_keys():
    cache = FakeCache()
    orchestrator = AsyncPipelineOrchestrator(
        embedding_backend=FakeEmbedder(),
        vector_retriever=FakeVectorRetriever(),
        lexical_retriever=FakeLexicalRetriever(),
        llm_client=FakeAsyncLLM(),
        corpus_hash="corp1",
        rule_chunks=[],
        cache=cache,
        prompt_version="p2",
        llm_model="gpt-5.4-mini",
        retrieval_config_hash="retr-v1",
        embedding_model="BAAI/bge-m3",
        embedding_config_hash="embed-v1",
    )
    await orchestrator.ask(
        AskRequest(
            question="Коли працівник може отримати відпустку?",
            request_id="req-1",
            client_id="client-1",
        )
    )

    assert cache.incr_keys == ["rate_limit:client-1:60s"]
    assert cache.set_nx_keys == ["idempotency:req-1"]
    assert any(key.startswith("query_embedding:BAAI/bge-m3:embed-v1:") for key in cache.get_keys + cache.set_keys)
    assert any(key.startswith("answer:corp1:p2:gpt-5.4-mini:retr-v1:") for key in cache.get_keys + cache.set_keys)
    assert cache.delete_keys == ["idempotency:req-1"]


@pytest.mark.anyio
async def test_async_orchestrator_uses_embedding_semaphore():
    semaphore = RecordingSemaphore()
    orchestrator = AsyncPipelineOrchestrator(
        embedding_backend=FakeEmbedder(),
        vector_retriever=FakeVectorRetriever(),
        lexical_retriever=FakeLexicalRetriever(),
        llm_client=FakeAsyncLLM(),
        corpus_hash="corp1",
        rule_chunks=[],
        embedding_max_concurrency=1,
    )
    orchestrator.embedding_semaphore = semaphore

    await orchestrator.ask(AskRequest(question="Коли працівник може отримати відпустку?"))

    assert semaphore.entered == 1
    assert semaphore.exited == 1


@pytest.mark.anyio
async def test_async_orchestrator_reranks_after_rrf_before_context():
    reranker = ReversingReranker()
    orchestrator = AsyncPipelineOrchestrator(
        embedding_backend=FakeEmbedder(),
        vector_retriever=FakeVectorRetriever(),
        lexical_retriever=FakeLexicalRetriever(),
        llm_client=FakeAsyncLLM(),
        corpus_hash="corp1",
        rule_chunks=[],
        reranker=reranker,
        max_chunks=8,
    )

    response = await orchestrator.ask(AskRequest(question="Коли працівник може отримати відпустку?"))

    assert reranker.calls == [
        ("Коли працівник може отримати відпустку?", ["v1", "l1"], 8)
    ]
    assert [source.chunk_id for source in response.sources[:2]] == ["l1", "v1"]


@pytest.mark.anyio
async def test_async_orchestrator_falls_back_to_rrf_order_when_reranker_fails():
    orchestrator = AsyncPipelineOrchestrator(
        embedding_backend=FakeEmbedder(),
        vector_retriever=FakeVectorRetriever(),
        lexical_retriever=FakeLexicalRetriever(),
        llm_client=FakeAsyncLLM(),
        corpus_hash="corp1",
        rule_chunks=[],
        reranker=FailingReranker(),
        max_chunks=8,
    )

    response = await orchestrator.ask(AskRequest(question="Коли працівник може отримати відпустку?"))

    assert [source.chunk_id for source in response.sources[:2]] == ["v1", "l1"]
