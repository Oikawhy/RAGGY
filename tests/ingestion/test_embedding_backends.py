import pytest

from app.ingestion.embedder import (
    EmbeddingCircuitBreaker,
    EmbeddingUnavailable,
    HttpEmbeddingBackend,
    build_embedding_backend,
)


class FakeHttpClient:
    def __init__(self):
        self.payload = None

    def post(self, url, json, timeout):
        self.payload = {"url": url, "json": json, "timeout": timeout}
        return FakeResponse()


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"embedding": [0.1] * 1024}


def test_http_embedding_backend_returns_service_vector():
    client = FakeHttpClient()
    backend = HttpEmbeddingBackend("http://embed.local/embed", timeout_seconds=3, http_client=client)
    vector = backend.embed_query("відпустка")
    assert len(vector) == 1024
    assert client.payload["json"] == {"text": "відпустка"}


def test_build_embedding_backend_fake_for_unit_tests():
    backend = build_embedding_backend(backend_name="fake", service_url=None, timeout_seconds=3)
    assert len(backend.embed_query("test")) == 1024


def test_circuit_breaker_opens_after_failures():
    breaker = EmbeddingCircuitBreaker(threshold=2, cooldown_seconds=60)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open() is True
    with pytest.raises(EmbeddingUnavailable):
        breaker.raise_if_open()
