import pytest

from app.ingestion.embedder import EmbeddingUnavailable, FakeEmbeddingBackend


def test_fake_embedding_backend_returns_1024_dimensions():
    backend = FakeEmbeddingBackend()
    vector = backend.embed_query("відпустка")
    assert len(vector) == 1024


def test_backend_can_signal_unavailable_for_lexical_only_fallback():
    backend = FakeEmbeddingBackend(available=False)
    with pytest.raises(EmbeddingUnavailable):
        backend.embed_query("відпустка")
