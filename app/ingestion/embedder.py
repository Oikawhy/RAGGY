from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from time import monotonic
from typing import Protocol

import requests


class EmbeddingUnavailable(RuntimeError):
    """Raised when semantic embedding is unavailable and retrieval should fall back."""


class EmbeddingBackend(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class FakeEmbeddingBackend:
    def __init__(self, available: bool = True, dimensions: int = 1024) -> None:
        self.available = available
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        if not self.available:
            raise EmbeddingUnavailable("embedding backend is unavailable")
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.dimensions:
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) == self.dimensions:
                    break
            digest = hashlib.sha256(digest).digest()
        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class HttpEmbeddingBackend:
    def __init__(self, service_url: str, timeout_seconds: int, http_client=None) -> None:
        self.service_url = service_url
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client or requests

    def embed_query(self, text: str) -> list[float]:
        response = self.http_client.post(self.service_url, json={"text": text}, timeout=self.timeout_seconds)
        response.raise_for_status()
        vector = response.json()["embedding"]
        if len(vector) != 1024:
            raise EmbeddingUnavailable("embedding service returned non-BGE-M3 vector dimensions")
        return [float(value) for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class LocalBGEEmbeddingBackend:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector.tolist()]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector.tolist()] for vector in vectors]

# NOTE: Dockerfile must pre-download BGE-M3 to avoid 2.2GB download at runtime:
# RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"


@dataclass
class EmbeddingCircuitBreaker:
    threshold: int
    cooldown_seconds: int
    failures: int = field(default=0, init=False)
    opened_at: float | None = field(default=None, init=False)

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = monotonic()

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if monotonic() - self.opened_at >= self.cooldown_seconds:
            self.failures = 0
            self.opened_at = None
            return False
        return True

    def raise_if_open(self) -> None:
        if self.is_open():
            raise EmbeddingUnavailable("embedding circuit breaker is open")


def build_embedding_backend(backend_name: str, service_url: str | None, timeout_seconds: int) -> EmbeddingBackend:
    if backend_name == "fake":
        return FakeEmbeddingBackend()
    if backend_name == "http":
        if not service_url:
            raise ValueError("EMBEDDING_SERVICE_URL is required for http embedding backend")
        return HttpEmbeddingBackend(service_url, timeout_seconds)
    if backend_name == "local_bge_m3":
        return LocalBGEEmbeddingBackend()
    raise ValueError(f"unsupported embedding backend: {backend_name}")
