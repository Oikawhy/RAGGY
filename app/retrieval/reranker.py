from __future__ import annotations

from dataclasses import replace
import math
from typing import Protocol

from app.retrieval.lexical import expand_query_tokens, tokenize
from app.retrieval.models import FusedChunk


class Reranker(Protocol):
    def rerank(self, question: str, chunks: list[FusedChunk], top_k: int) -> list[FusedChunk]:
        ...


class NoOpReranker:
    def rerank(self, question: str, chunks: list[FusedChunk], top_k: int) -> list[FusedChunk]:
        return chunks[:top_k]


class LexicalOverlapReranker:
    """Lightweight deterministic reranker for local/dev and fail-safe production use."""

    def rerank(self, question: str, chunks: list[FusedChunk], top_k: int) -> list[FusedChunk]:
        query_terms = set(expand_query_tokens(question))
        if not query_terms:
            return chunks[:top_k]

        max_rrf = max((chunk.rrf_score for chunk in chunks), default=0.0)
        scored: list[tuple[float, FusedChunk]] = []
        for rank, chunk in enumerate(chunks, start=1):
            chunk_terms = set(tokenize(chunk.content))
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            rrf_component = chunk.rrf_score / max(max_rrf, 1e-9)
            score = 0.75 * overlap + 0.25 * rrf_component
            scored.append(
                (
                    score,
                    replace(
                        chunk,
                        metadata={
                            **chunk.metadata,
                            "pre_rerank_rank": rank,
                            "rerank_score": score,
                            "reranker": "lexical_overlap",
                        },
                    ),
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].section_order,
                item[1].chunk_index,
                item[1].chunk_id,
            )
        )
        return [chunk for _, chunk in scored[:top_k]]


class LocalCrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, question: str, chunks: list[FusedChunk], top_k: int) -> list[FusedChunk]:
        if not chunks:
            return []

        scores = self.model.predict([(question, chunk.content) for chunk in chunks])
        scored: list[tuple[float, FusedChunk]] = []
        for rank, (chunk, raw_score) in enumerate(zip(chunks, scores, strict=True), start=1):
            score = float(raw_score)
            if math.isnan(score):
                score = float("-inf")
            scored.append(
                (
                    score,
                    replace(
                        chunk,
                        metadata={
                            **chunk.metadata,
                            "pre_rerank_rank": rank,
                            "rerank_score": score,
                            "reranker": self.model_name,
                        },
                    ),
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1].section_order, item[1].chunk_index))
        return [chunk for _, chunk in scored[:top_k]]


def build_reranker(enabled: bool, backend: str, model_name: str) -> Reranker | None:
    if not enabled or backend == "none":
        return None
    if backend == "lexical_overlap":
        return LexicalOverlapReranker()
    if backend == "local_cross_encoder":
        return LocalCrossEncoderReranker(model_name)
    raise ValueError(f"unsupported reranker backend: {backend}")
