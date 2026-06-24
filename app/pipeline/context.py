from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.chunker import estimate_tokens
from app.retrieval.models import FusedChunk


@dataclass(frozen=True)
class ContextItem:
    chunk_id: str
    section: str
    content: str
    score: float
    section_order: int = 0
    chunk_index: int = 0
    content_hash: str | None = None


@dataclass(frozen=True)
class AssembledContext:
    items: list[ContextItem]
    token_count: int


def assemble_context(
    chunks: list[FusedChunk],
    neighbors: list[FusedChunk],
    max_chunks: int,
    max_context_tokens: int,
) -> AssembledContext:
    deduped: dict[str, FusedChunk] = {}
    for chunk in [*chunks, *neighbors]:
        current = deduped.get(chunk.chunk_id)
        if current is None or chunk.rrf_score > current.rrf_score:
            deduped[chunk.chunk_id] = chunk

    ordered = list(deduped.values())

    items: list[ContextItem] = []
    token_count = 0
    for chunk in ordered:
        if len(items) >= max_chunks:
            break
        chunk_tokens = estimate_tokens(chunk.content)
        if token_count + chunk_tokens > max_context_tokens:
            continue
        items.append(
            ContextItem(
                chunk_id=chunk.chunk_id,
                section=chunk.section,
                content=chunk.content,
                score=chunk.rrf_score,
                section_order=chunk.section_order,
                chunk_index=chunk.chunk_index,
                content_hash=chunk.content_hash,
            )
        )
        token_count += chunk_tokens

    return AssembledContext(items=items, token_count=token_count)
