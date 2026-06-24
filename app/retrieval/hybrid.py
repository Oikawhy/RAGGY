from __future__ import annotations

from app.retrieval.models import FusedChunk, RetrievedChunk


def reciprocal_rank_fusion(
    vector_results: list[RetrievedChunk],
    lexical_results: list[RetrievedChunk],
    top_k: int,
    k: int = 60,
) -> list[FusedChunk]:
    merged: dict[str, dict[str, object]] = {}

    def add(result: RetrievedChunk, source: str) -> None:
        item = merged.setdefault(
            result.chunk_id,
            {
                "chunk": result,
                "score": 0.0,
                "vector_rank": None,
                "lexical_rank": None,
            },
        )
        item["score"] = float(item["score"]) + 1.0 / (k + result.rank)
        item[f"{source}_rank"] = result.rank

    for result in vector_results:
        add(result, "vector")
    for result in lexical_results:
        add(result, "lexical")

    fused = [
        FusedChunk(
            chunk_id=chunk.chunk_id,
            section=chunk.section,
            content=chunk.content,
            rrf_score=float(item["score"]),
            section_order=chunk.section_order,
            chunk_index=chunk.chunk_index,
            content_hash=chunk.content_hash,
            vector_rank=item["vector_rank"],
            lexical_rank=item["lexical_rank"],
            metadata=chunk.metadata,
        )
        for item in merged.values()
        for chunk in [item["chunk"]]
    ]
    return sorted(fused, key=lambda chunk: (-chunk.rrf_score, chunk.section_order, chunk.chunk_index))[:top_k]
