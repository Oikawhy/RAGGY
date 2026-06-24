from __future__ import annotations

from typing import Protocol

from app.retrieval.models import RetrievedChunk

VECTOR_OVERSAMPLE_FACTOR = 200  # 10x of typical top_k=20, tunable


def build_vector_search_sql() -> str:
    return """
    SELECT
        c.id::text AS chunk_id,
        c.section_title AS section,
        c.content,
        1 - (c.embedding <=> :query_embedding) AS score,
        c.chunk_index,
        s.section_num AS section_order,
        c.content_hash
    FROM knowledge_chunks c
    JOIN active_corpus_sources acs ON acs.source_id = c.source_id
    JOIN knowledge_sections s ON s.id = c.section_id
    WHERE acs.corpus_hash = :corpus_hash
      AND c.embedding IS NOT NULL
      AND c.is_rule = FALSE
    ORDER BY c.embedding <=> :query_embedding
    LIMIT :top_k
    """.strip()


def format_pgvector(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


class VectorRetriever(Protocol):
    def search(self, query_embedding: list[float], corpus_hash: str, top_k: int) -> list[RetrievedChunk]:
        ...


class PostgresVectorRetriever:
    def __init__(self, pool) -> None:
        self.pool = pool

    async def search(self, query_embedding: list[float], corpus_hash: str, top_k: int) -> list[RetrievedChunk]:
        async with self.pool.acquire() as connection:
            await connection.execute("SET LOCAL hnsw.ef_search = 100")
            oversample = max(top_k, VECTOR_OVERSAMPLE_FACTOR)
            # Oversampling CTE per ARCHITECTURE.md: HNSW scans globally,
            # then JOIN filters to active corpus
            rows = await connection.fetch(
                """
                WITH candidates AS (
                    SELECT c.id::text AS chunk_id, c.content, c.section_title AS section,
                           c.has_disclaimer, c.source_id, c.chunk_index, c.section_id,
                           c.content_hash,
                           1 - (c.embedding <=> $1::vector) AS score
                    FROM knowledge_chunks c
                    WHERE c.is_rule = FALSE AND c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> $1::vector
                    LIMIT $3
                )
                SELECT c.chunk_id, c.content, c.section, c.score,
                       c.chunk_index, s.section_num AS section_order, c.content_hash
                FROM candidates c
                JOIN active_corpus_sources acs ON acs.source_id = c.source_id AND acs.corpus_hash = $2
                JOIN knowledge_sections s ON s.id = c.section_id
                ORDER BY c.score DESC
                LIMIT $4
                """,
                format_pgvector(query_embedding),
                corpus_hash,
                oversample,
                top_k,
            )
            return [
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    section=row["section"],
                    content=row["content"],
                    score=float(row["score"]),
                    rank=index,
                    section_order=int(row["section_order"]),
                    chunk_index=int(row["chunk_index"]),
                    content_hash=row["content_hash"],
                )
                for index, row in enumerate(rows, start=1)
            ]

