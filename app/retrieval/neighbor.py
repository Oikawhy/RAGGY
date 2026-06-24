from __future__ import annotations


async def expand_with_neighbors(
    pool,
    chunk_ids: list[str],
    window: int,
    corpus_hash: str,
) -> list[dict]:
    if not chunk_ids or window < 1:
        return []
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            WITH targets AS (
                SELECT c.section_id, c.chunk_index
                FROM knowledge_chunks c
                WHERE c.id = ANY($1::uuid[])
            )
            SELECT
                n.id::text AS chunk_id,
                n.section_title AS section,
                n.content,
                n.chunk_index,
                s.section_num AS section_order,
                n.content_hash
            FROM targets t
            JOIN knowledge_chunks n ON n.section_id = t.section_id
                AND n.chunk_index BETWEEN t.chunk_index - $2 AND t.chunk_index + $2
                AND n.id != ALL($1::uuid[])
            JOIN active_corpus_sources acs ON acs.source_id = n.source_id AND acs.corpus_hash = $3
            JOIN knowledge_sections s ON s.id = n.section_id
            WHERE n.is_rule = FALSE
            ORDER BY s.section_num, n.chunk_index
            """,
            chunk_ids,
            window,
            corpus_hash,
        )
        return [dict(row) for row in rows]
