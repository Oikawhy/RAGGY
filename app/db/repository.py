from __future__ import annotations

from typing import Any

from app.ingestion.loader import SourceManifestItem


class KnowledgeRepository:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

    async def load_active_manifest_items(self) -> list[SourceManifestItem]:
        rows = await self.connection.fetch(
            """
            SELECT logical_source_key, version, hash_sha256
            FROM knowledge_sources
            WHERE is_active = TRUE
            ORDER BY logical_source_key, version
            """
        )
        return [
            SourceManifestItem(row["logical_source_key"], int(row["version"]), row["hash_sha256"])
            for row in rows
        ]

    async def load_active_retrievable_chunks(self, corpus_hash: str) -> list[dict]:
        rows = await self.connection.fetch(
            """
            SELECT
                c.id::text AS chunk_id,
                c.section_title AS section,
                c.content,
                0.0 AS score,
                row_number() OVER (ORDER BY s.section_num, c.chunk_index)::int AS rank,
                s.section_num AS section_order,
                c.chunk_index,
                c.content_hash,
                c.has_disclaimer
            FROM knowledge_chunks c
            JOIN knowledge_sections s ON s.id = c.section_id
            JOIN active_corpus_sources acs ON acs.source_id = c.source_id
            WHERE acs.corpus_hash = $1
              AND c.is_rule = FALSE
            ORDER BY s.section_num, c.chunk_index
            """,
            corpus_hash,
        )
        return [dict(row) for row in rows]

    async def load_active_rule_chunks(self, corpus_hash: str) -> list[dict]:
        rows = await self.connection.fetch(
            """
            SELECT c.id::text AS chunk_id, c.section_title AS section, c.content, s.section_num AS section_order
            FROM knowledge_chunks c
            JOIN knowledge_sections s ON s.id = c.section_id
            JOIN active_corpus_sources acs ON acs.source_id = c.source_id
            WHERE acs.corpus_hash = $1
              AND c.is_rule = TRUE
            ORDER BY s.section_num, c.chunk_index
            """,
            corpus_hash,
        )
        return [dict(row) for row in rows]
