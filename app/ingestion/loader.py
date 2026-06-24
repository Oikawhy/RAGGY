from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from app.ingestion.chunker import Chunk
from app.ingestion.parser import Section


@dataclass(frozen=True)
class SourceManifestItem:
    logical_source_key: str
    version: int
    hash_sha256: str


def compute_corpus_hash(items: list[SourceManifestItem]) -> str:
    manifest = sorted(
        (
            {
                "logical_source_key": item.logical_source_key,
                "version": item.version,
                "hash_sha256": item.hash_sha256,
            }
            for item in items
        ),
        key=lambda row: (row["logical_source_key"], row["version"], row["hash_sha256"]),
    )
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float] | None


def attach_embeddings_to_chunks(chunks: list[Chunk], backend) -> list[EmbeddedChunk]:
    if backend is None:
        return [EmbeddedChunk(chunk=chunk, embedding=None) for chunk in chunks]
    vectors = backend.embed_documents([chunk.content for chunk in chunks])
    return [EmbeddedChunk(chunk=chunk, embedding=vector) for chunk, vector in zip(chunks, vectors, strict=True)]


def format_vector(vector: list[float] | None) -> str | None:
    if vector is None:
        return None
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


@dataclass(frozen=True)
class IngestionRecord:
    source: SourceManifestItem
    filename: str
    sections: list[Section]
    chunks: list[Chunk]
    embedded_chunks: list[EmbeddedChunk] | None = None
    corpus_hash: str = ""


async def persist_ingestion_records(connection, record: IngestionRecord) -> None:
    async with connection.transaction():
        source_id = await connection.fetchval(
            """
            INSERT INTO knowledge_sources (
                filename, logical_source_key, version, hash_sha256,
                is_active, total_sections, total_chunks
            )
            VALUES ($1, $2, $3, $4, TRUE, $5, $6)
            ON CONFLICT (logical_source_key, version)
            DO UPDATE SET
                filename = EXCLUDED.filename,
                hash_sha256 = EXCLUDED.hash_sha256,
                is_active = TRUE,
                total_sections = EXCLUDED.total_sections,
                total_chunks = EXCLUDED.total_chunks,
                indexed_at = NOW()
            RETURNING id
            """,
            record.filename,
            record.source.logical_source_key,
            record.source.version,
            record.source.hash_sha256,
            len(record.sections),
            len(record.chunks),
        )

        section_ids: dict[int, str] = {}
        for section in record.sections:
            section_id = await connection.fetchval(
                """
                INSERT INTO knowledge_sections (source_id, section_num, title, language, is_meta, raw_content)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (source_id, section_num)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    language = EXCLUDED.language,
                    is_meta = EXCLUDED.is_meta,
                    raw_content = EXCLUDED.raw_content
                RETURNING id
                """,
                source_id,
                section.section_num,
                section.title,
                section.language,
                section.is_meta,
                section.raw_content,
            )
            section_ids[section.section_num] = section_id

        # Build chunk iteration: use embedded_chunks if available, otherwise plain chunks
        chunk_items: list[tuple[Chunk, list[float] | None]] = []
        if record.embedded_chunks:
            chunk_items = [(ec.chunk, ec.embedding) for ec in record.embedded_chunks]
        else:
            chunk_items = [(chunk, None) for chunk in record.chunks]

        for chunk, embedding in chunk_items:
            await connection.execute(
                """
                INSERT INTO knowledge_chunks (
                    section_id, source_id, chunk_index, content, content_hash,
                    content_tokens, language, block_type, has_numbers,
                    has_disclaimer, is_rule, section_title, embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (section_id, chunk_index)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    content_tokens = EXCLUDED.content_tokens,
                    language = EXCLUDED.language,
                    block_type = EXCLUDED.block_type,
                    has_numbers = EXCLUDED.has_numbers,
                    has_disclaimer = EXCLUDED.has_disclaimer,
                    is_rule = EXCLUDED.is_rule,
                    section_title = EXCLUDED.section_title,
                    embedding = EXCLUDED.embedding
                """,
                section_ids[chunk.section_num],
                source_id,
                chunk.chunk_index,
                chunk.content,
                chunk.content_hash,
                chunk.content_tokens,
                chunk.language,
                chunk.block_type,
                chunk.has_numbers,
                chunk.has_disclaimer,
                chunk.is_rule,
                chunk.section_title,
                format_vector(embedding),
            )

        await connection.execute(
            """
            INSERT INTO active_corpus_sources (corpus_hash, source_id)
            VALUES ($1, $2)
            ON CONFLICT (corpus_hash, source_id) DO NOTHING
            """,
            record.corpus_hash,
            source_id,
        )

