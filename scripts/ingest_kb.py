from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from app.ingestion.chunker import Chunk, chunk_sections
from app.ingestion.loader import (
    IngestionRecord,
    SourceManifestItem,
    attach_embeddings_to_chunks,
    compute_corpus_hash,
    persist_ingestion_records,
)
from app.ingestion.parser import Section, parse_markdown_sections


@dataclass(frozen=True)
class IngestionPlan:
    source: SourceManifestItem
    source_path: str
    corpus_hash: str
    sections: list[Section]
    chunks: list[Chunk]

    @property
    def total_sections(self) -> int:
        return len(self.sections)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)


def build_ingestion_plan(source: str, logical_source_key: str, version: int) -> IngestionPlan:
    path = Path(source)
    content = path.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest_item = SourceManifestItem(
        logical_source_key=logical_source_key,
        version=version,
        hash_sha256=source_hash,
    )
    sections = parse_markdown_sections(content)
    chunks = chunk_sections(sections)
    return IngestionPlan(
        source=manifest_item,
        source_path=source,
        corpus_hash=compute_corpus_hash([manifest_item]),
        sections=sections,
        chunks=chunks,
    )


async def apply_ingestion_plan(pool, plan: IngestionPlan, embedding_backend=None) -> None:
    # Embed chunks if backend provided
    embedded_chunks = None
    if embedding_backend is not None:
        print(f"Embedding {len(plan.chunks)} chunks...")
        embedded_chunks = attach_embeddings_to_chunks(plan.chunks, embedding_backend)
        print(f"Embedded {len(embedded_chunks)} chunks successfully.")

    async with pool.acquire() as connection:
        record = IngestionRecord(
            source=plan.source,
            filename=Path(plan.source_path).name,
            sections=plan.sections,
            chunks=plan.chunks,
            embedded_chunks=embedded_chunks,
            corpus_hash=plan.corpus_hash,
        )
        await persist_ingestion_records(connection, record)


async def async_main() -> None:
    from app.config import Settings
    from app.db.pool import create_pool
    from app.ingestion.embedder import build_embedding_backend

    parser = argparse.ArgumentParser(description="Build an ingestion plan for a knowledge base Markdown file.")
    parser.add_argument("--source", default="data/knowledge_base.md")
    parser.add_argument("--logical-source-key", default="assignment-kb")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--apply", action="store_true", help="Write ingestion results to PostgreSQL.")
    parser.add_argument("--embed", action="store_true", help="Generate BGE-M3 embeddings during ingestion.")
    args = parser.parse_args()

    plan = build_ingestion_plan(args.source, args.logical_source_key, args.version)

    if args.apply:
        settings = Settings()
        pool = await create_pool(settings.database_url)

        embedding_backend = None
        if args.embed:
            backend_name = settings.embedding_backend
            if backend_name == "fake":
                backend_name = "local_bge_m3"
                print("--embed flag: overriding EMBEDDING_BACKEND=fake → local_bge_m3")
            embedding_backend = build_embedding_backend(
                backend_name, settings.embedding_service_url, settings.embedding_timeout_seconds
            )

        try:
            await apply_ingestion_plan(pool, plan, embedding_backend=embedding_backend)
            embedded_note = f" (with embeddings)" if args.embed else " (no embeddings)"
            print(f"Applied: {plan.total_chunks} chunks, corpus_hash={plan.corpus_hash}{embedded_note}")
        finally:
            await pool.close()
    else:
        print(
            json.dumps(
                {
                    "logical_source_key": plan.source.logical_source_key,
                    "version": plan.source.version,
                    "source_hash": plan.source.hash_sha256,
                    "corpus_hash": plan.corpus_hash,
                    "total_sections": plan.total_sections,
                    "total_chunks": plan.total_chunks,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
