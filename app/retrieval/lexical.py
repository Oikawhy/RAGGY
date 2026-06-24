from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Protocol

from app.retrieval.models import RetrievedChunk


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9]+", re.UNICODE)

QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("лікарня", "захвор", "медич", "довід", "документ", "оплат"),
        ("sick", "leave", "medical", "certificate", "absence", "paid", "payment", "processing"),
    ),
)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def expand_query_tokens(query: str) -> list[str]:
    tokens = tokenize(query)
    normalized = " ".join(tokens)
    expanded = list(tokens)
    for triggers, additions in QUERY_EXPANSIONS:
        if any(trigger in normalized for trigger in triggers):
            expanded.extend(additions)
    return expanded


def build_fts_sql() -> str:
    return """
    SELECT
        c.id::text AS chunk_id,
        c.section_title AS section,
        c.content,
        ts_rank(c.content_tsv, plainto_tsquery('simple', :query)) AS score,
        c.chunk_index,
        s.section_num AS section_order,
        c.content_hash
    FROM knowledge_chunks c
    JOIN active_corpus_sources acs ON acs.source_id = c.source_id
    JOIN knowledge_sections s ON s.id = c.section_id
    WHERE acs.corpus_hash = :corpus_hash
      AND c.is_rule = FALSE
      AND c.content_tsv @@ plainto_tsquery('simple', :query)
    ORDER BY score DESC
    LIMIT :top_k
    """.strip()


@dataclass(frozen=True)
class _BM25Document:
    chunk_id: str
    content: str
    section: str
    section_order: int
    chunk_index: int
    content_hash: str | None
    frequencies: Counter[str]
    length: int


class BM25LexicalIndex:
    def __init__(self, documents: list[_BM25Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avg_doc_len = sum(doc.length for doc in documents) / len(documents) if documents else 0.0
        self.doc_freqs: Counter[str] = Counter()
        for doc in documents:
            self.doc_freqs.update(doc.frequencies.keys())

    @classmethod
    def from_chunks(cls, chunks: list[dict[str, Any]]) -> "BM25LexicalIndex":
        documents = []
        for index, chunk in enumerate(chunks):
            content = chunk["content"]
            tokens = tokenize(content)
            documents.append(
                _BM25Document(
                    chunk_id=chunk["chunk_id"],
                    content=content,
                    section=chunk.get("section") or chunk.get("section_title") or "",
                    section_order=chunk.get("section_order", 0),
                    chunk_index=chunk.get("chunk_index", index),
                    content_hash=chunk.get("content_hash"),
                    frequencies=Counter(tokens),
                    length=max(1, len(tokens)),
                )
            )
        return cls(documents)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_terms = expand_query_tokens(query)
        scored: list[tuple[float, _BM25Document]] = []
        total_docs = len(self.documents)
        for doc in self.documents:
            score = 0.0
            for term in query_terms:
                frequency = doc.frequencies.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (total_docs - self.doc_freqs[term] + 0.5) / (self.doc_freqs[term] + 0.5))
                norm = frequency + self.k1 * (1 - self.b + self.b * doc.length / max(self.avg_doc_len, 1))
                score += idf * ((frequency * (self.k1 + 1)) / norm)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: (-item[0], item[1].section_order, item[1].chunk_index))
        return [
            RetrievedChunk(
                chunk_id=doc.chunk_id,
                section=doc.section,
                content=doc.content,
                score=score,
                rank=rank,
                section_order=doc.section_order,
                chunk_index=doc.chunk_index,
                content_hash=doc.content_hash,
            )
            for rank, (score, doc) in enumerate(scored[:top_k], start=1)
        ]


class LexicalRetriever(Protocol):
    def search(self, query: str, corpus_hash: str, top_k: int) -> list[RetrievedChunk]:
        ...


class PostgresLexicalRetriever:
    def __init__(self, pool, bm25_index: BM25LexicalIndex) -> None:
        self.pool = pool
        self.bm25_index = bm25_index

    async def search(self, query: str, corpus_hash: str, top_k: int) -> list[RetrievedChunk]:
        # Pass 1: PostgreSQL FTS
        async with self.pool.acquire() as connection:
            fts_rows = await connection.fetch(
                """
                SELECT
                    c.id::text AS chunk_id,
                    c.section_title AS section,
                    c.content,
                    ts_rank(c.content_tsv, plainto_tsquery('simple', $1)) AS score,
                    c.chunk_index,
                    s.section_num AS section_order,
                    c.content_hash
                FROM knowledge_chunks c
                JOIN active_corpus_sources acs ON acs.source_id = c.source_id
                JOIN knowledge_sections s ON s.id = c.section_id
                WHERE acs.corpus_hash = $2
                  AND c.is_rule = FALSE
                  AND c.content_tsv @@ plainto_tsquery('simple', $1)
                ORDER BY score DESC
                LIMIT $3
                """,
                query,
                corpus_hash,
                top_k,
            )
        merged: dict[str, RetrievedChunk] = {}
        for index, row in enumerate(fts_rows, start=1):
            merged[row["chunk_id"]] = RetrievedChunk(
                chunk_id=row["chunk_id"],
                section=row["section"],
                content=row["content"],
                score=float(row["score"]),
                rank=index,
                section_order=int(row["section_order"]),
                chunk_index=int(row["chunk_index"]),
                content_hash=row["content_hash"],
            )
        # Pass 2: in-memory BM25
        for result in self.bm25_index.search(query, top_k=top_k):
            current = merged.get(result.chunk_id)
            if current is None or result.score > current.score:
                merged[result.chunk_id] = result
        ordered = sorted(merged.values(), key=lambda item: (-item.score, item.section_order, item.chunk_index))
        return [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                section=item.section,
                content=item.content,
                score=item.score,
                rank=index,
                section_order=item.section_order,
                chunk_index=item.chunk_index,
                content_hash=item.content_hash,
                metadata=item.metadata,
            )
            for index, item in enumerate(ordered[:top_k], start=1)
        ]
