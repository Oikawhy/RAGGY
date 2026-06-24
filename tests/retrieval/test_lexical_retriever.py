import contextlib

import pytest

from app.retrieval.lexical import PostgresLexicalRetriever, BM25LexicalIndex


class FakeConnection:
    async def fetch(self, sql, *args):
        assert "active_corpus_sources" in sql
        return [
            {
                "chunk_id": "fts1",
                "section": "4. Податкові строки",
                "content": "ЄСВ сплачується у строки звітного періоду",
                "score": 0.5,
                "chunk_index": 0,
                "section_order": 4,
                "content_hash": "h",
            }
        ]


class FakePool:
    @contextlib.asynccontextmanager
    async def acquire(self):
        yield FakeConnection()


@pytest.mark.anyio
async def test_postgres_lexical_retriever_merges_fts_and_bm25():
    bm25 = BM25LexicalIndex.from_chunks([
        {"chunk_id": "bm1", "section": "4. Податкові строки", "content": "ЄСВ дата", "section_order": 4, "chunk_index": 1}
    ])
    retriever = PostgresLexicalRetriever(FakePool(), bm25)
    results = await retriever.search("ЄСВ дата", "corpus1", top_k=5)
    assert {result.chunk_id for result in results} == {"fts1", "bm1"}
    assert results[0].rank == 1
