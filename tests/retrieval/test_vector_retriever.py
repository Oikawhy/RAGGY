import contextlib

import pytest

from app.retrieval.vector import PostgresVectorRetriever


class FakeConnection:
    def __init__(self):
        self.calls = []

    async def execute(self, sql):
        self.calls.append(("execute", sql))

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        return [
            {
                "chunk_id": "c1",
                "section": "1. Щорічна відпустка",
                "content": "Працівник може використати відпустку після 6 місяців",
                "score": 0.87,
                "chunk_index": 0,
                "section_order": 1,
                "content_hash": "h1",
            }
        ]


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield self.conn


@pytest.mark.anyio
async def test_vector_retriever_sets_hnsw_and_maps_rows():
    conn = FakeConnection()
    retriever = PostgresVectorRetriever(FakePool(conn))
    results = await retriever.search([0.1] * 1024, "corpus1", top_k=20)
    assert conn.calls[0] == ("execute", "SET LOCAL hnsw.ef_search = 100")
    assert results[0].chunk_id == "c1"
    assert results[0].rank == 1
