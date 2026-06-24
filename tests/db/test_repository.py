import pytest

from app.db.repository import KnowledgeRepository
from app.ingestion.loader import SourceManifestItem


class FakeConnection:
    def __init__(self):
        self.queries = []

    async def fetch(self, sql, *args):
        self.queries.append((sql, args))
        if "knowledge_sources" in sql:
            return [
                {"logical_source_key": "assignment-kb", "version": 1, "hash_sha256": "abc"},
            ]
        return [
            {
                "chunk_id": "c1",
                "section": "1. Щорічна відпустка",
                "content": "Працівник може використати відпустку після 6 місяців",
                "score": 0.0,
                "rank": 1,
                "section_order": 1,
                "chunk_index": 0,
                "content_hash": "h1",
                "has_disclaimer": False,
            }
        ]


@pytest.mark.anyio
async def test_load_active_manifest_items():
    repo = KnowledgeRepository(FakeConnection())
    items = await repo.load_active_manifest_items()
    assert items == [SourceManifestItem("assignment-kb", 1, "abc")]


@pytest.mark.anyio
async def test_load_active_retrievable_chunks_filters_by_corpus():
    conn = FakeConnection()
    repo = KnowledgeRepository(conn)
    chunks = await repo.load_active_retrievable_chunks("corpus1")
    assert chunks[0]["chunk_id"] == "c1"
    assert "active_corpus_sources" in conn.queries[-1][0]
    assert conn.queries[-1][1] == ("corpus1",)
