import pytest

from scripts.ingest_kb import apply_ingestion_plan, build_ingestion_plan


class FakePool:
    def __init__(self):
        self.connection = FakeConnection()

    def acquire(self):
        return self.connection


class FakeConnection:
    def __init__(self):
        self.applied = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.anyio
async def test_apply_ingestion_plan_calls_persistence(monkeypatch):
    called = {}

    async def fake_persist(connection, record):
        called["corpus_hash"] = record.corpus_hash
        called["chunks"] = len(record.chunks)

    monkeypatch.setattr("scripts.ingest_kb.persist_ingestion_records", fake_persist)
    plan = build_ingestion_plan("data/knowledge_base.md", "assignment-kb", 1)
    await apply_ingestion_plan(FakePool(), plan)
    assert len(called["corpus_hash"]) == 16
    assert called["chunks"] > 10
