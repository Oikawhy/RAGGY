import pytest

from app.ingestion.chunker import Chunk
from app.ingestion.loader import IngestionRecord, SourceManifestItem, persist_ingestion_records
from app.ingestion.parser import Section


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self):
        self.executed = []
        self.source_id = "source-uuid"
        self.section_id = "section-uuid"

    def transaction(self):
        return FakeTransaction()

    async def fetchval(self, sql, *args):
        self.executed.append((sql, args))
        if "knowledge_sources" in sql:
            return self.source_id
        if "knowledge_sections" in sql:
            return self.section_id
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "OK"


@pytest.mark.anyio
async def test_persist_ingestion_records_upserts_source_sections_chunks_and_manifest():
    conn = FakeConnection()
    source = SourceManifestItem("assignment-kb", 1, "abc")
    section = Section(section_num=1, title="Щорічна відпустка", raw_content="Text", language="uk")
    chunk = Chunk(
        chunk_id="s1-c0",
        section_num=1,
        section_title="1. Щорічна відпустка",
        chunk_index=0,
        content="Працівник може використати відпустку після 6 місяців",
        content_hash="h1",
        content_tokens=3,
        language="uk",
    )
    record = IngestionRecord(source=source, filename="knowledge_base.md", sections=[section], chunks=[chunk], corpus_hash="corpus1")
    await persist_ingestion_records(conn, record)
    sql_text = "\n".join(sql for sql, _ in conn.executed)
    assert "INSERT INTO knowledge_sources" in sql_text
    assert "INSERT INTO knowledge_sections" in sql_text
    assert "INSERT INTO knowledge_chunks" in sql_text
    assert "INSERT INTO active_corpus_sources" in sql_text
