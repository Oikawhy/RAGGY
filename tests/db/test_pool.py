import pytest

from app.db.pool import DatabaseHealth, DatabaseUnavailable, check_database_health


class FakeConnection:
    async def fetchrow(self, sql):
        assert "SELECT" in sql
        return {"ok": 1, "vector_available": True, "chunks_indexed": 28}


class FailingConnection:
    async def fetchrow(self, sql):
        raise RuntimeError("database down")


@pytest.mark.anyio
async def test_check_database_health_reports_pgvector_and_chunks():
    health = await check_database_health(FakeConnection())
    assert health == DatabaseHealth(status="up", pgvector=True, chunks_indexed=28)


@pytest.mark.anyio
async def test_check_database_health_wraps_connection_errors():
    with pytest.raises(DatabaseUnavailable):
        await check_database_health(FailingConnection())
