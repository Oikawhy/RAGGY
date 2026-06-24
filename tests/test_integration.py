import os

import pytest

pytestmark = pytest.mark.integration
skip_integration = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Integration tests require Docker + live services. Set RUN_INTEGRATION_TESTS=1.",
)


@skip_integration
@pytest.mark.anyio
async def test_database_schema_applies_cleanly():
    from app.db.pool import create_pool, check_database_health

    pool = await create_pool(os.environ.get("DATABASE_URL", "postgresql://consultant:consultant_dev_pw@localhost:5433/ai_consultant"))
    try:
        async with pool.acquire() as conn:
            health = await check_database_health(conn)
            assert health.status == "up"
            assert health.pgvector is True
    finally:
        await pool.close()


@skip_integration
@pytest.mark.anyio
async def test_ingestion_apply_inserts_chunks():
    from app.db.pool import create_pool
    from scripts.ingest_kb import build_ingestion_plan, apply_ingestion_plan

    pool = await create_pool(os.environ.get("DATABASE_URL", "postgresql://consultant:consultant_dev_pw@localhost:5433/ai_consultant"))
    try:
        plan = build_ingestion_plan("data/knowledge_base.md", "assignment-kb", 1)
        await apply_ingestion_plan(pool, plan)
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT count(*) FROM knowledge_chunks")
            assert count >= 10
    finally:
        await pool.close()


@skip_integration
@pytest.mark.anyio
async def test_live_ask_returns_structured_response():
    import httpx

    base_url = os.environ.get("LIVE_API_URL", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post("/ask", json={"question": "Коли працівник може отримати відпустку?"})
        assert response.status_code == 200
        body = response.json()
        assert {"answer", "sources", "confidence", "trace_id"} <= set(body)
