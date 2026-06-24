from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg


class DatabaseUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class DatabaseHealth:
    status: str
    pgvector: bool
    chunks_indexed: int


async def create_pool(database_url: str, min_size: int = 1, max_size: int = 10) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        command_timeout=30,
    )


async def check_database_health(connection: Any) -> DatabaseHealth:
    sql = """
    SELECT
      1 AS ok,
      EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS vector_available,
      COALESCE((SELECT COUNT(*) FROM knowledge_chunks), 0) AS chunks_indexed
    """
    try:
        row = await connection.fetchrow(sql)
    except Exception as exc:
        raise DatabaseUnavailable("database health check failed") from exc
    return DatabaseHealth(
        status="up",
        pgvector=bool(row["vector_available"]),
        chunks_indexed=int(row["chunks_indexed"]),
    )
