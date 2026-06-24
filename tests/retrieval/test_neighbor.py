import contextlib

import pytest

from app.retrieval.neighbor import expand_with_neighbors


class FakeConnection:
    async def fetch(self, sql, *args):
        return [
            {
                "chunk_id": "n0",
                "section": "1. Щорічна відпустка",
                "content": "Попередній параграф",
                "chunk_index": -1,
                "section_order": 1,
                "content_hash": "h0",
            },
            {
                "chunk_id": "n2",
                "section": "1. Щорічна відпустка",
                "content": "Наступний параграф",
                "chunk_index": 1,
                "section_order": 1,
                "content_hash": "h2",
            },
        ]


class FakePool:
    @contextlib.asynccontextmanager
    async def acquire(self):
        yield FakeConnection()


@pytest.mark.anyio
async def test_expand_with_neighbors_adds_adjacent_chunks():
    expanded = await expand_with_neighbors(
        pool=FakePool(),
        chunk_ids=["c1"],
        window=1,
        corpus_hash="corpus1",
    )
    assert len(expanded) == 2
    assert {chunk["chunk_id"] for chunk in expanded} == {"n0", "n2"}
