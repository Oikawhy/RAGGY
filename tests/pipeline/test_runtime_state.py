import pytest

from app.ingestion.loader import SourceManifestItem
from app.pipeline.runtime import RuntimeState, build_runtime_state


class FakeRepository:
    async def load_active_manifest_items(self):
        return [SourceManifestItem("assignment-kb", 1, "abc")]

    async def load_active_retrievable_chunks(self, corpus_hash):
        return [{"chunk_id": "c1", "content": "ЄСВ", "section": "4. Податкові строки", "section_order": 4, "chunk_index": 0}]

    async def load_active_rule_chunks(self, corpus_hash):
        return [{"chunk_id": "r1", "content": "Відповідай українською", "section": "10. Правила"}]


@pytest.mark.anyio
async def test_build_runtime_state_computes_corpus_hash_and_bm25():
    state = await build_runtime_state(FakeRepository())
    assert isinstance(state, RuntimeState)
    assert len(state.corpus_hash) == 16
    assert state.knowledge_base_version == 1
    assert state.bm25_index.search("ЄСВ", top_k=1)[0].chunk_id == "c1"
    assert state.rule_chunks[0]["chunk_id"] == "r1"
