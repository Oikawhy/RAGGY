from app.pipeline.context import assemble_context
from app.retrieval.models import FusedChunk


def test_context_assembler_deduplicates_and_respects_max_chunks():
    chunks = [
        FusedChunk(chunk_id="1", section="S", content="A", rrf_score=0.03, section_order=1, chunk_index=1),
        FusedChunk(chunk_id="1", section="S", content="A", rrf_score=0.03, section_order=1, chunk_index=1),
        FusedChunk(chunk_id="2", section="S", content="B", rrf_score=0.02, section_order=1, chunk_index=2),
    ]
    context = assemble_context(chunks, neighbors=[], max_chunks=2, max_context_tokens=100)
    assert [item.chunk_id for item in context.items] == ["1", "2"]
