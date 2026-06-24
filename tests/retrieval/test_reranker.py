from app.retrieval.models import FusedChunk
from app.retrieval.reranker import LexicalOverlapReranker


def test_lexical_reranker_prefers_question_matching_chunk_over_rrf_order():
    reranker = LexicalOverlapReranker()
    chunks = [
        FusedChunk(chunk_id="a", section="S", content="Загальне правило про документи", rrf_score=0.04),
        FusedChunk(
            chunk_id="b",
            section="S",
            content="Щорічна відпустка доступна після 6 місяців безперервної роботи",
            rrf_score=0.02,
        ),
    ]

    reranked = reranker.rerank("Коли доступна щорічна відпустка після роботи?", chunks, top_k=2)

    assert [chunk.chunk_id for chunk in reranked] == ["b", "a"]
    assert reranked[0].metadata["rerank_score"] > reranked[1].metadata["rerank_score"]
