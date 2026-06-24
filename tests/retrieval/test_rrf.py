from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.models import RetrievedChunk


def test_rrf_prefers_chunk_present_in_both_rankings():
    vector = [RetrievedChunk(chunk_id="a", section="S", content="A", score=0.9, rank=1)]
    lexical = [
        RetrievedChunk(chunk_id="b", section="S", content="B", score=0.8, rank=1),
        RetrievedChunk(chunk_id="a", section="S", content="A", score=0.7, rank=2),
    ]
    fused = reciprocal_rank_fusion(vector, lexical, top_k=2)
    assert fused[0].chunk_id == "a"
    assert fused[0].rrf_score > fused[1].rrf_score
