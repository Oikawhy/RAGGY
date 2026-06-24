from app.generation.validator import compute_confidence, detect_llm_insufficiency


def test_confidence_high_when_retrieval_is_strong_and_no_insufficiency():
    confidence = compute_confidence(
        top_rrf_score=0.033,
        supporting_chunks=3,
        vector_lexical_overlap=1.0,
        has_disclaimer=False,
        llm_signals_insufficiency=False,
    )
    assert confidence == "high"


def test_detects_llm_insufficiency_signal():
    assert detect_llm_insufficiency("У базі знань недостатньо даних для точної відповіді")
