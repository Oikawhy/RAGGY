from app.retrieval.vector import build_vector_search_sql


def test_vector_sql_uses_active_corpus_and_partial_hnsw_condition():
    sql = build_vector_search_sql()
    assert "JOIN active_corpus_sources" in sql
    assert "acs.corpus_hash" in sql
    assert "c.embedding IS NOT NULL" in sql
    assert "ORDER BY c.embedding <=>" in sql
