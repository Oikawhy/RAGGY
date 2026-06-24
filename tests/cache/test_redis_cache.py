from app.cache.redis_cache import (
    RedisFailurePolicy,
    build_answer_cache_key,
    build_idempotency_key,
    build_query_embedding_cache_key,
    build_rate_limit_key,
)


def test_answer_cache_key_contains_versions():
    key = build_answer_cache_key(
        corpus_hash="abc123",
        prompt_version="p1",
        llm_model="gpt-5.4-mini",
        retrieval_config_hash="r1",
        question_hash="q1",
    )
    assert key == "answer:abc123:p1:gpt-5.4-mini:r1:q1"


def test_query_embedding_cache_key_contains_embedding_config():
    key = build_query_embedding_cache_key(
        embedding_model="BAAI/bge-m3",
        embedding_config_hash="embed-v1",
        question_hash="q1",
    )
    assert key == "query_embedding:BAAI/bge-m3:embed-v1:q1"


def test_idempotency_key_uses_request_id():
    assert build_idempotency_key(request_id="req-123") == "idempotency:req-123"


def test_rate_limit_key_uses_subject_and_window():
    assert build_rate_limit_key(subject="client-1", window="60s") == "rate_limit:client-1:60s"


def test_cache_failure_policy_distinguishes_cache_from_control_plane():
    assert RedisFailurePolicy.CACHE_FAIL_OPEN.value == "cache_fail_open"
    assert RedisFailurePolicy.CONTROL_PLANE_CONFIGURABLE.value == "control_plane_configurable"
