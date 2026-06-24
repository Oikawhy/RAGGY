from app.config import Settings


def test_settings_defaults_are_production_rag_defaults():
    settings = Settings(openai_api_key="test-key")
    assert settings.llm_model == "gpt-5.4-mini"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_config_hash == "default"
    assert settings.max_context_tokens == 3000
    assert settings.neighbor_window == 1


def test_settings_include_live_runtime_controls():
    settings = Settings(openai_api_key="test-key")
    assert settings.embedding_backend in ("fake", "http", "local_bge_m3")
    assert settings.embedding_service_url is None
    assert settings.embedding_timeout_seconds == 10
    assert settings.embedding_circuit_breaker_threshold == 5
    assert settings.embedding_circuit_breaker_cooldown == 60
    assert settings.llm_timeout_seconds == 30
    assert settings.llm_max_concurrency == 10
    assert settings.embedding_max_concurrency == 5
    assert settings.redis_control_plane_failure_mode == "fail_open"
    assert settings.trace_retention_days == 30
    assert settings.reranker_enabled is True
    assert settings.reranker_backend == "lexical_overlap"
    assert settings.reranker_model == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
