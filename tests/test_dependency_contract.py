from pathlib import Path


def test_runtime_dependencies_are_declared():
    requirements = Path("requirements.txt").read_text()
    for package in ["asyncpg", "redis", "tenacity", "sentence-transformers"]:
        assert package in requirements


def test_pytest_integration_marker_is_declared():
    text = Path("pyproject.toml").read_text()
    assert "integration" in text
    assert "requires Docker or live external service dependencies" in text


def test_env_example_documents_live_runtime_settings():
    text = Path(".env.example").read_text()
    for key in [
        "DATABASE_URL=",
        "REDIS_URL=",
        "OPENAI_API_KEY=",
        "EMBEDDING_BACKEND=",
        "EMBEDDING_SERVICE_URL=",
        "RERANKER_ENABLED=",
        "RERANKER_BACKEND=",
        "RERANKER_MODEL=",
        "TRACE_RETENTION_DAYS=",
    ]:
        assert key in text
