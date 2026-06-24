from pathlib import Path


def test_docker_compose_declares_postgres_pgvector_and_redis():
    compose = Path("docker-compose.yml").read_text()
    assert "pgvector/pgvector:pg16" in compose
    assert "redis:7-alpine" in compose
    assert "5433:5432" in compose


def test_docker_compose_declares_advisory_resource_limits():
    compose = Path("docker-compose.yml").read_text()
    assert "memory: 2G" in compose
    assert 'cpus: "4"' in compose
    assert "memory: 256M" in compose


def test_dockerfile_runs_fastapi_app():
    dockerfile = Path("Dockerfile").read_text()
    assert "uvicorn" in dockerfile
    assert "app.main" in dockerfile
