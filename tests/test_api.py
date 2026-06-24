from fastapi.testclient import TestClient

from app.main import create_app


def test_ask_endpoint_returns_structured_response(fake_pipeline_deps):
    app = create_app(pipeline_deps=fake_pipeline_deps.with_vacation_context())
    client = TestClient(app)
    response = client.post("/ask", json={"question": "Чи можна у відпустку після 3 місяців?"})
    assert response.status_code == 200
    body = response.json()
    assert {"answer", "sources", "confidence", "fallback_reason", "trace_id", "latency_ms"} <= set(body)


def test_health_endpoint_reports_components(fake_pipeline_deps):
    app = create_app(pipeline_deps=fake_pipeline_deps.with_vacation_context())
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "postgres" in response.json()["checks"]
