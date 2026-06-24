from fastapi.testclient import TestClient

from app.main import create_app


def test_assignment_question_q001_returns_vacation_source(fake_pipeline_deps):
    app = create_app(pipeline_deps=fake_pipeline_deps.with_assignment_kb())
    client = TestClient(app)
    response = client.post("/ask", json={"question": "Я працюю в компанії 3 місяці. Чи можу вже піти у щорічну оплачувану відпустку?"})
    body = response.json()
    assert response.status_code == 200
    assert body["sources"][0]["section"] == "1. Щорічна відпустка"
    assert "6 місяців" in body["answer"]


def test_assignment_question_q005_falls_back_for_missing_base_month(fake_pipeline_deps):
    app = create_app(pipeline_deps=fake_pipeline_deps.with_assignment_kb())
    client = TestClient(app)
    response = client.post("/ask", json={"question": "Порахуй індексацію для працівника із зарплатою 25000 грн, якщо базовий місяць невідомий."})
    body = response.json()
    assert response.status_code == 200
    assert body["fallback_reason"] is not None
    assert body["confidence"] == "low"
