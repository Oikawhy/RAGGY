import pytest
from pydantic import ValidationError

from app.models import AskRequest, AskResponse, SourceRef


def test_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_ask_request_accepts_idempotency_request_id():
    request = AskRequest(question="Питання", request_id="req-1", client_id="client-1")
    assert request.request_id == "req-1"
    assert request.client_id == "client-1"


def test_ask_response_contract_accepts_valid_payload():
    response = AskResponse(
        answer="За наданою базою знань...",
        sources=[SourceRef(section="1. Щорічна відпустка", chunk="Працівник...", score=0.033)],
        confidence="high",
        fallback_reason=None,
        trace_id="trace-1",
        latency_ms=850,
    )
    assert response.confidence == "high"
    assert response.sources[0].section == "1. Щорічна відпустка"
