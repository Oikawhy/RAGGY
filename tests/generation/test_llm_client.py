import pytest

from app.generation.llm import FakeLLMClient, LLMResponse, MalformedLLMResponse


def test_fake_llm_returns_structured_response():
    client = FakeLLMClient(response={"answer": "Так", "fallback_reason": None, "used_sections": ["S"]})
    result = client.generate(prompt="prompt")
    assert isinstance(result, LLMResponse)
    assert result.answer == "Так"


def test_llm_client_rejects_missing_answer():
    client = FakeLLMClient(response={"fallback_reason": None, "used_sections": []})
    with pytest.raises(MalformedLLMResponse):
        client.generate(prompt="prompt")
