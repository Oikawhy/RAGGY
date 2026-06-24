import pytest

from app.generation.llm import OpenAILLMClient


class FakeResponses:
    def __init__(self, output_text='{"answer":"Так","fallback_reason":null,"used_sections":["S"]}'):
        self.kwargs = None
        self.output_text = output_text

    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("Response", (), {"output_text": self.output_text})()


class FakeOpenAI:
    def __init__(self, output_text='{"answer":"Так","fallback_reason":null,"used_sections":["S"]}'):
        self.responses = FakeResponses(output_text)


@pytest.mark.anyio
async def test_openai_client_uses_model_timeout_and_schema():
    fake = FakeOpenAI()
    client = OpenAILLMClient(api_key="key", model="gpt-5.4-mini", timeout_seconds=7, openai_client=fake)
    result = await client.generate("prompt")
    assert result.answer == "Так"
    assert fake.responses.kwargs["model"] == "gpt-5.4-mini"
    assert fake.responses.kwargs["timeout"] == 7
    assert fake.responses.kwargs["text"]["format"]["type"] == "json_schema"
