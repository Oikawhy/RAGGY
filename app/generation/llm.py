from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from app.generation.prompt import llm_response_json_schema


class MalformedLLMResponse(RuntimeError):
    """Raised when an LLM response does not match the internal schema."""


class LLMResponse(BaseModel):
    answer: str
    fallback_reason: str | None
    used_sections: list[str]


class LLMClient(Protocol):
    def generate(self, prompt: str) -> LLMResponse:
        ...


def _validate_response(payload: dict[str, Any]) -> LLMResponse:
    try:
        return LLMResponse.model_validate(payload)
    except ValidationError as exc:
        raise MalformedLLMResponse(str(exc)) from exc


class FakeLLMClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def generate(self, prompt: str) -> LLMResponse:
        return _validate_response(self.response)


class OpenAILLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int = 30,
        max_concurrency: int = 10,
        openai_client=None,
    ) -> None:
        from openai import OpenAI

        self.client = openai_client or OpenAI(api_key=api_key)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def generate(self, prompt: str) -> LLMResponse:
        async with self.semaphore:
            return await asyncio.to_thread(self._generate_sync, prompt)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    def _generate_sync(self, prompt: str) -> LLMResponse:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            timeout=self.timeout_seconds,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ai_consultant_answer",
                    "schema": llm_response_json_schema(),
                    "strict": True,
                }
            },
        )
        try:
            payload = json.loads(response.output_text)
        except (AttributeError, json.JSONDecodeError) as exc:
            raise MalformedLLMResponse("OpenAI response did not contain valid JSON text") from exc
        return _validate_response(payload)
