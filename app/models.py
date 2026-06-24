from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Confidence = Literal["high", "medium", "low"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    client_id: str | None = Field(default=None, description="Optional client identifier for per-client rate limiting")
    request_id: str | None = Field(default=None, description="Optional idempotency key for repeated request submissions")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class SourceRef(BaseModel):
    section: str
    chunk: str | None = None
    score: float = Field(ge=0)
    chunk_id: str | None = None
    content_hash: str | None = None
    preview: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    confidence: Confidence
    fallback_reason: str | None
    trace_id: str
    latency_ms: int = Field(ge=0)


class PipelineStageTrace(BaseModel):
    stage: str
    latency_ms: int = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    trace_id: str
    question: str | None = None
    pipeline_steps: list[PipelineStageTrace] = Field(default_factory=list)
    response: AskResponse | dict[str, Any] | None = None
    total_latency_ms: int = Field(ge=0)
    error: str | None = None
