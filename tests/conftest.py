from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import asyncio
import inspect

import pytest
import fastapi.testclient

from app.generation.llm import FakeLLMClient
from app.ingestion.chunker import chunk_sections
from app.ingestion.parser import parse_markdown_sections
from app.models import AskRequest
from app.pipeline.context import ContextItem


class _LocalResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _LocalClient:
    """Small pytest-only fallback because AnyIO's blocking portal hangs in this sandbox."""

    __test__ = False

    def __init__(self, app):
        self.app = app

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, path: str):
        return self._request("GET", path, None)

    def post(self, path: str, json=None):
        return self._request("POST", path, json)

    def _request(self, method: str, path: str, payload):
        for route in self.app.routes:
            if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
                if path == "/ask":
                    result = route.endpoint(AskRequest.model_validate(payload or {}))
                else:
                    result = route.endpoint()
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
                if hasattr(result, "model_dump"):
                    result = result.model_dump(mode="json")
                return _LocalResponse(200, result)
        return _LocalResponse(404, {"detail": "Not Found"})


def pytest_configure(config):
    fastapi.testclient.TestClient = _LocalClient


def _vacation_context() -> list[ContextItem]:
    return [
        ContextItem(
            chunk_id="s1-c0",
            section="1. Щорічна відпустка",
            content="Працівник може використати щорічну оплачувану відпустку після 6 місяців безперервної роботи у компанії.",
            score=0.033,
            section_order=1,
            chunk_index=0,
            content_hash="vacation",
        )
    ]


@dataclass(frozen=True)
class FakePipelineDeps:
    def with_no_formula_context(self) -> dict:
        return {
            "context_chunks": [],
            "calculation_supported": False,
            "llm_client": FakeLLMClient({"answer": "Не використовується", "fallback_reason": None, "used_sections": []}),
        }

    def with_vacation_context(self) -> dict:
        return {
            "context_chunks": _vacation_context(),
            "calculation_supported": True,
            "llm_client": FakeLLMClient(
                {
                    "answer": "У базі знань вказано, що щорічну оплачувану відпустку можна використати після 6 місяців безперервної роботи.",
                    "fallback_reason": None,
                    "used_sections": ["1. Щорічна відпустка"],
                }
            ),
        }

    def with_assignment_kb(self) -> dict:
        sections = parse_markdown_sections(Path("data/knowledge_base.md").read_text(encoding="utf-8"))
        chunks = chunk_sections(sections)

        def retrieve(question: str) -> list[ContextItem]:
            lowered = question.casefold()
            if "відпуст" in lowered:
                selected = [chunk for chunk in chunks if chunk.section_num == 1][:2]
            elif "лікарня" in lowered or "медич" in lowered:
                selected = [chunk for chunk in chunks if chunk.section_num == 2][:2]
            elif "індексац" in lowered or "зарплат" in lowered:
                selected = [chunk for chunk in chunks if chunk.section_num == 3][:3]
            elif "єсв" in lowered:
                selected = [chunk for chunk in chunks if chunk.section_num == 4][:2]
            else:
                selected = chunks[:1]
            return [
                ContextItem(
                    chunk_id=chunk.chunk_id,
                    section=chunk.section_title,
                    content=chunk.content,
                    score=0.033 if index == 0 else 0.02,
                    section_order=chunk.section_num,
                    chunk_index=chunk.chunk_index,
                    content_hash=chunk.content_hash,
                )
                for index, chunk in enumerate(selected)
            ]

        return {
            "retriever": retrieve,
            "calculation_supported": False,
            "llm_client": FakeLLMClient(
                {
                    "answer": "За наданою базою знань, щорічну оплачувану відпустку можна використати після 6 місяців безперервної роботи у компанії.",
                    "fallback_reason": None,
                    "used_sections": ["1. Щорічна відпустка"],
                }
            ),
        }


@pytest.fixture
def fake_pipeline_deps() -> FakePipelineDeps:
    return FakePipelineDeps()
