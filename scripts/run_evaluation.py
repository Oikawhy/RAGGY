from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from app.models import AskRequest
from app.pipeline.orchestrator import PipelineOrchestrator


def load_questions(path: str) -> list[dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("questions file must contain a JSON list")
    return data


def run_evaluation(
    questions_path: str,
    orchestrator: PipelineOrchestrator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in load_questions(questions_path):
        response = orchestrator.ask(AskRequest(question=item["question"]))
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": response.answer,
                "confidence": response.confidence,
                "fallback_reason": response.fallback_reason,
                "sources": [source.model_dump() for source in response.sources],
            }
        )
    return rows


async def run_live_evaluation(
    questions_path: str,
    orchestrator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in load_questions(questions_path):
        response = await orchestrator.ask(AskRequest(question=item["question"]))
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": response.answer,
                "confidence": response.confidence,
                "fallback_reason": response.fallback_reason,
                "sources": [source.model_dump() for source in response.sources],
            }
        )
    return rows


async def _run_live_against_server(questions_path: str, base_url: str) -> list[dict[str, Any]]:
    """Run evaluation by POSTing to a running production server."""
    import httpx

    rows: list[dict[str, Any]] = []
    questions = load_questions(questions_path)
    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in questions:
            resp = await client.post(
                f"{base_url}/ask",
                json={"question": item["question"]},
            )
            resp.raise_for_status()
            data = resp.json()
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": data.get("answer", ""),
                    "confidence": data.get("confidence", ""),
                    "fallback_reason": data.get("fallback_reason"),
                    "sources": data.get("sources", []),
                    "latency_ms": data.get("latency_ms", 0),
                    "trace_id": data.get("trace_id", ""),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation questions against an injected/local pipeline.")
    parser.add_argument("--questions", default="data/test_questions.json")
    parser.add_argument("--live", action="store_true", help="Use live async orchestrator via running server.")
    parser.add_argument("--server", default="http://localhost:8765", help="Server URL for --live mode.")
    args = parser.parse_args()

    if args.live:
        rows = asyncio.run(_run_live_against_server(args.questions, args.server))
    else:
        rows = run_evaluation(args.questions, PipelineOrchestrator())

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
