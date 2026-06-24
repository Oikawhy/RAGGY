from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    line_number: int | None = None


def validate_trace_line(line: str, line_number: int | None = None) -> TraceValidationResult:
    errors: list[str] = []
    try:
        record: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError as exc:
        return TraceValidationResult(ok=False, errors=[f"invalid JSON: {exc}"], line_number=line_number)

    for key in ("trace_id", "pipeline_steps", "response", "total_latency_ms", "error"):
        if key not in record:
            errors.append(f"missing {key}")
    response = record.get("response")
    if not isinstance(response, dict):
        errors.append("response must be an object")
    else:
        for key in ("answer", "sources", "confidence", "fallback_reason"):
            if key not in response:
                errors.append(f"missing response.{key}")
        if "sources" in response and not isinstance(response["sources"], list):
            errors.append("response.sources must be a list")

    return TraceValidationResult(ok=not errors, errors=errors, line_number=line_number)


def validate_trace_file(path: str) -> list[TraceValidationResult]:
    results: list[TraceValidationResult] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            results.append(validate_trace_line(line, line_number=line_number))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trace JSONL records.")
    parser.add_argument("path")
    args = parser.parse_args()
    results = validate_trace_file(args.path)
    failures = [result for result in results if not result.ok]
    if failures:
        for failure in failures:
            print(f"line {failure.line_number}: {', '.join(failure.errors)}")
        raise SystemExit(1)
    print(f"validated {len(results)} trace lines")


if __name__ == "__main__":
    main()
