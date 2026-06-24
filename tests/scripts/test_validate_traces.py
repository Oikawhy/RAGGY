import json

from scripts.validate_traces import validate_trace_line


def test_validate_trace_line_requires_sources_in_response():
    line = json.dumps({
        "trace_id": "t1",
        "pipeline_steps": [],
        "response": {"answer": "A", "sources": [], "confidence": "low", "fallback_reason": "x"},
        "total_latency_ms": 1,
        "error": None,
    })
    assert validate_trace_line(line).ok is True
