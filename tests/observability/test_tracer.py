from app.observability.tracer import TraceWriter


def test_trace_writer_serializes_sources(tmp_path):
    writer = TraceWriter(trace_dir=tmp_path, max_queue_size=10)
    record = {
        "trace_id": "t1",
        "response": {
            "answer": "A",
            "sources": [{"section": "S", "chunk_id": "c1", "score": 0.033, "content_hash": "h", "preview": "P"}],
            "confidence": "high",
            "fallback_reason": None,
        },
    }
    writer.write_sync(record)
    content = (tmp_path / "traces.jsonl").read_text()
    assert '"chunk_id": "c1"' in content
