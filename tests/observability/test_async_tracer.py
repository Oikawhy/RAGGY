import json
from pathlib import Path

import pytest

from app.observability.tracer import AsyncTraceWriter


@pytest.mark.anyio
async def test_async_trace_writer_writes_jsonl(tmp_path):
    writer = AsyncTraceWriter(trace_dir=tmp_path, max_queue_size=10)
    await writer.start()
    await writer.enqueue({"question": "test", "confidence": 0.5})
    await writer.flush()
    await writer.stop()
    trace_file = tmp_path / "traces.jsonl"
    assert trace_file.exists()
    lines = trace_file.read_text().strip().splitlines()
    assert json.loads(lines[0])["question"] == "test"


@pytest.mark.anyio
async def test_async_trace_writer_retention_removes_old_files(tmp_path):
    old_file = tmp_path / "traces-2020-01-01.jsonl"
    old_file.write_text("{}\n")
    writer = AsyncTraceWriter(trace_dir=tmp_path, max_queue_size=10, retention_days=1)
    writer.cleanup_old_traces()
    assert not old_file.exists()


@pytest.mark.anyio
async def test_async_trace_writer_reports_backpressure(tmp_path, capsys):
    writer = AsyncTraceWriter(trace_dir=tmp_path, max_queue_size=1, enqueue_timeout_ms=1)
    await writer._queue.put({"already": "queued"})

    ok = await writer.enqueue({"trace_id": "t1"})

    assert ok is False
    assert writer.dropped_trace_count == 1
    captured = capsys.readouterr()
    assert "trace_enqueue_dropped" in captured.err
    assert "t1" in captured.err
