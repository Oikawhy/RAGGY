from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
from queue import Queue
import sys
from typing import Any


class TraceWriter:
    def __init__(self, trace_dir: str | Path, max_queue_size: int, enqueue_timeout_ms: int = 50) -> None:
        self.trace_dir = Path(trace_dir)
        self.max_queue_size = max_queue_size
        self.enqueue_timeout_ms = enqueue_timeout_ms
        self.queue: Queue[dict[str, Any]] = Queue(maxsize=max_queue_size)
        self.dropped_trace_count = 0
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_today(self) -> Path:
        return self.trace_dir / "traces.jsonl"

    def write_sync(self, record: dict[str, Any]) -> None:
        path = self._path_for_today()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def enqueue(self, record: dict[str, Any]) -> bool:
        timeout = self.enqueue_timeout_ms / 1000
        try:
            self.queue.put(record, block=True, timeout=timeout)
        except Exception:
            self.dropped_trace_count += 1
            self._write_dropped_trace(record, "trace_enqueue_dropped")
            return False
        return True

    def _write_dropped_trace(self, record: dict[str, Any], reason: str) -> None:
        payload = {
            "event": reason,
            "trace_id": record.get("trace_id"),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)


class AsyncTraceWriter:
    def __init__(
        self,
        trace_dir: str | Path,
        max_queue_size: int,
        enqueue_timeout_ms: int = 50,
        retention_days: int = 30,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.retention_days = retention_days
        self.enqueue_timeout_ms = enqueue_timeout_ms
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=max_queue_size)
        self._task: asyncio.Task | None = None
        self.dropped_trace_count = 0
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_today(self) -> Path:
        return self.trace_dir / "traces.jsonl"

    async def start(self) -> None:
        self._task = asyncio.create_task(self._writer_loop())

    async def stop(self) -> None:
        await self._queue.put(None)
        if self._task:
            await self._task

    async def enqueue(self, record: dict[str, Any]) -> bool:
        try:
            timeout = self.enqueue_timeout_ms / 1000
            await asyncio.wait_for(self._queue.put(record), timeout=timeout)
            return True
        except (asyncio.TimeoutError, asyncio.QueueFull):
            self.dropped_trace_count += 1
            self._write_dropped_trace(record, "trace_enqueue_dropped")
            return False

    def _write_dropped_trace(self, record: dict[str, Any], reason: str) -> None:
        payload = {
            "event": reason,
            "trace_id": record.get("trace_id"),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)

    async def flush(self, timeout_seconds: float = 5.0) -> None:
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            pass  # don't hang; best-effort flush

    async def _writer_loop(self) -> None:
        while True:
            record = await self._queue.get()
            if record is None:
                self._queue.task_done()
                break
            path = self._path_for_today()
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            await asyncio.to_thread(self._append_line, path, line)
            self._queue.task_done()

    def _append_line(self, path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def cleanup_old_traces(self) -> None:
        import re
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(days=self.retention_days)
        for file in self.trace_dir.glob("traces-*.jsonl"):
            match = re.search(r"traces-(\d{4}-\d{2}-\d{2})\.jsonl$", file.name)
            if match:
                file_date = datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=UTC)
                if file_date < cutoff:
                    file.unlink(missing_ok=True)
