from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    section: str
    content: str
    score: float
    rank: int
    section_order: int = 0
    chunk_index: int = 0
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedChunk:
    chunk_id: str
    section: str
    content: str
    rrf_score: float
    section_order: int = 0
    chunk_index: int = 0
    content_hash: str | None = None
    vector_rank: int | None = None
    lexical_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
