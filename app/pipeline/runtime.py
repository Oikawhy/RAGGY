from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.loader import SourceManifestItem, compute_corpus_hash
from app.retrieval.lexical import BM25LexicalIndex


@dataclass(frozen=True)
class RuntimeState:
    corpus_hash: str
    bm25_index: BM25LexicalIndex
    rule_chunks: list[dict]
    retrievable_chunks: list[dict]
    manifest_items: list[SourceManifestItem] = field(default_factory=list)

    @property
    def knowledge_base_version(self) -> int | list[int] | None:
        versions = sorted({item.version for item in self.manifest_items})
        if not versions:
            return None
        if len(versions) == 1:
            return versions[0]
        return versions


async def build_runtime_state(repository) -> RuntimeState:
    manifest_items = await repository.load_active_manifest_items()
    corpus_hash = compute_corpus_hash(manifest_items)
    retrievable_chunks = await repository.load_active_retrievable_chunks(corpus_hash)
    rule_chunks = await repository.load_active_rule_chunks(corpus_hash)
    return RuntimeState(
        corpus_hash=corpus_hash,
        manifest_items=manifest_items,
        bm25_index=BM25LexicalIndex.from_chunks(retrievable_chunks),
        rule_chunks=rule_chunks,
        retrievable_chunks=retrievable_chunks,
    )
