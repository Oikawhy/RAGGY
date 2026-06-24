from app.ingestion.loader import attach_embeddings_to_chunks
from app.ingestion.chunker import Chunk


class FakeBackend:
    def embed_documents(self, texts):
        assert texts == ["A", "B"]
        return [[0.1] * 1024, [0.2] * 1024]


def make_chunk(chunk_index, content):
    return Chunk(
        chunk_id=f"c{chunk_index}",
        section_num=1,
        section_title="1. Test",
        chunk_index=chunk_index,
        content=content,
        content_hash=f"h{chunk_index}",
        content_tokens=1,
        language="uk",
    )


def test_attach_embeddings_to_chunks_keeps_chunk_metadata():
    embedded = attach_embeddings_to_chunks([make_chunk(0, "A"), make_chunk(1, "B")], FakeBackend())
    assert embedded[0].chunk.chunk_id == "c0"
    assert embedded[0].embedding == [0.1] * 1024
    assert embedded[1].embedding == [0.2] * 1024
