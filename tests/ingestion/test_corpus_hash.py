from app.ingestion.loader import SourceManifestItem, compute_corpus_hash


def test_corpus_hash_is_order_independent_and_version_sensitive():
    a = SourceManifestItem(logical_source_key="kb", version=1, hash_sha256="aaa")
    b = SourceManifestItem(logical_source_key="policy", version=2, hash_sha256="bbb")
    assert compute_corpus_hash([a, b]) == compute_corpus_hash([b, a])
    assert compute_corpus_hash([a, b]) != compute_corpus_hash([SourceManifestItem("kb", 2, "aaa"), b])
    assert len(compute_corpus_hash([a, b])) == 16
