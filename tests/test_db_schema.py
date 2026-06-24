from pathlib import Path


def test_schema_contains_active_corpus_manifest_and_composite_fk():
    sql = Path("db/init.sql").read_text()
    assert "CREATE TABLE active_corpus_sources" in sql
    assert "FOREIGN KEY (section_id, source_id)" in sql
    assert "REFERENCES knowledge_sections(id, source_id)" in sql


def test_schema_does_not_globally_unique_source_hash():
    sql = Path("db/init.sql").read_text()
    assert "hash_sha256  TEXT NOT NULL UNIQUE" not in sql
    assert "CREATE INDEX idx_sources_hash" in sql
