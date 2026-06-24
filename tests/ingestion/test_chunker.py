from pathlib import Path

from app.ingestion.parser import parse_markdown_sections
from app.ingestion.chunker import chunk_sections


def test_chunker_preserves_section_title_and_marks_meta_rules():
    sections = parse_markdown_sections(Path("data/knowledge_base.md").read_text())
    chunks = chunk_sections(sections)
    vacation_chunks = [chunk for chunk in chunks if chunk.section_num == 1]
    rule_chunks = [chunk for chunk in chunks if chunk.is_rule]

    assert vacation_chunks
    assert vacation_chunks[0].section_title == "1. Щорічна відпустка"
    assert all(chunk.section_num in {9, 10} for chunk in rule_chunks)
