from pathlib import Path

from app.ingestion.parser import parse_markdown_sections


def test_parser_extracts_numbered_sections_from_kb():
    text = Path("data/knowledge_base.md").read_text()
    sections = parse_markdown_sections(text)
    assert len(sections) == 10
    assert sections[0].section_num == 1
    assert sections[0].title == "Щорічна відпустка"
    assert sections[8].is_meta is True
    assert sections[9].is_meta is True
