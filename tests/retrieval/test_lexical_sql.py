from app.retrieval.lexical import BM25LexicalIndex, build_fts_sql
from app.ingestion.chunker import chunk_sections
from app.ingestion.parser import parse_markdown_sections


def test_fts_sql_uses_active_corpus_manifest():
    sql = build_fts_sql()
    assert "JOIN active_corpus_sources" in sql
    assert "plainto_tsquery('simple'" in sql
    assert "c.is_rule = FALSE" in sql


def test_bm25_index_prefers_exact_hr_term():
    index = BM25LexicalIndex.from_chunks([
        {"chunk_id": "a", "content": "ЄСВ сплачується у строки звітного періоду"},
        {"chunk_id": "b", "content": "Працівник може використати щорічну відпустку"},
    ])
    results = index.search("Яка дата сплати ЄСВ?", top_k=1)
    assert results[0].chunk_id == "a"


def test_bm25_expands_ukrainian_sick_leave_question_to_english_policy_terms():
    index = BM25LexicalIndex.from_chunks([
        {
            "chunk_id": "sick",
            "section": "2. Sick leave policy",
            "content": (
                "Sick leave must be confirmed by an official medical certificate. "
                "If the medical certificate is missing, the absence cannot be automatically treated as paid sick leave."
            ),
        },
        {
            "chunk_id": "documents",
            "section": "5. Документи при прийнятті на роботу",
            "content": "Працівник має надати роботодавцю документи, необхідні для оформлення трудових відносин.",
        },
    ])

    results = index.search(
        "Працівник захворів, але ще не надав медичний документ. Чи можна одразу оплатити лікарняний?",
        top_k=2,
    )

    assert results[0].chunk_id == "sick"


def test_assignment_bm25_finds_sick_leave_section_for_ukrainian_question():
    sections = parse_markdown_sections(open("data/knowledge_base.md", encoding="utf-8").read())
    chunks = chunk_sections(sections)
    index = BM25LexicalIndex.from_chunks([
        {
            "chunk_id": chunk.chunk_id,
            "section": chunk.section_title,
            "content": chunk.content,
            "section_order": chunk.section_num,
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
        }
        for chunk in chunks
    ])

    results = index.search(
        "Працівник захворів, але ще не надав медичний документ. Чи можна одразу оплатити лікарняний?",
        top_k=3,
    )

    assert results[0].section == "2. Sick leave policy"
