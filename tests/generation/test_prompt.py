from app.generation.prompt import build_llm_prompt, llm_response_json_schema


def test_prompt_does_not_ask_llm_for_confidence():
    prompt = build_llm_prompt(context_chunks=[], question="Питання?")
    assert "confidence" not in prompt
    assert "Відповідай ТІЛЬКИ українською" in prompt


def test_llm_schema_contains_internal_generation_fields_only():
    schema = llm_response_json_schema()
    props = schema["properties"]
    assert set(props) == {"answer", "fallback_reason", "used_sections"}


def test_prompt_injects_rule_chunks_separately_from_sources():
    prompt = build_llm_prompt(
        context_chunks=[{"section": "1. Щорічна відпустка", "content": "Працівник може використати відпустку після 6 місяців"}],
        question="Питання?",
        rule_chunks=[{"section": "10. Правила", "content": "Відповідай українською"}],
    )
    assert "ПРАВИЛА З БАЗИ ЗНАНЬ" in prompt
    assert "Відповідай українською" in prompt
    assert "ДЖЕРЕЛЬНИЙ КОНТЕКСТ" in prompt

