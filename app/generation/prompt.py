from __future__ import annotations

from typing import Any


def build_llm_prompt(context_chunks: list[Any], question: str, rule_chunks: list[Any] | None = None) -> str:
    # Build rule section from meta/rule chunks
    rules = "\n\n".join(
        f"- {getattr(chunk, 'content', None) or chunk.get('content', '')}"
        for chunk in (rule_chunks or [])
    ) or "Немає додаткових правил."

    # Build source context section
    context_lines: list[str] = []
    for index, chunk in enumerate(context_chunks, start=1):
        section = getattr(chunk, "section", None) or chunk.get("section", "")
        content = getattr(chunk, "content", None) or chunk.get("content", "")
        context_lines.append(f"[{index}] {section}\n{content}")
    context = "\n\n".join(context_lines) if context_lines else "Контекст не знайдено."

    return (
        "Відповідай ТІЛЬКИ українською мовою.\n"
        "Ти — AI-консультант з кадрових питань. Давай чіткі, прямі відповіді.\n\n"
        "ГОЛОВНІ ПРАВИЛА:\n"
        "1. Якщо контекст містить чітке правило — дай пряму відповідь (так/ні), потім поясни з посиланням на джерело.\n"
        "2. Застосуй правило до конкретної ситуації користувача. Наприклад, якщо правило каже '6 місяців', а користувач працює 3 — скажи прямо, що ще не може.\n"
        "3. Не вигадуй дати, суми, формули, винятки або юридичні норми, яких немає в контексті.\n"
        "4. Якщо даних у контексті реально недостатньо для відповіді, прямо скажи про це та заповни fallback_reason.\n"
        "5. НЕ будь надмірно обережним: якщо контекст дає достатньо інформації, відповідай впевнено.\n\n"
        f"ПРАВИЛА З БАЗИ ЗНАНЬ:\n{rules}\n\n"
        f"ДЖЕРЕЛЬНИЙ КОНТЕКСТ:\n{context}\n\n"
        f"Питання користувача:\n{question}\n\n"
        "Поверни JSON з полями answer, fallback_reason, used_sections."
    )


def llm_response_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "fallback_reason", "used_sections"],
        "properties": {
            "answer": {"type": "string"},
            "fallback_reason": {"type": ["string", "null"]},
            "used_sections": {"type": "array", "items": {"type": "string"}},
        },
    }
