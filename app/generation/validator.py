from __future__ import annotations

from app.models import Confidence


INSUFFICIENCY_SIGNALS = (
    "недостатньо даних",
    "недостатньо інформації",
    "не містить",
    "відсутні",
    "cannot be completed",
    "does not contain",
)


def detect_llm_insufficiency(answer: str) -> bool:
    normalized = answer.casefold()
    return any(signal in normalized for signal in INSUFFICIENCY_SIGNALS)


def compute_confidence(
    *,
    top_rrf_score: float,
    supporting_chunks: int,
    vector_lexical_overlap: float,
    has_disclaimer: bool,
    llm_signals_insufficiency: bool,
) -> Confidence:
    if supporting_chunks == 0 or llm_signals_insufficiency:
        return "low"

    score = 0
    if top_rrf_score >= 0.03:
        score += 2
    elif top_rrf_score >= 0.015:
        score += 1
    if supporting_chunks >= 3:
        score += 2
    elif supporting_chunks >= 1:
        score += 1
    if vector_lexical_overlap >= 0.5:
        score += 1
    if has_disclaimer:
        score -= 1

    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def decide_fallback_reason(confidence: Confidence, explicit_reason: str | None = None) -> str | None:
    if explicit_reason:
        return explicit_reason
    if confidence == "low":
        return "У базі знань недостатньо даних для точної відповіді."
    return None
