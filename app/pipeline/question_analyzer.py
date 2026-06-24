from __future__ import annotations

from dataclasses import dataclass
import re


CALCULATION_TERMS = ("порахуй", "розрахуй", "обчисли", "скільки", "сума")
NUMERIC_SIGNALS = ("грн", "%", "відсот")


@dataclass(frozen=True)
class QuestionAnalysis:
    normalized_question: str
    requires_calculation: bool
    matched_signals: tuple[str, ...]


def analyze_question(question: str) -> QuestionAnalysis:
    normalized = " ".join(question.strip().lower().split())
    matched: list[str] = []
    if any(term in normalized for term in CALCULATION_TERMS):
        matched.append("calculation_keyword")
    if re.search(r"\d", normalized) or any(signal in normalized for signal in NUMERIC_SIGNALS):
        matched.append("numeric_signal")
    requires_calculation = "calculation_keyword" in matched and "numeric_signal" in matched
    return QuestionAnalysis(
        normalized_question=normalized,
        requires_calculation=requires_calculation,
        matched_signals=tuple(matched),
    )
