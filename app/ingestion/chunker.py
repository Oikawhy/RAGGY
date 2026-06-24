from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from app.ingestion.parser import Section, detect_language


DISCLAIMER_SIGNALS = (
    "не описані",
    "does not contain",
    "не наведено",
    "немає достатніх",
    "не має вигадувати",
    "відсутня",
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section_num: int
    section_title: str
    chunk_index: int
    content: str
    content_hash: str
    content_tokens: int
    language: str
    block_type: str = "paragraph"
    has_numbers: bool = False
    has_disclaimer: bool = False
    is_rule: bool = False


def _chunk_language(text: str, section_language: str) -> str:
    language = detect_language(text)
    if language == "mixed":
        cyrillic = len(re.findall(r"[А-Яа-яІіЇїЄєҐґ]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        return "uk" if cyrillic >= latin else "en"
    if language in {"uk", "en"}:
        return language
    return "uk" if section_language != "en" else "en"


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))

def _split_long_chunk(text: str, max_tokens: int = 512, overlap_ratio: float = 0.20) -> list[str]:
    """Sliding window split for chunks exceeding max_tokens (Architecture step 6)."""
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    if len(tokens) <= max_tokens:
        return [text]
    overlap = max(1, int(max_tokens * overlap_ratio))
    step = max_tokens - overlap
    windows: list[str] = []
    for i in range(0, len(tokens), step):
        window_tokens = tokens[i : i + max_tokens]
        windows.append(" ".join(window_tokens))
        if i + max_tokens >= len(tokens):
            break
    return windows


def _merge_short_chunks(paragraphs: list[str], min_tokens: int = 30) -> list[str]:
    """Merge chunks shorter than min_tokens with previous (Architecture step 7)."""
    if not paragraphs:
        return paragraphs
    merged: list[str] = [paragraphs[0]]
    for para in paragraphs[1:]:
        if estimate_tokens(merged[-1]) < min_tokens:
            merged[-1] = merged[-1] + "\n\n" + para
        else:
            merged.append(para)
    # Handle trailing short chunk
    if len(merged) > 1 and estimate_tokens(merged[-1]) < min_tokens:
        merged[-2] = merged[-2] + "\n\n" + merged[-1]
        merged.pop()
    return merged


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section.raw_content) if part.strip()]

        # Architecture step 7: merge short chunks
        paragraphs = _merge_short_chunks(paragraphs)

        expanded: list[str] = []
        for paragraph in paragraphs:
            # Architecture step 6: sliding window for long chunks
            expanded.extend(_split_long_chunk(paragraph))

        for chunk_index, paragraph in enumerate(expanded):
            content_hash = hashlib.sha256(paragraph.encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=f"s{section.section_num}-c{chunk_index}",
                    section_num=section.section_num,
                    section_title=section.display_title,
                    chunk_index=chunk_index,
                    content=paragraph,
                    content_hash=content_hash,
                    content_tokens=estimate_tokens(paragraph),
                    language=_chunk_language(paragraph, section.language),
                    has_numbers=bool(re.search(r"\d", paragraph)),
                    has_disclaimer=any(signal in paragraph.lower() for signal in DISCLAIMER_SIGNALS),
                    is_rule=section.is_meta,
                )
            )
    return chunks
