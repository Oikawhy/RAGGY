from __future__ import annotations

from dataclasses import dataclass
import re


SECTION_HEADING_RE = re.compile(r"^##\s+(?P<num>\d+)\.\s+(?P<title>.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    section_num: int
    title: str
    raw_content: str
    language: str
    is_meta: bool = False

    @property
    def display_title(self) -> str:
        return f"{self.section_num}. {self.title}"


def detect_language(text: str) -> str:
    cyrillic = len(re.findall(r"[А-Яа-яІіЇїЄєҐґ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cyrillic and latin:
        return "mixed"
    if latin and not cyrillic:
        return "en"
    return "uk"


def parse_markdown_sections(text: str) -> list[Section]:
    matches = list(SECTION_HEADING_RE.finditer(text))
    sections: list[Section] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_content = text[start:end].strip()
        section_num = int(match.group("num"))
        title = match.group("title").strip()
        sections.append(
            Section(
                section_num=section_num,
                title=title,
                raw_content=raw_content,
                language=detect_language(f"{title}\n{raw_content}"),
                is_meta=section_num in {9, 10},
            )
        )
    return sections
