from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from backend.core.config import config


EU_AI_ACT_DISPLAY = "EU AI Act"
DPDP_DISPLAY = "DPDP Act"

REGULATION_DISPLAY = {
    "eu_ai_act": EU_AI_ACT_DISPLAY,
    "dpdp": DPDP_DISPLAY,
}

ARTICLE_PATTERNS = {
    "eu_ai_act": re.compile(r"\bArticle\s+(\d+[A-Z]?)\b", re.IGNORECASE),
    "dpdp": re.compile(r"\bSection\s+(\d+[A-Z]?)\b", re.IGNORECASE),
}

CHAPTER_PATTERN = re.compile(r"\bChapter\s+([IVXLC]+|\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RegulationChunk:
    chunk_id: str
    regulation: str
    regulation_display: str
    text: str
    article: Optional[str]
    chapter: Optional[str]
    page_start: int
    char_start: int
    char_end: int
    chunk_index: int
    total_chunks: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict) -> "RegulationChunk":
        return RegulationChunk(**payload)


def chunk_regulation_pdf(regulation: str, pdf_path: Path) -> list[RegulationChunk]:
    if regulation not in REGULATION_DISPLAY:
        raise ValueError(f"Unknown regulation key: {regulation}")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    page_texts = _extract_pages(reader)
    full_text, page_offsets = _combine_pages(page_texts)

    print(f"✓ {regulation}: parsed {len(page_texts)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=[
            "\n\nArticle ",
            "\nArticle ",
            "\n\nSection ",
            "\nSection ",
            "\n\nChapter ",
            "\nChapter ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = splitter.split_text(full_text)
    print(f"✓ {regulation}: created {len(chunks)} chunks")

    article_positions = _find_positions(ARTICLE_PATTERNS[regulation], full_text)
    chapter_positions = _find_positions(CHAPTER_PATTERN, full_text)

    results: list[RegulationChunk] = []
    cursor = 0
    total_chunks = len(chunks)

    for index, chunk_text in enumerate(chunks, start=1):
        start = full_text.find(chunk_text, cursor)
        if start == -1:
            start = cursor
        end = start + len(chunk_text)
        cursor = end

        page_start = _page_for_offset(page_offsets, start)
        article = _latest_label(article_positions, start)
        chapter = _latest_label(chapter_positions, start)

        chunk_id = _build_chunk_id(
            regulation=regulation,
            article=article,
            chunk_index=index,
        )

        results.append(
            RegulationChunk(
                chunk_id=chunk_id,
                regulation=regulation,
                regulation_display=REGULATION_DISPLAY[regulation],
                text=chunk_text,
                article=article,
                chapter=chapter,
                page_start=page_start,
                char_start=start,
                char_end=end,
                chunk_index=index,
                total_chunks=total_chunks,
            )
        )

    print(f"✓ {regulation}: chunk metadata ready")
    return results


def save_chunks(
    regulation: str, chunks: Iterable[RegulationChunk], output_path: Path | None = None
) -> Path:
    if output_path is None:
        config.processed_data_dir.mkdir(parents=True, exist_ok=True)
        output_path = config.processed_data_dir / f"{regulation}_chunks.json"

    payload = [chunk.to_dict() for chunk in chunks]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

    print(f"✓ {regulation}: saved {len(payload)} chunks to {output_path}")
    return output_path


def load_chunks(path: Path) -> list[RegulationChunk]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    chunks = [RegulationChunk.from_dict(item) for item in payload]
    print(f"✓ loaded {len(chunks)} chunks from {path}")
    return chunks


def _extract_pages(reader: PdfReader) -> list[str]:
    page_texts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        page_texts.append(text.strip())
    return page_texts


def _combine_pages(page_texts: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
    offsets: list[tuple[int, int, int]] = []
    chunks: list[str] = []
    cursor = 0

    for page_index, text in enumerate(page_texts, start=1):
        if chunks:
            chunks.append("\n\n")
            cursor += 2
        start = cursor
        chunks.append(text)
        cursor += len(text)
        end = cursor
        offsets.append((page_index, start, end))

    return "".join(chunks), offsets


def _find_positions(pattern: re.Pattern[str], text: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for match in pattern.finditer(text):
        label = match.group(1)
        matches.append((match.start(), label))
    return matches


def _latest_label(matches: list[tuple[int, str]], offset: int) -> Optional[str]:
    latest = None
    for position, label in matches:
        if position > offset:
            break
        latest = label
    return latest


def _page_for_offset(page_offsets: list[tuple[int, int, int]], offset: int) -> int:
    for page_index, start, end in page_offsets:
        if start <= offset < end:
            return page_index
    return page_offsets[-1][0] if page_offsets else 1


def _build_chunk_id(regulation: str, article: Optional[str], chunk_index: int) -> str:
    if regulation == "dpdp":
        prefix = "sec"
    else:
        prefix = "art"

    if article:
        label = re.sub(r"[^0-9A-Za-z]+", "", article).lower()
    else:
        label = "0"

    return f"{regulation}_{prefix}{label}_chunk_{chunk_index:04d}"
