"""
Regulatory text ingestion pipeline (Week 1).

Responsibilities
----------------
- Download or accept raw regulatory documents (PDF / HTML / plain text).
- Parse and clean the raw text.
- Chunk the text into overlapping windows suitable for embedding.
- Persist chunks to the vector store and/or relational database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from backend.core.utils import chunk_text

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single regulatory document loaded from disk or a URL."""

    source: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A single text chunk derived from a :class:`Document`."""

    document_source: str
    index: int
    text: str
    metadata: dict = field(default_factory=dict)


class RegulatoryIngestionPipeline:
    """
    End-to-end pipeline that loads regulatory documents, chunks them,
    and prepares them for embedding / storage.

    Usage::

        pipeline = RegulatoryIngestionPipeline(chunk_size=512, overlap=64)
        chunks = pipeline.run(Path("data/raw/eu_ai_act.txt"))
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, source: Path | str) -> list[Chunk]:
        """
        Load *source*, split into chunks, and return the chunk list.

        Args:
            source: Path to a local file.

        Returns:
            List of :class:`Chunk` objects ready for embedding.
        """
        document = self._load(source)
        chunks = self._chunk(document)
        logger.info("Ingested %d chunks from %s", len(chunks), document.source)
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, source: Path | str) -> Document:
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Source path is not a file: {path}")
        text = path.read_text(encoding="utf-8")
        text = self._clean(text)
        return Document(source=str(path), text=text)

    @staticmethod
    def _clean(text: str) -> str:
        """Remove excessive whitespace and normalise line endings."""
        import re

        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _chunk(self, document: Document) -> list[Chunk]:
        raw_chunks = chunk_text(document.text, self.chunk_size, self.overlap)
        return [
            Chunk(
                document_source=document.source,
                index=i,
                text=chunk,
                metadata=document.metadata,
            )
            for i, chunk in enumerate(raw_chunks)
        ]
