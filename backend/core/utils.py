"""
Shared utility functions for CompliVibe backend.
"""

import re


def slugify(text: str) -> str:
    """Convert a string to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split *text* into overlapping chunks of *chunk_size* characters.

    Args:
        text: The source text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
