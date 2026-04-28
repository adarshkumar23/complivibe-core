"""
Tests for backend.core.utils
"""

import pytest

from backend.core.utils import chunk_text, slugify


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_characters(self):
        assert slugify("EU AI Act (2024)") == "eu-ai-act-2024"

    def test_leading_trailing_hyphens(self):
        assert slugify("  --hello--  ") == "hello"

    def test_empty_string(self):
        assert slugify("") == ""


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("hello", chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == "hello"

    def test_exact_chunk_size(self):
        text = "a" * 512
        chunks = chunk_text(text, chunk_size=512, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_chunks_no_overlap(self):
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=40, overlap=0)
        assert len(chunks) == 3
        assert chunks[0] == "a" * 40
        assert chunks[1] == "a" * 40
        assert chunks[2] == "a" * 20

    def test_multiple_chunks_with_overlap(self):
        text = "abcdefghij"  # 10 chars
        chunks = chunk_text(text, chunk_size=6, overlap=2)
        # step = 6 - 2 = 4
        # starts: 0, 4, 8
        assert chunks[0] == "abcdef"
        assert chunks[1] == "efghij"
        assert chunks[2] == "ij"

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("text", chunk_size=0)

    def test_negative_overlap(self):
        with pytest.raises(ValueError):
            chunk_text("text", chunk_size=10, overlap=-1)

    def test_overlap_ge_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("text", chunk_size=5, overlap=5)
