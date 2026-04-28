"""
Tests for the regulatory ingestion pipeline.
"""

import textwrap
from pathlib import Path

import pytest

from backend.ingestion.pipeline import RegulatoryIngestionPipeline


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    text = textwrap.dedent(
        """\
        Article 1 – Subject matter

        This Regulation lays down harmonised rules on artificial intelligence.

        Article 2 – Scope

        This Regulation applies to providers placing on the market or putting
        into service AI systems in the Union, irrespective of whether those
        providers are established within the Union or in a third country.
        """
    )
    p = tmp_path / "eu_ai_act_sample.txt"
    p.write_text(text, encoding="utf-8")
    return p


class TestRegulatoryIngestionPipeline:
    def test_run_returns_chunks(self, sample_file: Path):
        pipeline = RegulatoryIngestionPipeline(chunk_size=100, overlap=10)
        chunks = pipeline.run(sample_file)
        assert len(chunks) > 0

    def test_chunk_index_is_sequential(self, sample_file: Path):
        pipeline = RegulatoryIngestionPipeline(chunk_size=100, overlap=10)
        chunks = pipeline.run(sample_file)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_source_matches_file(self, sample_file: Path):
        pipeline = RegulatoryIngestionPipeline(chunk_size=100, overlap=10)
        chunks = pipeline.run(sample_file)
        for chunk in chunks:
            assert chunk.document_source == str(sample_file)

    def test_missing_file_raises(self, tmp_path: Path):
        pipeline = RegulatoryIngestionPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.run(tmp_path / "nonexistent.txt")

    def test_cleans_extra_whitespace(self, tmp_path: Path):
        dirty = tmp_path / "dirty.txt"
        dirty.write_text("word1   word2\r\n\r\n\r\nword3", encoding="utf-8")
        pipeline = RegulatoryIngestionPipeline(chunk_size=512, overlap=0)
        chunks = pipeline.run(dirty)
        assert "   " not in chunks[0].text
        assert "\r" not in chunks[0].text
