from __future__ import annotations

import json
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from backend.core.config import Config
from backend.ingestion.chunker import RegulationChunk, chunk_regulation_pdf, load_chunks, save_chunks
from backend.ingestion.embedder import RegulationEmbedder
from backend.ingestion.fetcher import fetch_regulation_pdfs


@pytest.fixture
def temp_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        yield Config(
            anthropic_api_key="test_key",
            chroma_persist_dir=str(temp_dir / "chromadb"),
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            mappings_dir=temp_dir / "mappings",
        )


@pytest.fixture
def sample_pdf_bytes():
    pdf_content = b"%PDF-1.4\n%test\nThis is a test PDF.\nArticle 1\nContent here."
    return pdf_content


@pytest.fixture
def sample_regulation_text():
    return """
    Article 1 - Scope and Objectives
    This Regulation lays down harmonised rules on artificial intelligence.
    
    Article 2 - Definitions
    For the purposes of this Regulation, the following definitions apply.
    
    Article 3 - Prohibited Practices
    Certain practices shall be prohibited.
    
    Article 4 - High-Risk Systems
    High-risk AI systems shall comply with the requirements set out in Chapter 2.
    """


@pytest.fixture
def sample_dpdp_text():
    return """
    Section 1 - Short title and commencement
    This Act may be called the Digital Personal Data Protection Act, 2023.
    
    Section 2 - Definitions
    In this Act, unless the context otherwise requires.
    
    Section 3 - Data Principal
    A Data Principal is a natural person to whom personal data relates.
    
    Section 4 - Processing of Personal Data
    Personal data shall be processed in accordance with this Act.
    """


class TestFetcher:
    @patch("backend.ingestion.fetcher.httpx.stream")
    def test_fetch_regulation_downloads_pdfs(self, mock_stream, temp_config, sample_pdf_bytes):
        with patch("backend.ingestion.fetcher.config", temp_config):
            mock_response = MagicMock()
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"Content-Length": str(len(sample_pdf_bytes))}
            mock_response.iter_bytes = Mock(return_value=[sample_pdf_bytes])
            mock_stream.return_value = mock_response

            temp_config.raw_data_dir.mkdir(parents=True, exist_ok=True)
            results = fetch_regulation_pdfs(force_redownload=True)

            assert "eu_ai_act" in results
            assert "dpdp" in results
            assert results["eu_ai_act"].exists()
            assert results["dpdp"].exists()

    @patch("backend.ingestion.fetcher.httpx.stream")
    def test_fetch_skips_cached_pdfs(self, mock_stream, temp_config, sample_pdf_bytes):
        with patch("backend.ingestion.fetcher.config", temp_config):
            temp_config.raw_data_dir.mkdir(parents=True, exist_ok=True)
            
            eu_ai_act_path = temp_config.raw_data_dir / "eu_ai_act.pdf"
            eu_ai_act_path.write_bytes(sample_pdf_bytes)

            results = fetch_regulation_pdfs(force_redownload=False)

            assert "eu_ai_act" in results
            assert results["eu_ai_act"] == eu_ai_act_path
            mock_stream.assert_called()

    def test_fetch_validates_pdf_header(self, temp_config):
        with patch("backend.ingestion.fetcher.config", temp_config):
            temp_config.raw_data_dir.mkdir(parents=True, exist_ok=True)
            
            invalid_path = temp_config.raw_data_dir / "invalid.pdf"
            invalid_path.write_bytes(b"not a real pdf")

            is_valid = invalid_path.exists() and invalid_path.open("rb").read(5) == b"%PDF-"
            assert not is_valid


class TestChunker:
    @patch("backend.ingestion.chunker.PdfReader")
    def test_chunk_regulation_produces_chunks(self, mock_reader, sample_regulation_text, temp_config):
        with patch("backend.ingestion.chunker.config", temp_config):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = sample_regulation_text
            mock_reader.return_value.pages = [mock_page] * 3

            pdf_path = temp_config.raw_data_dir / "test.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\ntest")

            chunks = chunk_regulation_pdf("eu_ai_act", pdf_path)

            assert len(chunks) > 0
            assert all(isinstance(chunk, RegulationChunk) for chunk in chunks)

    @patch("backend.ingestion.chunker.PdfReader")
    def test_chunk_metadata_populated(self, mock_reader, sample_regulation_text, temp_config):
        with patch("backend.ingestion.chunker.config", temp_config):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = sample_regulation_text
            mock_reader.return_value.pages = [mock_page]

            pdf_path = temp_config.raw_data_dir / "test.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\ntest")

            chunks = chunk_regulation_pdf("eu_ai_act", pdf_path)

            for chunk in chunks:
                assert chunk.chunk_id is not None
                assert chunk.regulation == "eu_ai_act"
                assert chunk.regulation_display == "EU AI Act"
                assert chunk.text is not None
                assert chunk.page_start > 0
                assert chunk.char_start >= 0
                assert chunk.char_end > chunk.char_start
                assert chunk.chunk_index > 0
                assert chunk.total_chunks > 0

    @patch("backend.ingestion.chunker.PdfReader")
    def test_chunk_ids_unique(self, mock_reader, sample_regulation_text, temp_config):
        with patch("backend.ingestion.chunker.config", temp_config):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = sample_regulation_text
            mock_reader.return_value.pages = [mock_page]

            pdf_path = temp_config.raw_data_dir / "test.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\ntest")

            chunks = chunk_regulation_pdf("eu_ai_act", pdf_path)
            chunk_ids = [chunk.chunk_id for chunk in chunks]

            assert len(chunk_ids) == len(set(chunk_ids))

    @patch("backend.ingestion.chunker.PdfReader")
    def test_chunk_id_naming_convention(self, mock_reader, sample_regulation_text, temp_config):
        with patch("backend.ingestion.chunker.config", temp_config):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = sample_regulation_text
            mock_reader.return_value.pages = [mock_page]

            pdf_path = temp_config.raw_data_dir / "test.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\ntest")

            chunks = chunk_regulation_pdf("eu_ai_act", pdf_path)

            for chunk in chunks:
                assert chunk.chunk_id.startswith("eu_ai_act_art")
                assert "_chunk_" in chunk.chunk_id

    @patch("backend.ingestion.chunker.PdfReader")
    def test_save_and_load_chunks(self, mock_reader, sample_regulation_text, temp_config):
        with patch("backend.ingestion.chunker.config", temp_config):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = sample_regulation_text
            mock_reader.return_value.pages = [mock_page]

            pdf_path = temp_config.raw_data_dir / "test.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\ntest")

            chunks = chunk_regulation_pdf("eu_ai_act", pdf_path)
            temp_config.processed_data_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_chunks("eu_ai_act", chunks)

            assert output_path.exists()
            loaded = load_chunks(output_path)
            assert len(loaded) == len(chunks)
            assert all(isinstance(chunk, RegulationChunk) for chunk in loaded)


class TestEmbedder:
    @patch("backend.ingestion.embedder.chromadb.PersistentClient")
    def test_embedder_search_returns_results(self, mock_client, temp_config):
        with patch("backend.ingestion.embedder.config", temp_config):
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection

            mock_collection.query.return_value = {
                "ids": [["chunk_1", "chunk_2"]],
                "documents": [["text 1", "text 2"]],
                "metadatas": [
                    [
                        {
                            "chunk_id": "chunk_1",
                            "regulation": "eu_ai_act",
                            "article": "1",
                            "chapter": None,
                            "page_start": 1,
                        },
                        {
                            "chunk_id": "chunk_2",
                            "regulation": "eu_ai_act",
                            "article": "2",
                            "chapter": None,
                            "page_start": 2,
                        },
                    ]
                ],
                "distances": [[0.1, 0.2]],
            }

            embedder = RegulationEmbedder()
            results = embedder.search("test query", top_k=2)

            assert len(results) == 2
            assert "id" in results[0]
            assert "text" in results[0]
            assert "metadata" in results[0]
            assert "distance" in results[0]

    @patch("backend.ingestion.embedder.chromadb.PersistentClient")
    def test_embedder_search_filters_by_regulation(self, mock_client, temp_config):
        with patch("backend.ingestion.embedder.config", temp_config):
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

            embedder = RegulationEmbedder()
            embedder.search("test", regulation="eu_ai_act")

            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args[1]
            assert call_kwargs["where"]["regulation"] == "eu_ai_act"

    @patch("backend.ingestion.embedder.chromadb.PersistentClient")
    def test_embedder_get_stats(self, mock_client, temp_config):
        with patch("backend.ingestion.embedder.config", temp_config):
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            mock_collection.get.return_value = {
                "metadatas": [
                    {"regulation": "eu_ai_act"},
                    {"regulation": "eu_ai_act"},
                    {"regulation": "dpdp"},
                ]
            }

            embedder = RegulationEmbedder()
            stats = embedder.get_stats()

            assert stats["eu_ai_act"] == 2
            assert stats["dpdp"] == 1


class TestPipeline:
    @patch("backend.ingestion.pipeline.embedder.RegulationEmbedder")
    @patch("backend.ingestion.pipeline.fetch_regulation_pdfs")
    @patch("backend.ingestion.pipeline.chunk_regulation_pdf")
    @patch("backend.ingestion.pipeline.save_chunks")
    @patch("backend.ingestion.pipeline.load_chunks")
    def test_pipeline_runs_end_to_end(
        self,
        mock_load_chunks,
        mock_save_chunks,
        mock_chunk_pdf,
        mock_fetch,
        mock_embedder_class,
        temp_config,
    ):
        with patch("backend.ingestion.pipeline.config", temp_config):
            temp_config.raw_data_dir.mkdir(parents=True, exist_ok=True)
            temp_config.processed_data_dir.mkdir(parents=True, exist_ok=True)

            eu_ai_act_path = temp_config.raw_data_dir / "eu_ai_act.pdf"
            dpdp_path = temp_config.raw_data_dir / "dpdp.pdf"
            eu_ai_act_path.write_bytes(b"%PDF-test")
            dpdp_path.write_bytes(b"%PDF-test")

            mock_fetch.return_value = {
                "eu_ai_act": eu_ai_act_path,
                "dpdp": dpdp_path,
            }

            chunk1 = RegulationChunk(
                chunk_id="eu_ai_act_art1_chunk_0001",
                regulation="eu_ai_act",
                regulation_display="EU AI Act",
                text="Test chunk",
                article="1",
                chapter=None,
                page_start=1,
                char_start=0,
                char_end=10,
                chunk_index=1,
                total_chunks=1,
            )

            mock_chunk_pdf.return_value = [chunk1]
            mock_save_chunks.return_value = temp_config.processed_data_dir / "chunks.json"
            mock_load_chunks.return_value = [chunk1]

            mock_embedder_instance = MagicMock()
            mock_embedder_instance.get_stats.return_value = {"eu_ai_act": 1, "dpdp": 0}
            mock_embedder_class.return_value = mock_embedder_instance

            from backend.ingestion.pipeline import run

            run(force_redownload=False)

            mock_fetch.assert_called_once()
            assert mock_chunk_pdf.call_count >= 1
            mock_embedder_instance.embed_chunks.assert_called()
