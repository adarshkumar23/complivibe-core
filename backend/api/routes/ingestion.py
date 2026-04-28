"""
Ingestion API routes.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.ingestion.pipeline import RegulatoryIngestionPipeline

router = APIRouter()


class IngestResponse(BaseModel):
    source: str
    chunk_count: int


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(source: str) -> IngestResponse:
    """
    Ingest a local regulatory document and return the number of chunks produced.

    Args:
        source: Path to the local file.
    """
    try:
        pipeline = RegulatoryIngestionPipeline()
        chunks = pipeline.run(source)
        return IngestResponse(source=source, chunk_count=len(chunks))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
