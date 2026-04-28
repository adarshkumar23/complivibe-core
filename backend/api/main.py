"""
FastAPI application entry point for CompliVibe.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import classifier, ingestion, mapper

app = FastAPI(
    title="CompliVibe API",
    description=(
        "Regulatory compliance intelligence for the EU AI Act and India DPDP Act."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["ingestion"])
app.include_router(classifier.router, prefix="/api/v1/classifier", tags=["classifier"])
app.include_router(mapper.router, prefix="/api/v1/mapper", tags=["mapper"])


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok", "service": "complivibe-api"}
