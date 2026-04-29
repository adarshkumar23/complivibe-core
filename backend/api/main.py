from __future__ import annotations
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routers import api_router

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("CompliVibe API starting")
    yield
    logger.info("CompliVibe API shutting down")

app = FastAPI(title="CompliVibe API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
