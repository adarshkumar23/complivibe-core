from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer


class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PATH: str = "data/chromadb"
    CHROMA_EU_COLLECTION: str = "regulations"
    CHROMA_DPDP_COLLECTION: str = "regulations"
    EU_OBLIGATIONS_PATH: str = "data/processed/eu_ai_act_obligations.json"
    DPDP_OBLIGATIONS_PATH: str = "data/processed/dpdp_obligations.json"
    MAPPINGS_PATH: str = "data/mappings/cross_mappings.json"
    SECRET_KEY: str = "dev"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_chroma_client() -> chromadb.PersistentClient:
    settings = get_settings()
    return chromadb.PersistentClient(
        path=settings.CHROMA_PATH,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


@lru_cache
def get_eu_collection():
    settings = get_settings()
    client = get_chroma_client()
    return client.get_or_create_collection(settings.CHROMA_EU_COLLECTION)


@lru_cache
def get_dpdp_collection():
    settings = get_settings()
    client = get_chroma_client()
    return client.get_or_create_collection(settings.CHROMA_DPDP_COLLECTION)


@lru_cache
def get_embedder() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def _load_json(path: str) -> list[dict[str, Any]]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"JSON data not found: {data_path}")
    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {data_path}")
    return payload


@lru_cache
def get_eu_obligations() -> list[dict[str, Any]]:
    settings = get_settings()
    return _load_json(settings.EU_OBLIGATIONS_PATH)


@lru_cache
def get_dpdp_obligations() -> list[dict[str, Any]]:
    settings = get_settings()
    return _load_json(settings.DPDP_OBLIGATIONS_PATH)


@lru_cache
def get_cross_mappings() -> list[dict[str, Any]]:
    settings = get_settings()
    return _load_json(settings.MAPPINGS_PATH)
