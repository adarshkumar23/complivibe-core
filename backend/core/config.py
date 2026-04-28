"""
Core configuration for CompliVibe.

Reads settings from environment variables (see .env.example).
"""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Application
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "CompliVibe"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./complivibe.db")
    )

    # Vector store (Qdrant)
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))

    # OpenAI / embedding model
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Ingestion
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64")))


settings = Settings()
