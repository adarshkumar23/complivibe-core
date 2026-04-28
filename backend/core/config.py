from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    llm_provider: str = "openai"
    llm_model_extraction: str = "gpt-4o-mini"
    llm_model_mapping: str = "gpt-4.1"

    claude_model: str = "claude-sonnet-4-20250514"

    chroma_persist_dir: str = "./data/chromadb"
    chroma_collection_regulations: str = "regulations"

    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    raw_data_dir: Path = Path("./data/raw")
    processed_data_dir: Path = Path("./data/processed")
    mappings_dir: Path = Path("./data/mappings")

    eu_ai_act_url: str = (
        "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689"
    )
    dpdp_url: str = "https://www.meity.gov.in/writereaddata/files/DPDP_Act_2023.pdf"

    @staticmethod
    def from_env() -> "Config":
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if not openai_api_key and not gemini_api_key:
            print("⚠ No OPENAI_API_KEY or GEMINI_API_KEY set")

        return Config(
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_model_extraction=os.getenv("LLM_MODEL_EXTRACTION", "gpt-4o-mini"),
            llm_model_mapping=os.getenv("LLM_MODEL_MAPPING", "gpt-4.1"),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb"),
            chroma_collection_regulations=os.getenv(
                "CHROMA_COLLECTION_REGULATIONS", "regulations"
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            raw_data_dir=Path(os.getenv("RAW_DATA_DIR", "./data/raw")),
            processed_data_dir=Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed")),
            mappings_dir=Path(os.getenv("MAPPINGS_DIR", "./data/mappings")),
            eu_ai_act_url=os.getenv(
                "EU_AI_ACT_URL",
                "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689",
            ),
            dpdp_url=os.getenv(
                "DPDP_URL",
                "https://www.meity.gov.in/writereaddata/files/DPDP_Act_2023.pdf",
            ),
        )

    def get_llm_client(self) -> OpenAI:
        if self.llm_provider == "openai":
            return OpenAI(api_key=self.openai_api_key or None)
        raise NotImplementedError(f"LLM provider '{self.llm_provider}' is not supported")


config = Config.from_env()
