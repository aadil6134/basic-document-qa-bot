from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT_DIR / "data"
    db_path: Path = ROOT_DIR / "db"
    collection_name: str = os.getenv("CHROMA_COLLECTION", "document_knowledge_base")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    generation_model: str = os.getenv(
        "GEMINI_GENERATION_MODEL",
        "gemini-2.5-flash",
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "4"))


settings = Settings()


def require_api_key() -> str:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your Gemini API key."
        )
    return settings.gemini_api_key
