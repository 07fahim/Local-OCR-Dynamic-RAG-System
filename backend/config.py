"""
Central configuration loaded from environment / .env.

Every tunable lives here so the rest of the codebase never reads os.environ
directly. Import `settings` and use its attributes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # reads .env at process start; no-op if the file is absent


def _split_langs(raw: str) -> list[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


@dataclass(frozen=True)
class Settings:
    # OCR (Surya). Languages recognised per page, as ISO codes.
    ocr_langs: list[str] = field(default_factory=lambda: _split_langs(os.getenv("OCR_LANGS", "bn,en")))

    # Embedding (BGE-M3, local)
    embedding_model: str = "BAAI/bge-m3"

    # Answer generation (fully local)
    llm_mode: str = os.getenv("LLM_MODE", "local").lower()  # local | none
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")  # 3b is a lighter fallback

    # Storage
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma_db")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./data/ingestion.db")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")

    # Retrieval
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))


settings = Settings()
