from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # General
    APP_NAME: str = "ResearchMind"
    ENVIRONMENT: str = "development"
    ANALYSIS_DOMAIN: str = "general"  # general | finance
    ENABLE_FINANCIAL_ENRICHMENT: bool = False
    FAST_QUERY_MODE: bool = False
    QUERY_CACHE_TTL_SECONDS: int = 180
    QUERY_MAX_SUBTASKS: int = 4
    FAST_TOP_K_CAP: int = 3
    ANALYST_CONTEXT_CHUNKS: int = 6
    SYNTH_CONTEXT_CHUNKS: int = 4

    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_PASSWORD: str = "neo4j"

    # API Keys (loaded from .env)
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # Storage
    UPLOAD_DIR: str = "./uploads"
    FAISS_INDEX_DIR: str = "./faiss_index"
    MAX_UPLOAD_SIZE_MB: int = 20

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Chunking defaults (used in Phase 2)
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()