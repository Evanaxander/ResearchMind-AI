from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # General
    APP_NAME: str = "ResearchMind"
    ENVIRONMENT: str = "development"

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
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Chunking defaults (used in Phase 2)
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()