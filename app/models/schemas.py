from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid


# ── Document models ──────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Returned after a successful file upload."""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    size_bytes: int
    content_type: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    chunk_count: Optional[int] = None  # filled in Phase 2 after indexing


class UploadResponse(BaseModel):
    success: bool
    message: str
    document: DocumentMetadata


# ── Query models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Body of a /query request."""
    question: str = Field(..., min_length=3, max_length=2000,
                          example="What are the main findings in the uploaded paper?")
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Limit retrieval to specific document IDs. None = search all."
    )
    top_k: int = Field(default=5, ge=1, le=20,
                       description="Number of chunks to retrieve from FAISS.")


class SourceChunk(BaseModel):
    """A retrieved context chunk returned with the answer."""
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    score: float  # cosine similarity score


class QueryResponse(BaseModel):
    """Full response from the research agent."""
    question: str
    answer: str
    sources: List[SourceChunk] = []
    agent_steps: List[str] = []       # LangGraph trace — filled in Phase 3
    latency_ms: Optional[float] = None


# ── Health model ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str