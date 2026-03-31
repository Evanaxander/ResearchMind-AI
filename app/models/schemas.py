from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid


# ── Document models ───────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Returned after a successful file upload."""
    doc_id:        str      = Field(default_factory=lambda: str(uuid.uuid4()))
    filename:      str
    size_bytes:    int
    content_type:  str
    uploaded_at:   datetime = Field(default_factory=datetime.utcnow)
    chunk_count:   Optional[int]  = None

    # ── Financial metadata (new in FinanceIQ) ─────────────────────────────
    doc_type:      Optional[str]  = None   # 10-K, 10-Q, earnings, analyst, general
    ticker:        Optional[str]  = None   # detected stock ticker e.g. AAPL
    fiscal_period: Optional[str]  = None   # e.g. FY2023, Q3 2024
    metrics_found: Optional[List[str]] = None  # financial metrics detected
    has_tables:    bool            = False  # whether document contains tables
    extraction_confidence: Optional[float] = None  # metric extraction quality


class UploadResponse(BaseModel):
    success:  bool
    message:  str
    document: DocumentMetadata

    # ── Financial summary (shown immediately after upload) ────────────────
    financial_summary: Optional[str] = None  # extracted metrics summary


# ── Query models ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Body of a /query request."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Your financial research question"
    )
    doc_ids:       Optional[List[str]] = Field(
        default=None,
        description="Limit search to specific documents. None = search all."
    )
    top_k:         int = Field(default=5, ge=1, le=20)

    # ── Financial filters (new in FinanceIQ) ──────────────────────────────
    ticker_filter: Optional[str] = Field(
        default=None,
        description="Filter by ticker symbol e.g. AAPL"
    )
    period_filter: Optional[str] = Field(
        default=None,
        description="Filter by fiscal period e.g. FY2023"
    )
    doc_type_filter: Optional[str] = Field(
        default=None,
        description="Filter by document type: 10-K, 10-Q, earnings, analyst"
    )


class SourceChunk(BaseModel):
    """A retrieved context chunk returned with the answer."""
    doc_id:      str
    filename:    str
    chunk_index: int
    text:        str
    score:       float

    # ── Financial metadata on each source ────────────────────────────────
    doc_type:      Optional[str] = None
    ticker:        Optional[str] = None
    fiscal_period: Optional[str] = None
    has_tables:    Optional[bool] = None


class QueryResponse(BaseModel):
    """Full response from the financial research agent."""
    question:    str
    answer:      str
    query_type:  Optional[str]       = None   # metric, comparison, risk, general
    sources:     List[SourceChunk]   = []
    agent_steps: List[str]           = []
    latency_ms:  Optional[float]     = None

    # ── Financial analysis summary ────────────────────────────────────────
    analysis:    Optional[str]       = None   # analyst agent findings


# ── Financial document summary model ─────────────────────────────────────────

class FinancialDocumentSummary(BaseModel):
    """
    Structured financial summary returned when listing documents.
    Shows extracted metrics at a glance.
    """
    doc_id:        str
    filename:      str
    doc_type:      Optional[str]  = None
    ticker:        Optional[str]  = None
    fiscal_period: Optional[str]  = None
    revenue:       Optional[str]  = None
    net_income:    Optional[str]  = None
    eps:           Optional[str]  = None
    top_risks:     List[str]      = []
    uploaded_at:   Optional[datetime] = None


# ── Health model ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    environment: str
    version:     str