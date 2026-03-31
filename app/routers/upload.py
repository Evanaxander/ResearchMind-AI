from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from app.models.schemas import UploadResponse, DocumentMetadata
from app.services.document_service import DocumentService
from app.core.config import settings

router = APIRouter()


class FullUploadResponse(BaseModel):
    """Extended upload response including financial analysis."""
    success:              bool
    message:              str
    document:             DocumentMetadata
    financial_summary:    Optional[str]  = None
    contradictions_found: int            = 0
    contradictions:       List[dict]     = []


def get_document_service() -> DocumentService:
    return DocumentService()


@router.post("/upload", response_model=FullUploadResponse, status_code=201)
async def upload_document(
    file:    UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
):
    """
    Upload a financial document (PDF, TXT, DOCX).

    FinanceIQ automatically:
    - Detects document type (10-K, 10-Q, earnings call, analyst report)
    - Extracts ticker symbol and fiscal period
    - Preserves table structure during chunking
    - Extracts key financial metrics (revenue, EPS, margins, guidance)
    - Adds document to the knowledge graph
    - Detects contradictions with existing related documents
    - Returns a full financial summary immediately
    """
    allowed_types = {
        "application/pdf",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}."
        )

    contents   = await file.read()
    size_bytes = len(contents)
    max_bytes  = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_bytes / 1e6:.1f} MB). "
                   f"Max: {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    doc_meta, financial_summary, contradictions = await service.save_document(
        filename     = file.filename,
        contents     = contents,
        content_type = file.content_type,
        size_bytes   = size_bytes,
    )

    message = f"'{file.filename}' uploaded and analyzed successfully."
    if contradictions:
        message += (
            f" WARNING: {len(contradictions)} contradiction(s) detected "
            f"with existing documents."
        )

    return FullUploadResponse(
        success              = True,
        message              = message,
        document             = doc_meta,
        financial_summary    = financial_summary,
        contradictions_found = len(contradictions),
        contradictions       = contradictions,
    )


@router.get("/documents", response_model=list[DocumentMetadata])
async def list_documents(
    service: DocumentService = Depends(get_document_service),
):
    return await service.list_documents()


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(
    doc_id:  str,
    service: DocumentService = Depends(get_document_service),
):
    deleted = await service.delete_document(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Document {doc_id} not found."
        )