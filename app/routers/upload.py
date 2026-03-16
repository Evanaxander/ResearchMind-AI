from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.models.schemas import UploadResponse, DocumentMetadata
from app.services.document_service import DocumentService
from app.core.config import settings

router = APIRouter()


def get_document_service() -> DocumentService:
    """Dependency injection — swap for a mock in tests."""
    return DocumentService()


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
):
    """
    Upload a PDF, TXT, or DOCX file.

    - Validates file type and size
    - Saves to disk (Phase 2 will also index it into FAISS)
    - Returns document metadata including a stable doc_id
    """
    # Validate content type
    allowed_types = {"application/pdf", "text/plain",
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, TXT, DOCX."
        )

    # Read and validate size
    contents = await file.read()
    size_bytes = len(contents)
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_bytes / 1e6:.1f} MB). Max allowed: {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    # Delegate to service layer
    doc_meta = await service.save_document(
        filename=file.filename,
        contents=contents,
        content_type=file.content_type,
        size_bytes=size_bytes,
    )

    return UploadResponse(
        success=True,
        message=f"'{file.filename}' uploaded successfully.",
        document=doc_meta,
    )


@router.get("/documents", response_model=list[DocumentMetadata])
async def list_documents(
    service: DocumentService = Depends(get_document_service),
):
    """List all uploaded documents."""
    return await service.list_documents()


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(
    doc_id: str,
    service: DocumentService = Depends(get_document_service),
):
    """Delete a document and its FAISS index (Phase 2)."""
    deleted = await service.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")