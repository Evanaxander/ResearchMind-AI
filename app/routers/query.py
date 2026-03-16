from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.query_service import QueryService
import time

router = APIRouter()
query_service = QueryService()


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question over your uploaded documents.

    Phase 1: Returns a stub answer (wires up real RAG + agents in Phases 2 & 3).
    The response shape is final — front-end can integrate against this now.
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    start = time.time()
    response = await query_service.answer(request)
    response.latency_ms = round((time.time() - start) * 1000, 2)

    return response