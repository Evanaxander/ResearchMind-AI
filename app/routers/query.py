"""
Query Router — with role-based access and audit logging.

Every query is:
1. Authenticated (JWT token extracted)
2. Processed by the financial agent pipeline
3. Shaped based on the user's role
4. Logged to the audit trail
"""

import time
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models.schemas import QueryRequest, QueryResponse
from app.services.query_service import QueryService
from app.services import auth_service
from app.middleware.audit import audit_logger

router      = APIRouter()
bearer      = HTTPBearer(auto_error=False)
query_svc   = QueryService()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer)
) -> dict:
    if not credentials:
        return {"sub": "anonymous", "role": "analyst"}

    payload = auth_service.decode_token(credentials.credentials)

    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token."
        )
    return payload


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    user:    dict = Depends(get_current_user),
):
    """
    Ask a question over your financial documents.

    The response is automatically shaped based on your role:
    - analyst          → full answer with all sources and analysis
    - portfolio_manager → answer with source citations (no raw text)
    - compliance       → answer with compliance notes added
    - executive        → brief one-paragraph summary only

    Include your JWT token in the Authorization header:
        Authorization: Bearer <your_token>
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    username = user.get("sub", "anonymous")
    role     = user.get("role", "analyst")

    start    = time.time()
    error    = None

    try:
        # Run the full financial agent pipeline
        response = await query_svc.answer(request)
        response.latency_ms = round((time.time() - start) * 1000, 2)

        # Shape the response based on user role
        shaped = auth_service.shape_answer_for_role(
            role        = role,
            answer      = response.answer,
            analysis    = response.analysis or "",
            sources     = [s.model_dump() for s in response.sources],
            agent_steps = response.agent_steps,
        )

        # Apply shaped response
        response.answer      = shaped["answer"]
        response.analysis    = shaped["analysis"]
        response.agent_steps = shaped["agent_steps"]

        # Apply source shaping
        from app.models.schemas import SourceChunk
        response.sources = [
            SourceChunk(**{
                **s,
                "text": shaped["sources"][i].get("text", s["text"])
                        if i < len(shaped["sources"]) else s["text"]
            })
            for i, s in enumerate(
                [src.model_dump() for src in response.sources]
            )
        ][:len(shaped["sources"])]

        # Log to audit trail
        audit_logger.log_query(
            username      = username,
            role          = role,
            question      = request.question,
            doc_ids       = request.doc_ids,
            query_type    = response.query_type,
            sources_count = len(response.sources),
            latency_ms    = response.latency_ms,
            success       = True,
        )

        return response

    except Exception as e:
        error = str(e)
        audit_logger.log_query(
            username      = username,
            role          = role,
            question      = request.question,
            doc_ids       = request.doc_ids,
            query_type    = None,
            sources_count = 0,
            latency_ms    = round((time.time() - start) * 1000, 2),
            success       = False,
            error         = error,
        )
        raise HTTPException(status_code=500, detail=error)


@router.get("/audit/recent")
async def get_audit_log(
    limit: int = 50,
    user:  dict = Depends(get_current_user),
):
    """
    Returns recent audit log entries.

    Only compliance officers and analysts can access this.
    Executives and portfolio managers cannot view the audit log.
    """
    role = user.get("role", "analyst")

    if role not in ["compliance", "analyst"]:
        raise HTTPException(
            status_code=403,
            detail=f"Role '{role}' does not have access to the audit log."
        )

    return {
        "entries": audit_logger.get_recent(limit=limit),
        "stats":   audit_logger.get_stats(),
    }


@router.get("/audit/stats")
async def get_audit_stats(user: dict = Depends(get_current_user)):
    """
    Returns aggregate audit statistics.
    Available to all authenticated users.
    """
    return audit_logger.get_stats()