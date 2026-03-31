"""
Graph Router
-------------
API endpoints for the Neo4j knowledge graph features.

Endpoints:
  GET  /graph/overview          → full document graph (nodes + edges)
  GET  /graph/company/{ticker}  → all documents for a company
  GET  /graph/contradictions    → all detected contradictions
  POST /graph/check/{doc_id}    → manually trigger contradiction check
  GET  /graph/related/{doc_id}  → find documents related to a given doc
"""

from fastapi import APIRouter, HTTPException
from app.services.graph_service import GraphService

router = APIRouter()
graph  = GraphService()


@router.get("/graph/overview")
async def get_graph_overview():
    """
    Returns the complete document knowledge graph.

    Response includes:
    - All document nodes with metadata
    - All relationships between documents
    - Summary stats (total docs, relationships, contradictions)

    Used by the frontend dashboard to render the graph visualization.
    """
    try:
        return graph.get_document_graph()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/company/{ticker}")
async def get_company_documents(ticker: str):
    """
    Returns all documents uploaded for a specific company.

    Example: GET /graph/company/AAPL
    Returns all Apple documents — 10-Ks, earnings calls, analyst reports.

    This is how an analyst quickly sees everything they have
    on a company without searching manually.
    """
    try:
        ticker = ticker.upper()
        docs   = graph.find_docs_by_ticker(ticker)

        if not docs:
            return {
                "ticker":    ticker,
                "documents": [],
                "message":   f"No documents found for {ticker}. "
                             f"Upload a 10-K or earnings transcript to get started."
            }

        return {
            "ticker":          ticker,
            "document_count":  len(docs),
            "documents":       docs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/contradictions")
async def get_all_contradictions(ticker: str = None):
    """
    Returns all detected contradictions across all documents.

    Optional: filter by ticker symbol.
    Example: GET /graph/contradictions?ticker=AAPL

    This is one of FinanceIQ's most valuable features —
    automatically surfaces conflicts between filings and
    earnings call statements that analysts would otherwise
    spend hours finding manually.
    """
    try:
        contradictions = graph.find_contradictions(ticker=ticker)

        return {
            "total":           len(contradictions),
            "ticker_filter":   ticker,
            "contradictions":  contradictions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/related/{doc_id}")
async def get_related_documents(doc_id: str):
    """
    Returns documents related to a given document via graph traversal.

    This is different from FAISS similarity search —
    it follows explicit graph relationships (same company,
    same period, updates, contradicts) rather than
    semantic similarity.

    Use this when you want to see what else is connected
    to a document you're analyzing.
    """
    try:
        related = graph.find_related_docs(doc_id)

        return {
            "doc_id":          doc_id,
            "related_count":   len(related),
            "related_docs":    related,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats")
async def get_graph_stats():
    """Quick graph statistics — used by health check and dashboard."""
    try:
        return graph.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))