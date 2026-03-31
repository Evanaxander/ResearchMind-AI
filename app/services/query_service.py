from app.models.schemas import QueryRequest, QueryResponse
from app.services.financial_agent import financial_graph
from app.core.config import settings
import time
import asyncio


class QueryService:
    """
    FinanceIQ query orchestrator.
    Delegates to the four-agent financial research pipeline:
    Planner → Researcher → Analyst → Synthesizer
    """

    def __init__(self):
        self._cache: dict[tuple, tuple[float, QueryResponse]] = {}

    def _cache_key(self, request: QueryRequest) -> tuple:
        return (
            request.question.strip().lower(),
            tuple(sorted(request.doc_ids or [])),
            request.top_k,
            (request.doc_type_filter or "").strip().lower(),
            (request.ticker_filter or "").strip().upper(),
            (request.period_filter or "").strip().lower(),
            settings.ANALYSIS_DOMAIN,
            settings.FAST_QUERY_MODE,
        )

    async def answer(self, request: QueryRequest) -> QueryResponse:
        key = self._cache_key(request)
        now = time.time()
        ttl = max(0, settings.QUERY_CACHE_TTL_SECONDS)

        if ttl > 0:
            cached = self._cache.get(key)
            if cached and (now - cached[0] <= ttl):
                return cached[1]

        initial_state = {
            "question":         request.question,
            "query_type":       "general",
            "sub_tasks":        [],
            "retrieved_chunks": [],
            "analysis":         "",
            "answer":           "",
            "sources":          [],
            "agent_steps":      [],
            "doc_ids":          request.doc_ids,
            "top_k":            request.top_k,
            "doc_type_filter":  request.doc_type_filter,
            "ticker_filter":    request.ticker_filter,
            "period_filter":    request.period_filter,
            "domain_mode":      settings.ANALYSIS_DOMAIN,
        }

        final_state = await asyncio.to_thread(financial_graph.invoke, initial_state)

        response = QueryResponse(
            question    = request.question,
            answer      = final_state["answer"],
            query_type  = final_state["query_type"],
            sources     = final_state["sources"],
            agent_steps = final_state["agent_steps"],
            analysis    = final_state.get("analysis"),
        )

        if ttl > 0:
            self._cache[key] = (now, response)

        return response