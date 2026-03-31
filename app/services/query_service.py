from app.models.schemas import QueryRequest, QueryResponse
from app.services.financial_agent import financial_graph
from app.core.config import settings


class QueryService:
    """
    FinanceIQ query orchestrator.
    Delegates to the four-agent financial research pipeline:
    Planner → Researcher → Analyst → Synthesizer
    """

    async def answer(self, request: QueryRequest) -> QueryResponse:

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

        final_state = financial_graph.invoke(initial_state)

        return QueryResponse(
            question    = request.question,
            answer      = final_state["answer"],
            query_type  = final_state["query_type"],
            sources     = final_state["sources"],
            agent_steps = final_state["agent_steps"],
            analysis    = final_state.get("analysis"),
        )