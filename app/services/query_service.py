from app.models.schemas import QueryRequest, QueryResponse
from app.services.agent_service import research_graph


class QueryService:
    """
    Phase 3: Delegates everything to the LangGraph multi-agent graph.
    The graph runs: Planner → Researcher → Synthesizer.
    """

    async def answer(self, request: QueryRequest) -> QueryResponse:

        # Initial state fed into the graph
        initial_state = {
            "question": request.question,
            "sub_tasks": [],
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "agent_steps": [],
            "doc_ids": request.doc_ids,
        }

        # Run the full agent graph
        final_state = research_graph.invoke(initial_state)

        return QueryResponse(
            question=request.question,
            answer=final_state["answer"],
            sources=final_state["sources"],
            agent_steps=final_state["agent_steps"],
        )