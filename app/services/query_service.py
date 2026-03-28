from app.models.schemas import QueryRequest, QueryResponse
from app.services.agent_service import research_graph


class QueryService:
    """
    Phase 3: Delegates everything to the LangGraph multi-agent graph.
    Now runs fully locally using Ollama/Mistral — no API key needed.
    """

    async def answer(self, request: QueryRequest) -> QueryResponse:

        initial_state = {
            "question": request.question,
            "sub_tasks": [],
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "agent_steps": [],
            "doc_ids": request.doc_ids,
        }

        final_state = research_graph.invoke(initial_state)

        return QueryResponse(
            question=request.question,
            answer=final_state["answer"],
            sources=final_state["sources"],
            agent_steps=final_state["agent_steps"],
        )