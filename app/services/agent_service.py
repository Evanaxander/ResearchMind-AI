from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from app.services.rag_service import RAGService
from app.models.schemas import SourceChunk


# ── Shared state passed between all agents ────────────────────────────────────

class ResearchState(TypedDict):
    question: str
    sub_tasks: List[str]
    retrieved_chunks: List[dict]
    answer: str
    sources: List[SourceChunk]
    agent_steps: List[str]
    doc_ids: Optional[List[str]]


# ── LLM and RAG shared across agents ─────────────────────────────────────────

llm = ChatOllama(
    model="mistral",
    temperature=0.2,
)
rag = RAGService()


# ── Agent 1: Planner ──────────────────────────────────────────────────────────

def planner_agent(state: ResearchState) -> ResearchState:
    """
    Reads the user question and breaks it into 2-4 focused sub-tasks.
    Each sub-task becomes a separate FAISS search query.
    """
    question = state["question"]

    prompt = f"""You are a research planner.
Break this question into 2-4 specific search queries.
Each query should target a different aspect of the question.
Return ONLY a numbered list, one query per line. Nothing else.

Question: {question}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    sub_tasks = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            line = line.split(".", 1)[-1].strip()
            line = line.split(")", 1)[-1].strip()
        if line:
            sub_tasks.append(line)

    if not sub_tasks:
        sub_tasks = [question]

    return {
        **state,
        "sub_tasks": sub_tasks,
        "agent_steps": state["agent_steps"] + [
            f"[Planner] Broke question into {len(sub_tasks)} sub-tasks: {sub_tasks}"
        ],
    }


# ── Agent 2: Researcher ───────────────────────────────────────────────────────

def researcher_agent(state: ResearchState) -> ResearchState:
    """
    Runs a FAISS search for each sub-task from the planner.
    Collects and deduplicates all retrieved chunks.
    """
    sub_tasks = state["sub_tasks"]
    doc_ids = state.get("doc_ids")
    all_chunks = []
    seen_texts = set()

    for task in sub_tasks:
        results = rag.search(task, doc_ids=doc_ids, top_k=3)

        for doc, score in results:
            text = doc.page_content.strip()
            if text in seen_texts:
                continue
            seen_texts.add(text)

            all_chunks.append({
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "filename": doc.metadata.get("filename", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "text": text,
                "score": round(float(score), 4),
                "sub_task": task,
            })

    return {
        **state,
        "retrieved_chunks": all_chunks,
        "agent_steps": state["agent_steps"] + [
            f"[Researcher] Retrieved {len(all_chunks)} unique chunks "
            f"across {len(sub_tasks)} sub-tasks"
        ],
    }


# ── Agent 3: Synthesizer ──────────────────────────────────────────────────────

def synthesizer_agent(state: ResearchState) -> ResearchState:
    """
    Reads all retrieved chunks and writes a comprehensive cited answer.
    Only uses information from the retrieved context.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    if not chunks:
        return {
            **state,
            "answer": (
                "I could not find relevant information in your uploaded documents. "
                "Please upload a document and try again."
            ),
            "sources": [],
            "agent_steps": state["agent_steps"] + [
                "[Synthesizer] No chunks found — returned empty answer"
            ],
        }

    context = "\n\n---\n\n".join(
        f"[Source: {c['filename']}, chunk {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""You are a research synthesizer.
Answer the question using ONLY the provided context.
Be comprehensive but concise.
Always cite your sources using [filename, chunk N] format.
If the context does not contain enough information, say so clearly.

CONTEXT:
{context}

QUESTION:
{question}

Write a well-structured answer with citations:"""

    response = llm.invoke(prompt)

    sources = [
        SourceChunk(
            doc_id=c["doc_id"],
            filename=c["filename"],
            chunk_index=c["chunk_index"],
            text=c["text"],
            score=c["score"],
        )
        for c in chunks
    ]

    return {
        **state,
        "answer": response.content.strip(),
        "sources": sources,
        "agent_steps": state["agent_steps"] + [
            f"[Synthesizer] Generated answer using {len(chunks)} chunks"
        ],
    }


# ── Build the LangGraph graph ─────────────────────────────────────────────────

def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("synthesizer", synthesizer_agent)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


research_graph = build_research_graph()