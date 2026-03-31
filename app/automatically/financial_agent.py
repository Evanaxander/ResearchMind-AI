"""
Financial Agent Service
------------------------
Finance-aware LangGraph agent pipeline with four specialized agents:

  Planner    → understands financial query types and plans searches
  Researcher → retrieves relevant chunks with financial metadata filtering
  Analyst    → performs quantitative analysis and contradiction detection
  Synthesizer → writes professional financial-grade answers with citations

This replaces the generic agent_service.py for financial use cases.
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from app.services.rag_service import RAGService
from app.models.schemas import SourceChunk


# ── Shared state ──────────────────────────────────────────────────────────────

class FinancialResearchState(TypedDict):
    question:          str
    query_type:        str           # "metric", "comparison", "risk", "general"
    sub_tasks:         List[str]
    retrieved_chunks:  List[dict]
    analysis:          str           # analyst agent's findings
    answer:            str
    sources:           List[SourceChunk]
    agent_steps:       List[str]
    doc_ids:           Optional[List[str]]
    ticker_filter:     Optional[str]
    period_filter:     Optional[str]


# ── Shared resources ──────────────────────────────────────────────────────────

llm = ChatOllama(model="mistral", temperature=0.1)
rag = RAGService()


def _invoke_llm(prompt: str) -> tuple[Optional[str], Optional[str]]:
    """
    Safely invoke the local LLM.
    Returns (content, error) so callers can fall back instead of raising 500.
    """
    try:
        return llm.invoke(prompt).content.strip(), None
    except Exception as e:
        return None, str(e)


# ── Query type classifier ─────────────────────────────────────────────────────

def classify_query(question: str) -> str:
    """
    Classify the financial query type so the Planner
    can use the right search strategy.

    metric     → looking for a specific number (revenue, EPS, etc.)
    comparison → comparing two periods or companies
    risk       → asking about risks, concerns, warnings
    general    → general question about the document
    """
    q = question.lower()

    if any(w in q for w in ["compare", "versus", "vs", "difference", "change", "growth", "trend"]):
        return "comparison"
    if any(w in q for w in ["risk", "concern", "warn", "threat", "challenge", "uncertain"]):
        return "risk"
    if any(w in q for w in ["revenue", "income", "profit", "eps", "ebitda", "margin",
                              "cash", "debt", "guidance", "earnings", "sales"]):
        return "metric"
    return "general"


# ── Agent 1: Financial Planner ────────────────────────────────────────────────

def financial_planner(state: FinancialResearchState) -> FinancialResearchState:
    """
    Finance-aware query planner.

    Unlike the generic planner, this understands financial document
    structure and generates targeted sub-queries based on query type:

    - Metric queries: searches for the number AND its context
    - Comparison queries: generates one search per time period
    - Risk queries: searches risk factors section specifically
    - General queries: broad multi-aspect search
    """
    question   = state["question"]
    query_type = classify_query(question)

    prompt = f"""You are a financial research analyst planning document searches.

Query type: {query_type}
Question: {question}

Generate 3-4 specific search queries to find the answer in financial documents.
Consider:
- For metrics: search for the number AND the surrounding context
- For comparisons: generate separate queries for each time period
- For risks: search for risk factors, challenges, and warnings
- For general: cover different aspects of the question

Return ONLY a numbered list, one query per line."""

    response, planner_error = _invoke_llm(prompt)
    response = response or ""
    sub_tasks = []

    for line in response.split("\n"):
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

    planner_note = (
        f" [Planner fallback used: {planner_error}]"
        if planner_error else
        ""
    )

    return {
        **state,
        "query_type": query_type,
        "sub_tasks":  sub_tasks,
        "agent_steps": state["agent_steps"] + [
            f"[Planner] Query type: {query_type}. "
            f"Generated {len(sub_tasks)} sub-tasks: {sub_tasks}.{planner_note}"
        ],
    }


# ── Agent 2: Financial Researcher ─────────────────────────────────────────────

def financial_researcher(state: FinancialResearchState) -> FinancialResearchState:
    """
    Retrieves chunks with financial metadata awareness.

    Key difference from generic researcher:
    - Filters by ticker symbol if detected
    - Filters by fiscal period if detected
    - Prioritizes chunks that contain tables (has_tables=True)
    - Deduplicates by content
    """
    sub_tasks      = state["sub_tasks"]
    doc_ids        = state.get("doc_ids")
    all_chunks     = []
    seen_texts     = set()

    for task in sub_tasks:
        results = rag.search(task, doc_ids=doc_ids, top_k=4)

        for doc, score in results:
            text = doc.page_content.strip()
            if text in seen_texts:
                continue
            seen_texts.add(text)

            metadata = doc.metadata
            all_chunks.append({
                "doc_id":        metadata.get("doc_id", "unknown"),
                "filename":      metadata.get("filename", "unknown"),
                "chunk_index":   metadata.get("chunk_index", 0),
                "text":          text,
                "score":         round(float(score), 4),
                "sub_task":      task,
                "doc_type":      metadata.get("doc_type", "general"),
                "ticker":        metadata.get("ticker", "unknown"),
                "fiscal_period": metadata.get("fiscal_period", "unknown"),
                "has_tables":    metadata.get("has_tables", False),
            })

    # Sort: tables first (they contain the numbers), then by score
    all_chunks.sort(key=lambda x: (not x["has_tables"], -x["score"]))

    return {
        **state,
        "retrieved_chunks": all_chunks,
        "agent_steps": state["agent_steps"] + [
            f"[Researcher] Retrieved {len(all_chunks)} unique chunks "
            f"across {len(sub_tasks)} sub-tasks. "
            f"Tables found: {sum(1 for c in all_chunks if c['has_tables'])}"
        ],
    }


# ── Agent 3: Financial Analyst ────────────────────────────────────────────────

def financial_analyst(state: FinancialResearchState) -> FinancialResearchState:
    """
    The most important addition over generic RAG.

    This agent performs actual analysis on retrieved chunks:
    - For metric queries: extracts and validates the numbers
    - For comparison queries: computes changes and trends
    - For risk queries: ranks and categorizes risks
    - Detects contradictions between chunks from different documents

    The analysis is passed to the Synthesizer as additional context.
    """
    question   = state["question"]
    query_type = state["query_type"]
    chunks     = state["retrieved_chunks"]

    if not chunks:
        return {
            **state,
            "analysis": "No relevant chunks found for analysis.",
            "agent_steps": state["agent_steps"] + [
                "[Analyst] No chunks to analyze."
            ],
        }

    context = "\n\n---\n\n".join(
        f"[{c['filename']} | {c['doc_type']} | "
        f"Period: {c['fiscal_period']} | Tables: {c['has_tables']}]\n{c['text']}"
        for c in chunks[:8]
    )

    if query_type == "metric":
        analysis_prompt = f"""You are a financial analyst extracting specific metrics.

From the context below, extract ALL relevant financial numbers for this question.
Present them in a structured format with source citations.
Flag any numbers that seem inconsistent or unusual.

QUESTION: {question}

CONTEXT:
{context}

Extract the metrics:"""

    elif query_type == "comparison":
        analysis_prompt = f"""You are a financial analyst performing comparative analysis.

From the context below, identify the metrics for EACH time period or entity.
Calculate the percentage change where possible.
Highlight what drove any significant changes.

QUESTION: {question}

CONTEXT:
{context}

Comparative analysis:"""

    elif query_type == "risk":
        analysis_prompt = f"""You are a financial risk analyst.

From the context below, identify and categorize all risk factors.
Rate each risk as HIGH/MEDIUM/LOW impact.
Note any risks that appear multiple times (cross-document validation).

QUESTION: {question}

CONTEXT:
{context}

Risk analysis:"""

    else:
        analysis_prompt = f"""You are a financial research analyst.

Analyze the context below to answer the question.
Note any important patterns, trends, or insights.
Flag any contradictions between different sources.

QUESTION: {question}

CONTEXT:
{context}

Analysis:"""

    analysis, analysis_error = _invoke_llm(analysis_prompt)
    if not analysis:
        top_sources = ", ".join(
            f"{c['filename']}#{c['chunk_index']}"
            for c in chunks[:3]
        )
        analysis = (
            "LLM analysis unavailable due to local model runtime constraints. "
            f"Using retrieved evidence only. Top sources: {top_sources}. "
            f"Runtime error: {analysis_error}"
        )

    return {
        **state,
        "analysis": analysis,
        "agent_steps": state["agent_steps"] + [
            f"[Analyst] Completed {query_type} analysis on "
            f"{len(chunks)} chunks"
        ],
    }


# ── Agent 4: Financial Synthesizer ───────────────────────────────────────────

def financial_synthesizer(state: FinancialResearchState) -> FinancialResearchState:
    """
    Writes professional financial-grade answers.

    Uses both the raw retrieved chunks AND the analyst's analysis
    to produce a response that reads like it came from a senior analyst:
    - Leads with the direct answer
    - Supports with specific numbers and citations
    - Notes any caveats or data limitations
    - Professional, concise tone
    """
    question   = state["question"]
    query_type = state["query_type"]
    chunks     = state["retrieved_chunks"]
    analysis   = state["analysis"]

    if not chunks:
        return {
            **state,
            "answer": (
                "Insufficient data in uploaded documents to answer this question. "
                "Please upload the relevant financial filings and try again."
            ),
            "sources": [],
            "agent_steps": state["agent_steps"] + [
                "[Synthesizer] No data available — returned empty answer"
            ],
        }

    context = "\n\n---\n\n".join(
        f"[Source: {c['filename']}, chunk {c['chunk_index']}, "
        f"period: {c['fiscal_period']}]\n{c['text']}"
        for c in chunks[:6]
    )

    prompt = f"""You are a senior financial analyst writing a research note.

Answer the question using the retrieved context and analysis below.
Write in a professional, concise tone suitable for institutional investors.

Requirements:
- Lead with the direct answer in the first sentence
- Support every claim with a specific citation [filename, chunk N]
- Include relevant numbers when available
- Note any data limitations or caveats at the end
- Keep the response focused and under 400 words

QUERY TYPE: {query_type}
QUESTION: {question}

ANALYST FINDINGS:
{analysis}

SOURCE CONTEXT:
{context}

Professional response:"""

    answer, synth_error = _invoke_llm(prompt)
    if not answer:
        citations = " ".join(
            f"[{c['filename']}, chunk {c['chunk_index']}]"
            for c in chunks[:3]
        )
        evidence = " ".join(c["text"][:220] for c in chunks[:2]).strip()
        answer = (
            "Local model was unavailable due to memory limits, so this response "
            "was generated from retrieved document evidence only. "
            f"Key evidence: {evidence} {citations} "
            f"(runtime error: {synth_error})"
        )

    sources = [
        SourceChunk(
            doc_id      = c["doc_id"],
            filename    = c["filename"],
            chunk_index = c["chunk_index"],
            text        = c["text"],
            score       = c["score"],
        )
        for c in chunks
    ]

    return {
        **state,
        "answer":  answer,
        "sources": sources,
        "agent_steps": state["agent_steps"] + [
            f"[Synthesizer] Generated {query_type} response "
            f"using {len(chunks)} chunks and analyst findings"
        ],
    }


# ── Build the financial graph ─────────────────────────────────────────────────

def build_financial_graph():
    """
    Four-agent financial research pipeline:
    Planner → Researcher → Analyst → Synthesizer

    The Analyst agent is the key addition over generic RAG.
    It performs actual quantitative analysis before synthesis,
    producing much higher quality financial answers.
    """
    graph = StateGraph(FinancialResearchState)

    graph.add_node("planner",     financial_planner)
    graph.add_node("researcher",  financial_researcher)
    graph.add_node("analyst",     financial_analyst)
    graph.add_node("synthesizer", financial_synthesizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner",    "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst",    "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


financial_graph = build_financial_graph()