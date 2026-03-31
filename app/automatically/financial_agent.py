"""Domain-aware LangGraph pipeline used for general and finance modes."""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from app.core.config import settings
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
    top_k:             int
    doc_type_filter:   Optional[str]
    ticker_filter:     Optional[str]
    period_filter:     Optional[str]
    domain_mode:       str


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


def _clean_text_excerpt(text: str, max_len: int = 220) -> str:
    cleaned = " ".join((text or "").split())
    return (cleaned[:max_len] + "...") if len(cleaned) > max_len else cleaned


# ── Query type classifier ─────────────────────────────────────────────────────

def classify_query(question: str) -> str:
    """
    Classify the query type so the Planner
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
    if any(w in q for w in ["revenue", "income", "profit", "eps", "margin", "kpi", "metric",
                            "count", "rate", "budget", "cost", "sales", "growth"]):
        return "metric"
    return "general"


# ── Agent 1: Financial Planner ────────────────────────────────────────────────

def financial_planner(state: FinancialResearchState) -> FinancialResearchState:
    """
    Domain-aware query planner.
    """
    question   = state["question"]
    query_type = classify_query(question)
    fast_mode = settings.FAST_QUERY_MODE

    if fast_mode:
        return {
            **state,
            "query_type": query_type,
            "sub_tasks": [question],
            "agent_steps": state["agent_steps"] + [
                "[Planner] Fast mode enabled: using direct query without LLM planning"
            ],
        }

    domain_mode = state.get("domain_mode", "general")
    domain_guidance = (
        "Use finance terminology (ticker, fiscal period, guidance) when relevant."
        if domain_mode == "finance" else
        "Use document-agnostic wording suitable for any report or policy corpus."
    )

    prompt = f"""You are a research analyst planning document searches.

Query type: {query_type}
Domain mode: {domain_mode}
Question: {question}

Generate 3-4 specific search queries to find the answer in uploaded documents.
Consider:
- For metrics: search for the number AND the surrounding context
- For comparisons: generate separate queries for each time period
- For risks: search for risk factors, challenges, and warnings
- For general: cover different aspects of the question
{domain_guidance}

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
    sub_tasks = sub_tasks[:settings.QUERY_MAX_SUBTASKS]

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
    Retrieves chunks with metadata awareness and lightweight filtering.
    """
    sub_tasks      = state["sub_tasks"]
    doc_ids        = state.get("doc_ids")
    ticker_filter  = (state.get("ticker_filter") or "").strip().upper()
    period_filter  = (state.get("period_filter") or "").strip().lower()
    type_filter    = (state.get("doc_type_filter") or "").strip().lower()
    top_k          = max(2, min(int(state.get("top_k", 5) or 5), 12))
    if settings.FAST_QUERY_MODE:
        top_k = min(top_k, settings.FAST_TOP_K_CAP)
    all_chunks     = []
    seen_texts     = set()

    for task in sub_tasks:
        results = rag.search(task, doc_ids=doc_ids, top_k=top_k)

        for doc, score in results:
            text = doc.page_content.strip()
            if text in seen_texts:
                continue
            seen_texts.add(text)

            metadata = doc.metadata

            if ticker_filter and str(metadata.get("ticker", "")).upper() != ticker_filter:
                continue
            if period_filter and period_filter not in str(metadata.get("fiscal_period", "")).lower():
                continue
            if type_filter and type_filter not in str(metadata.get("doc_type", "")).lower():
                continue

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

    # Keep table-rich chunks near the top while preserving score importance.
    all_chunks.sort(key=lambda x: (x["score"], 0 if x["has_tables"] else 1))

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
    Build intermediate reasoning notes for synthesis.
    """
    question   = state["question"]
    query_type = state["query_type"]
    chunks     = state["retrieved_chunks"]
    domain_mode = state.get("domain_mode", "general")

    if settings.FAST_QUERY_MODE:
        return {
            **state,
            "analysis": "",
            "agent_steps": state["agent_steps"] + [
                "[Analyst] Fast mode enabled: skipped LLM analysis pass"
            ],
        }

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
        for c in chunks[:settings.ANALYST_CONTEXT_CHUNKS]
    )

    if query_type == "metric":
        analysis_prompt = f"""You are an analyst extracting evidence-backed metrics.

From the context below, extract ALL relevant numbers and quantitative statements.
Present them in a structured format with source citations.
Flag any numbers that seem inconsistent or unusual.
Domain mode: {domain_mode}

QUESTION: {question}

CONTEXT:
{context}

Extract the metrics:"""

    elif query_type == "comparison":
        analysis_prompt = f"""You are an analyst performing comparative analysis.

From the context below, identify the metrics for EACH time period or entity.
Calculate the percentage change where possible.
Highlight what drove any significant changes.

QUESTION: {question}

CONTEXT:
{context}

Comparative analysis:"""

    elif query_type == "risk":
        analysis_prompt = f"""You are a risk analyst.

From the context below, identify and categorize all risk factors.
Rate each risk as HIGH/MEDIUM/LOW impact.
Note any risks that appear multiple times (cross-document validation).

QUESTION: {question}

CONTEXT:
{context}

Risk analysis:"""

    else:
        analysis_prompt = f"""You are a research analyst.

Analyze the context below to answer the question.
Note any important patterns, trends, or insights.
Flag any contradictions between different sources.
    Domain mode: {domain_mode}

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
    Writes concise evidence-backed answers for any report domain.
    """
    question   = state["question"]
    query_type = state["query_type"]
    chunks     = state["retrieved_chunks"]
    analysis   = state["analysis"]
    domain_mode = state.get("domain_mode", "general")

    if settings.FAST_QUERY_MODE and chunks:
        top = chunks[:3]
        citations = " ".join(f"[{c['filename']}, chunk {c['chunk_index']}]" for c in top)
        snippets = " ".join(_clean_text_excerpt(c["text"], 180) for c in top[:2]).strip()
        quick_answer = (
            "Direct answer based on retrieved evidence: "
            f"{snippets} {citations} "
            "(Fast mode is enabled; disable FAST_QUERY_MODE for deeper synthesized reasoning.)"
        )

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
            "answer": quick_answer,
            "sources": sources,
            "agent_steps": state["agent_steps"] + [
                "[Synthesizer] Fast mode enabled: returned extractive answer without LLM synthesis"
            ],
        }

    if not chunks:
        return {
            **state,
            "answer": (
                "Insufficient data in uploaded documents to answer this question. "
                "Please upload relevant source documents and try again."
            ),
            "sources": [],
            "agent_steps": state["agent_steps"] + [
                "[Synthesizer] No data available — returned empty answer"
            ],
        }

    context = "\n\n---\n\n".join(
        f"[Source: {c['filename']}, chunk {c['chunk_index']}, "
        f"period: {c['fiscal_period']}]\n{c['text']}"
        for c in chunks[:settings.SYNTH_CONTEXT_CHUNKS]
    )

    prompt = f"""You are a senior analyst writing a research response.

Answer the question using the retrieved context and analysis below.
Write in a professional, concise tone.

Requirements:
- Lead with the direct answer in the first sentence
- Support every claim with a specific citation [filename, chunk N]
- Include relevant numbers when available
- Note any data limitations or caveats at the end
- Keep the response focused and under 400 words
Domain mode: {domain_mode}

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
    Four-agent research pipeline:
    Planner → Researcher → Analyst → Synthesizer
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