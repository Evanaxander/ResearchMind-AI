"""
ResearchMind — Baseline Comparison Suite (Ollama, manual scoring)
------------------------------------------------------------------
Scores three systems using simple synchronous LLM-as-judge calls.
Avoids RAGAS async timeout issues with local Ollama models.

Usage (from project root):
    python evaluation/run_comparison.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from app.services.rag_service import RAGService
from app.services.agent_service import research_graph


DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
REPORT_PATH  = Path(__file__).parent / "comparison_report.json"

llm   = ChatOllama(model="mistral", temperature=0.2)
judge = ChatOllama(model="mistral", temperature=0)
rag   = RAGService()


# ── Load test cases ───────────────────────────────────────────────────────────

def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data['test_cases'])} test cases\n")
    return data["test_cases"]


# ── System 1: Base LLM ────────────────────────────────────────────────────────

def run_base_llm(test_cases: list[dict]) -> list[dict]:
    print("=" * 55)
    print("Running System 1: Base LLM (no retrieval)")
    print("=" * 55)
    results = []

    for i, case in enumerate(test_cases):
        q = case["question"]
        print(f"  [{i+1}/{len(test_cases)}] {q[:60]}...")
        try:
            answer = llm.invoke(
                f"You are a helpful assistant. Answer this question:\n\n{q}"
            ).content.strip()
        except Exception as e:
            print(f"    ERROR: {e}")
            answer = ""

        results.append({
            "question":     q,
            "answer":       answer,
            "context":      [],
            "ground_truth": case["ground_truth"],
        })
        print(f"    Preview: {answer[:80]}...")

    print()
    return results


# ── System 2: RAG only ────────────────────────────────────────────────────────

def run_rag_only(test_cases: list[dict]) -> list[dict]:
    print("=" * 55)
    print("Running System 2: RAG only (single retrieval)")
    print("=" * 55)
    results = []

    for i, case in enumerate(test_cases):
        q = case["question"]
        print(f"  [{i+1}/{len(test_cases)}] {q[:60]}...")
        try:
            hits          = rag.search(q, top_k=5)
            context_texts = [doc.page_content for doc, _ in hits]
            context_block = "\n\n---\n\n".join(context_texts)

            prompt = f"""Answer the question using ONLY the context below.
Cite sources clearly.

CONTEXT:
{context_block}

QUESTION: {q}"""

            answer = llm.invoke(prompt).content.strip()
        except Exception as e:
            print(f"    ERROR: {e}")
            answer, context_texts = "", []

        results.append({
            "question":     q,
            "answer":       answer,
            "context":      context_texts,
            "ground_truth": case["ground_truth"],
        })
        print(f"    Chunks: {len(context_texts)}  Preview: {answer[:70]}...")

    print()
    return results


# ── System 3: Multi-Agent RAG ─────────────────────────────────────────────────

def run_multi_agent(test_cases: list[dict]) -> list[dict]:
    print("=" * 55)
    print("Running System 3: Multi-Agent RAG")
    print("=" * 55)
    results = []

    for i, case in enumerate(test_cases):
        q = case["question"]
        print(f"  [{i+1}/{len(test_cases)}] {q[:60]}...")
        try:
            state = research_graph.invoke({
                "question":         q,
                "sub_tasks":        [],
                "retrieved_chunks": [],
                "answer":           "",
                "sources":          [],
                "agent_steps":      [],
                "doc_ids":          None,
            })
            answer        = state["answer"]
            context_texts = [c["text"] for c in state["retrieved_chunks"]]
        except Exception as e:
            print(f"    ERROR: {e}")
            answer, context_texts = "", []

        results.append({
            "question":     q,
            "answer":       answer,
            "context":      context_texts,
            "ground_truth": case["ground_truth"],
        })
        print(f"    Chunks: {len(context_texts)}  Preview: {answer[:70]}...")

    print()
    return results


# ── Manual LLM-as-judge scoring ───────────────────────────────────────────────

def score_faithfulness(answer: str, context: list[str]) -> float:
    """
    Ask the judge: does every claim in the answer come from the context?
    Returns a score 0.0 to 1.0.
    """
    if not answer or not context:
        return 0.0

    context_block = "\n\n".join(context[:3])  # use top 3 chunks

    prompt = f"""You are an evaluation judge.

CONTEXT:
{context_block}

ANSWER:
{answer}

Rate how faithfully the answer sticks to the context on a scale from 0 to 10.
10 = every claim is supported by the context.
0  = the answer contains many unsupported claims.

Reply with a single integer between 0 and 10. Nothing else."""

    try:
        response = judge.invoke(prompt).content.strip()
        # Extract first number found
        for token in response.split():
            clean = token.strip(".,")
            if clean.isdigit():
                return min(int(clean), 10) / 10.0
    except Exception:
        pass
    return 0.5


def score_relevancy(question: str, answer: str) -> float:
    """
    Ask the judge: does the answer actually address the question?
    Returns a score 0.0 to 1.0.
    """
    if not answer:
        return 0.0

    prompt = f"""You are an evaluation judge.

QUESTION:
{question}

ANSWER:
{answer}

Rate how directly and completely the answer addresses the question on a scale from 0 to 10.
10 = the answer directly and completely addresses the question.
0  = the answer is off-topic or unhelpful.

Reply with a single integer between 0 and 10. Nothing else."""

    try:
        response = judge.invoke(prompt).content.strip()
        for token in response.split():
            clean = token.strip(".,")
            if clean.isdigit():
                return min(int(clean), 10) / 10.0
    except Exception:
        pass
    return 0.5


def score_context_recall(answer: str, ground_truth: str) -> float:
    """
    Ask the judge: does the answer cover the key points in the ground truth?
    Returns a score 0.0 to 1.0.
    """
    if not answer:
        return 0.0

    prompt = f"""You are an evaluation judge.

EXPECTED ANSWER:
{ground_truth}

ACTUAL ANSWER:
{answer}

Rate how many key points from the expected answer appear in the actual answer, on a scale from 0 to 10.
10 = all key points are covered.
0  = none of the key points are covered.

Reply with a single integer between 0 and 10. Nothing else."""

    try:
        response = judge.invoke(prompt).content.strip()
        for token in response.split():
            clean = token.strip(".,")
            if clean.isdigit():
                return min(int(clean), 10) / 10.0
    except Exception:
        pass
    return 0.5


def score_system(results: list[dict], label: str) -> dict:
    """Score all questions for one system and return averages."""
    print(f"Scoring {label}...")

    faith_scores   = []
    rel_scores     = []
    recall_scores  = []

    for i, r in enumerate(results):
        print(f"  [{i+1}/{len(results)}] scoring...")

        faith  = score_faithfulness(r["answer"], r["context"])
        rel    = score_relevancy(r["question"], r["answer"])
        recall = score_context_recall(r["answer"], r["ground_truth"])

        faith_scores.append(faith)
        rel_scores.append(rel)
        recall_scores.append(recall)

        print(f"    faithfulness={faith:.2f}  relevancy={rel:.2f}  recall={recall:.2f}")

    scores = {
        "faithfulness":     round(sum(faith_scores)  / len(faith_scores),  4),
        "answer_relevancy": round(sum(rel_scores)    / len(rel_scores),    4),
        "context_recall":   round(sum(recall_scores) / len(recall_scores), 4)
                            if label != "Base LLM" else "N/A",
    }

    print(f"  Final: {scores}\n")
    return scores


# ── Print comparison table ────────────────────────────────────────────────────

def print_table(all_scores: dict):
    print("\n")
    print("=" * 65)
    print("  BASELINE COMPARISON REPORT")
    print("=" * 65)
    print(f"  {'System':<22} {'Faithfulness':>14} {'Relevancy':>12} {'Ctx Recall':>12}")
    print("-" * 65)

    for system, scores in all_scores.items():
        faith  = f"{scores['faithfulness']:.4f}"
        rel    = f"{scores['answer_relevancy']:.4f}"
        recall = scores["context_recall"]
        recall = f"{recall:.4f}" if isinstance(recall, float) else recall
        print(f"  {system:<22} {faith:>14} {rel:>12} {recall:>12}")

    print("=" * 65)

    base  = all_scores["Base LLM"]
    multi = all_scores["Multi-Agent RAG"]

    faith_pct = round(((multi["faithfulness"] - base["faithfulness"])
                       / max(base["faithfulness"], 0.01)) * 100, 1)
    rel_pct   = round(((multi["answer_relevancy"] - base["answer_relevancy"])
                       / max(base["answer_relevancy"], 0.01)) * 100, 1)

    print(f"\n  Multi-Agent RAG vs Base LLM:")
    print(f"    Faithfulness improvement:  +{faith_pct}%")
    print(f"    Relevancy improvement:     +{rel_pct}%\n")


def save_report(all_scores: dict):
    base  = all_scores["Base LLM"]
    multi = all_scores["Multi-Agent RAG"]

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "model":     "mistral (local via Ollama)",
        "scorer":    "LLM-as-judge (synchronous)",
        "scores":    all_scores,
        "improvements": {
            "faithfulness_vs_base": str(round(
                ((multi["faithfulness"] - base["faithfulness"])
                 / max(base["faithfulness"], 0.01)) * 100, 1)) + "%",
            "relevancy_vs_base": str(round(
                ((multi["answer_relevancy"] - base["answer_relevancy"])
                 / max(base["answer_relevancy"], 0.01)) * 100, 1)) + "%",
        }
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Full report saved to: {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nResearchMind — Baseline Comparison Suite")
    print("=" * 55)

    test_cases = load_dataset()

    base_results  = run_base_llm(test_cases)
    rag_results   = run_rag_only(test_cases)
    agent_results = run_multi_agent(test_cases)

    all_scores = {
        "Base LLM":        score_system(base_results,  "Base LLM"),
        "RAG only":        score_system(rag_results,   "RAG only"),
        "Multi-Agent RAG": score_system(agent_results, "Multi-Agent RAG"),
    }

    print_table(all_scores)
    save_report(all_scores)