"""
RAGAS Evaluation Suite for ResearchMind
----------------------------------------
Run this script to benchmark your RAG pipeline.

Usage:
    python evaluation/run_eval.py

Output:
    - Per-question scores printed to terminal
    - Summary report saved to evaluation/eval_report.json

Metrics:
    faithfulness     → did the answer stay grounded in retrieved context?
    answer_relevancy → does the answer actually address the question?
    context_recall   → did FAISS retrieve the right chunks?
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.services.agent_service import research_graph
from app.core.config import settings


# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
REPORT_PATH  = Path(__file__).parent / "eval_report.json"


# ── Step 1: Load test cases ───────────────────────────────────────────────────

def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data['test_cases'])} test cases\n")
    return data["test_cases"]


# ── Step 2: Run each question through the agent pipeline ─────────────────────

def run_pipeline(test_cases: list[dict]) -> dict:
    """
    Runs every question through the full LangGraph agent pipeline
    (Planner → Researcher → Synthesizer) and collects:
      - the generated answer
      - the retrieved context chunks
    """
    questions        = []
    answers          = []
    contexts         = []
    ground_truths    = []

    for i, case in enumerate(test_cases):
        question     = case["question"]
        ground_truth = case["ground_truth"]

        print(f"[{i+1}/{len(test_cases)}] Running: {question[:60]}...")

        # Run the full agent graph
        initial_state = {
            "question": question,
            "sub_tasks": [],
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "agent_steps": [],
            "doc_ids": None,
        }

        try:
            final_state = research_graph.invoke(initial_state)

            answer = final_state["answer"]
            # Extract raw text from each retrieved chunk for RAGAS context
            context_texts = [
                chunk["text"]
                for chunk in final_state["retrieved_chunks"]
            ]

            print(f"    Answer preview: {answer[:80]}...")
            print(f"    Chunks retrieved: {len(context_texts)}\n")

        except Exception as e:
            print(f"    ERROR: {e}\n")
            answer        = ""
            context_texts = []

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

    return {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    }


# ── Step 3: Score with RAGAS ──────────────────────────────────────────────────

def score_with_ragas(pipeline_output: dict) -> dict:
    """
    Passes pipeline output to RAGAS and returns metric scores.
    RAGAS uses OpenAI internally to judge answer quality.
    """
    dataset = Dataset.from_dict(pipeline_output)

    # Tell RAGAS which LLM and embeddings to use for judging
    judge_llm        = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
    judge_embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    print("Scoring with RAGAS (this calls OpenAI)...\n")

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    return results


# ── Step 4: Print and save report ────────────────────────────────────────────

def save_report(results, pipeline_output: dict):
    scores = {
        "faithfulness":     round(float(results["faithfulness"]), 4),
        "answer_relevancy": round(float(results["answer_relevancy"]), 4),
        "context_recall":   round(float(results["context_recall"]), 4),
    }

    # Interpret scores
    def interpret(score: float) -> str:
        if score >= 0.8: return "GOOD"
        if score >= 0.6: return "FAIR"
        return "NEEDS IMPROVEMENT"

    print("=" * 55)
    print("  RAGAS EVALUATION REPORT")
    print("=" * 55)
    print(f"  Faithfulness      {scores['faithfulness']:.4f}  {interpret(scores['faithfulness'])}")
    print(f"  Answer Relevancy  {scores['answer_relevancy']:.4f}  {interpret(scores['answer_relevancy'])}")
    print(f"  Context Recall    {scores['context_recall']:.4f}  {interpret(scores['context_recall'])}")
    print("=" * 55)
    print()

    # Guidance based on scores
    if scores["faithfulness"] < 0.6:
        print("  Low faithfulness: your LLM is hallucinating.")
        print("  Fix: tighten the synthesizer system prompt.")
    if scores["answer_relevancy"] < 0.6:
        print("  Low answer relevancy: answers don't address questions.")
        print("  Fix: improve the planner sub-task generation.")
    if scores["context_recall"] < 0.6:
        print("  Low context recall: FAISS isn't finding the right chunks.")
        print("  Fix: reduce chunk size or increase top_k.")
    print()

    # Save full report to JSON
    report = {
        "timestamp":      datetime.utcnow().isoformat(),
        "scores":         scores,
        "interpretations": {k: interpret(v) for k, v in scores.items()},
        "num_test_cases": len(pipeline_output["question"]),
        "per_question": [
            {
                "question":      pipeline_output["question"][i],
                "answer":        pipeline_output["answer"][i],
                "chunks_retrieved": len(pipeline_output["contexts"][i]),
                "ground_truth":  pipeline_output["ground_truth"][i],
            }
            for i in range(len(pipeline_output["question"]))
        ],
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Full report saved to: {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nResearchMind — RAGAS Evaluation Suite")
    print("=" * 55)

    # Verify OpenAI key is set
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("sk-..."):
        print("ERROR: Set OPENAI_API_KEY in your .env file first.")
        sys.exit(1)

    test_cases      = load_dataset()
    pipeline_output = run_pipeline(test_cases)
    results         = score_with_ragas(pipeline_output)
    save_report(results, pipeline_output)