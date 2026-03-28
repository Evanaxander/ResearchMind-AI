"""
ResearchMind — RAGAS Evaluation Suite (Ollama, manual scoring)
--------------------------------------------------------------
Scores the full multi-agent pipeline using synchronous LLM-as-judge.

Usage (from project root):
    python evaluation/run_eval.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from app.services.agent_service import research_graph


DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
REPORT_PATH  = Path(__file__).parent / "eval_report.json"

judge = ChatOllama(model="mistral", temperature=0)


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data['test_cases'])} test cases\n")
    return data["test_cases"]


def run_pipeline(test_cases: list[dict]) -> list[dict]:
    results = []

    for i, case in enumerate(test_cases):
        q = case["question"]
        print(f"[{i+1}/{len(test_cases)}] {q[:60]}...")

        try:
            state         = research_graph.invoke({
                "question": q, "sub_tasks": [], "retrieved_chunks": [],
                "answer": "", "sources": [], "agent_steps": [], "doc_ids": None,
            })
            answer        = state["answer"]
            context_texts = [c["text"] for c in state["retrieved_chunks"]]
            print(f"    Answer: {answer[:80]}...")
            print(f"    Chunks: {len(context_texts)}\n")
        except Exception as e:
            print(f"    ERROR: {e}\n")
            answer, context_texts = "", []

        results.append({
            "question":     q,
            "answer":       answer,
            "context":      context_texts,
            "ground_truth": case["ground_truth"],
        })

    return results


def judge_score(prompt: str) -> float:
    try:
        response = judge.invoke(prompt).content.strip()
        for token in response.split():
            clean = token.strip(".,")
            if clean.isdigit():
                return min(int(clean), 10) / 10.0
    except Exception:
        pass
    return 0.5


def score_all(results: list[dict]) -> dict:
    print("Scoring with LLM-as-judge...\n")

    faith_scores, rel_scores, recall_scores = [], [], []

    for i, r in enumerate(results):
        print(f"  [{i+1}/{len(results)}] scoring...")

        # Faithfulness
        ctx = "\n\n".join(r["context"][:3]) if r["context"] else "No context"
        faith = judge_score(f"""Rate 0-10: does every claim in the ANSWER come from the CONTEXT?
10=fully grounded, 0=hallucinated. Reply with one integer only.

CONTEXT: {ctx}
ANSWER: {r['answer']}""")

        # Relevancy
        rel = judge_score(f"""Rate 0-10: does the ANSWER directly address the QUESTION?
10=fully answers it, 0=off-topic. Reply with one integer only.

QUESTION: {r['question']}
ANSWER: {r['answer']}""")

        # Context recall
        recall = judge_score(f"""Rate 0-10: does the ANSWER cover the key points in the EXPECTED answer?
10=all key points covered, 0=none covered. Reply with one integer only.

EXPECTED: {r['ground_truth']}
ANSWER: {r['answer']}""")

        faith_scores.append(faith)
        rel_scores.append(rel)
        recall_scores.append(recall)
        print(f"    faithfulness={faith:.2f}  relevancy={rel:.2f}  recall={recall:.2f}")

    return {
        "faithfulness":     round(sum(faith_scores)  / len(faith_scores),  4),
        "answer_relevancy": round(sum(rel_scores)    / len(rel_scores),    4),
        "context_recall":   round(sum(recall_scores) / len(recall_scores), 4),
    }


def interpret(score: float) -> str:
    if score >= 0.8: return "GOOD"
    if score >= 0.6: return "FAIR"
    return "NEEDS IMPROVEMENT"


def save_report(scores: dict, results: list[dict]):
    print("\n" + "=" * 55)
    print("  RAGAS EVALUATION REPORT")
    print("=" * 55)
    print(f"  Faithfulness      {scores['faithfulness']:.4f}  {interpret(scores['faithfulness'])}")
    print(f"  Answer Relevancy  {scores['answer_relevancy']:.4f}  {interpret(scores['answer_relevancy'])}")
    print(f"  Context Recall    {scores['context_recall']:.4f}  {interpret(scores['context_recall'])}")
    print("=" * 55)

    report = {
        "timestamp":       datetime.utcnow().isoformat(),
        "model":           "mistral (local via Ollama)",
        "scorer":          "LLM-as-judge (synchronous)",
        "scores":          scores,
        "interpretations": {k: interpret(v) for k, v in scores.items()},
        "num_test_cases":  len(results),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    print("\nResearchMind — Evaluation Suite (Ollama)")
    print("=" * 55)

    test_cases = load_dataset()
    results    = run_pipeline(test_cases)
    scores     = score_all(results)
    save_report(scores, results)