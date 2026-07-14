"""
Task 3: RAG pipeline demo + qualitative evaluation.

Runs the full retriever -> prompt -> LLM pipeline against 10 representative
questions, prints each answer with its sources, and exports the evaluation
table to evaluation/eval_questions.md.

Usage:
    python -m src.run_rag_demo
"""
import logging

from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EVAL_QUESTIONS = [
    "Why are people unhappy with Credit Cards?",
    "What are the most common complaints about Personal Loans?",
    "Are customers reporting unauthorized transactions in Money Transfers?",
    "What issues come up most with Savings Accounts?",
    "Do complaints mention long resolution times?",
    "Are there complaints about hidden fees across any product?",
    "What fraud-related patterns appear in the complaints?",
    "How do customers describe problems with customer service?",
    "Are there recurring complaints about a specific company?",
    "What would you flag as the top 3 emerging issues this month?",
]


def main():
    pipeline = RAGPipeline()

    print("=" * 80)
    print("SINGLE QUESTION TEST")
    print("=" * 80)
    result = pipeline.ask("Why are people unhappy with Credit Cards?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nGrounded in {len(result['sources'])} sources:")
    for s in result["sources"][:2]:
        print(f"  #{s['metadata']['complaint_id']}: {s['text'][:120]}...")

    print("\n" + "=" * 80)
    print("FULL EVALUATION — 10 QUESTIONS")
    print("=" * 80)

    rows = []
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        logger.info(f"[{i}/{len(EVAL_QUESTIONS)}] {q}")
        result = pipeline.ask(q)
        top_sources = "; ".join(
            f"#{s['metadata']['complaint_id']}: {s['text'][:100]}..."
            for s in result["sources"][:2]
        )
        rows.append({
            "question": q,
            "answer": result["answer"],
            "sources": top_sources,
        })
        print(f"\nQ{i}: {q}")
        print(f"A: {result['answer']}")
        print("-" * 80)

    # Export to markdown for the report
    lines = ["# RAG Pipeline — Qualitative Evaluation\n"]
    lines.append("| # | Question | Generated Answer | Retrieved Sources (top 2) | Quality Score | Comments |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        answer_short = r["answer"].replace("\n", " ").replace("|", "-")
        sources_short = r["sources"].replace("\n", " ").replace("|", "-")
        lines.append(f"| {i} | {r['question']} | {answer_short} | {sources_short} | | |")

    with open("evaluation/eval_questions.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved evaluation table to evaluation/eval_questions.md")
    logger.info("Open it and fill in Quality Score (1-5) + Comments for each row.")


if __name__ == "__main__":
    main()