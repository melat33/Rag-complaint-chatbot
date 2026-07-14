"""
Task 3: RAG pipeline orchestration.

This is the single class the backend (and evaluation scripts) call. It ties
retrieval + generation together and returns both the answer and the sources,
because showing sources is a hard requirement for user trust (Task 4).
"""
import logging
from typing import Optional

from src.retriever import Retriever
from src.generator import generate, generate_stream

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, top_k: int = None):
        self.retriever = Retriever(top_k=top_k)

    def ask(self, question: str, product_filter: Optional[str] = None) -> dict:
        """
        Full RAG turn: retrieve -> generate -> return answer + sources.

        Returns:
            {
              "answer": str,
              "sources": [{"text": str, "metadata": dict, "distance": float}, ...]
            }
        """
        chunks = self.retriever.retrieve(question, product_filter=product_filter)

        if not chunks:
            return {
                "answer": "I don't have enough information in the complaint data to answer that.",
                "sources": [],
            }

        context_texts = [c["text"] for c in chunks]
        answer = generate(question, context_texts)

        return {"answer": answer, "sources": chunks}

    def ask_stream(self, question: str, product_filter: Optional[str] = None):
        """
        Streaming variant for the chat UI. Yields dicts:
            {"type": "sources", "sources": [...]}   -- sent once, first
            {"type": "token", "token": str}          -- sent repeatedly
            {"type": "done"}                         -- sent once, last
        """
        chunks = self.retriever.retrieve(question, product_filter=product_filter)

        if not chunks:
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "token": "I don't have enough information in the complaint data to answer that."}
            yield {"type": "done"}
            return

        yield {"type": "sources", "sources": chunks}

        context_texts = [c["text"] for c in chunks]
        for token in generate_stream(question, context_texts):
            yield {"type": "token", "token": token}

        yield {"type": "done"}
