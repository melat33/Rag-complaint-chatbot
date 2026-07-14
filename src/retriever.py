"""
Task 3: Retriever.

Embeds a user question with the same model used at indexing time (this
consistency matters - a mismatched embedding model silently degrades
retrieval quality) and runs similarity search against ChromaDB, optionally
filtered by product category.
"""
import logging
from typing import List, Dict, Optional

from src import config
from src.embedding import get_embedding_model, get_chroma_collection

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, top_k: int = None):
        self.top_k = top_k or config.TOP_K
        self.model = get_embedding_model()
        self.collection = get_chroma_collection()

    def retrieve(self, question: str, product_filter: Optional[str] = None) -> List[Dict]:
        """
        Return the top-k most relevant chunks for `question`.
        Each result: {"text": ..., "metadata": {...}, "distance": float}
        """
        query_embedding = self.model.encode([question]).tolist()

        where = {"product_category": product_filter} if product_filter else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
            where=where,
        )

        chunks = []
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append({"text": text, "metadata": meta, "distance": dist})

        logger.info(f"Retrieved {len(chunks)} chunks for question: {question[:60]!r}")
        return chunks
