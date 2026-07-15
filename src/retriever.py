"""
Task 3: Retriever.

Uses the Hugging Face Inference API (src/query_embedding.py) to embed the
user's question, rather than loading the sentence-transformers model
locally -- this keeps the DEPLOYED backend's memory footprint small enough
to run on Render's free tier. The index itself was still built locally with
the full sentence-transformers model (src/build_index.py, run on your own
machine), so retrieval quality is unaffected -- both paths use the exact
same model, just at different points in the pipeline.
"""
import logging
from typing import List, Dict, Optional

from src import config
from src.embedding import get_chroma_collection
from src.query_embedding import embed_query

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, top_k: int = None):
        self.top_k = top_k or config.TOP_K
        self.collection = get_chroma_collection()

    def retrieve(self, question: str, product_filter: Optional[str] = None) -> List[Dict]:
        """
        Return the top-k most relevant chunks for `question`.
        Each result: {"text": ..., "metadata": {...}, "distance": float}
        """
        query_embedding = embed_query(question)

        where = {"product_category": product_filter} if product_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
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