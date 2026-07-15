"""
Task 2: Embedding + vector store indexing.

Used for OFFLINE bulk indexing only (src/build_index.py, run on your own
machine) -- loads sentence-transformers/torch locally, which is fine here
since this never runs on the memory-constrained deployed backend. The
deployed backend's query-time embedding uses src/query_embedding.py instead
(calls a hosted API, no torch needed).

We use sentence-transformers/all-MiniLM-L6-v2 because it's the right
trade-off for this project:
- 384-dim vectors, ~80MB model -> fast to embed 1M+ chunks on CPU
- strong performance on semantic similarity benchmarks relative to its size
- widely supported by both ChromaDB and FAISS, so the pipeline isn't locked in
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
from typing import List, Dict

from sentence_transformers import SentenceTransformer

from src import config
from src.vector_store import get_chroma_collection  # noqa: F401  (re-exported for backward compatibility)

logger = logging.getLogger(__name__)

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_and_index(chunk_records: List[Dict], batch_size: int = 256) -> None:
    """
    Embed every chunk and upsert it into ChromaDB with its metadata.
    Batched to keep memory bounded when indexing hundreds of thousands
    of chunks.
    """
    model = get_embedding_model()
    collection = get_chroma_collection()

    for start in range(0, len(chunk_records), batch_size):
        batch = chunk_records[start:start + batch_size]
        texts = [r["chunk_text"] for r in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        ids = [f"{r['complaint_id']}_{r['chunk_index']}" for r in batch]
        metadatas = [{k: ("" if v is None else v) for k, v in r.items() if k != "chunk_text"}
                     for r in batch]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {min(start + batch_size, len(chunk_records)):,}/{len(chunk_records):,} chunks")

    logger.info(f"Indexing complete. Collection count: {collection.count():,}")