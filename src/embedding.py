"""
Task 2: Embedding + vector store indexing.

We use sentence-transformers/all-MiniLM-L6-v2 because it's the right
trade-off for this project:
- 384-dim vectors, ~80MB model -> fast to embed 1M+ chunks on CPU
- strong performance on semantic similarity benchmarks relative to its size
- widely supported by both ChromaDB and FAISS, so the pipeline isn't locked in

ChromaDB is used as the vector store because it persists to disk out of the
box and stores metadata alongside vectors natively (no separate ID-mapping
file to keep in sync, which you'd need with raw FAISS).
"""
import logging
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

from src import config

logger = logging.getLogger(__name__)

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def get_chroma_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


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
