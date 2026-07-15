"""
ChromaDB collection access, deliberately with ZERO dependency on
sentence-transformers/torch, so the deployed backend (src/retriever.py,
backend/services/rag_service.py) can import this without pulling in the
heavy local embedding stack. src/embedding.py (used only for offline bulk
indexing on your own machine) re-exports from here too, so there's one
single source of truth for how the Chroma client is created.
"""
import chromadb
from src import config


def get_chroma_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )