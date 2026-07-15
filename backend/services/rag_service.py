"""
Thin service layer between the FastAPI routes and src/rag_pipeline.py.
"""
import logging
from functools import lru_cache

from src.rag_pipeline import RAGPipeline
from src.vector_store import get_chroma_collection
from backend.schemas import AskRequest, AskResponse, SourceChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    logger.info("Initializing RAG pipeline (query API + vector store)...")
    return RAGPipeline()


def _flatten_source(s: dict) -> dict:
    return {
        "text": s["text"],
        "complaint_id": str(s["metadata"].get("complaint_id", "")),
        "product_category": s["metadata"].get("product_category"),
        "issue": s["metadata"].get("issue"),
        "distance": s["distance"],
    }


def ask(request: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    result = pipeline.ask(request.question, product_filter=request.product_filter)
    sources = [SourceChunk(**_flatten_source(s)) for s in result["sources"]]
    return AskResponse(answer=result["answer"], sources=sources)


def ask_stream(request: AskRequest):
    pipeline = get_pipeline()
    for event in pipeline.ask_stream(request.question, product_filter=request.product_filter):
        if event["type"] == "sources":
            yield {"type": "sources", "sources": [_flatten_source(s) for s in event["sources"]]}
        else:
            yield event


def health_check() -> dict:
    collection = get_chroma_collection()
    return {"status": "ok", "collection_count": collection.count()}