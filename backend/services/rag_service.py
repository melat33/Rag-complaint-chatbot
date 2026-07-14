"""
Thin service layer between the FastAPI routes and src/rag_pipeline.py.

Keeping this separation means src/ stays a standalone, importable, testable
library (usable from notebooks, scripts, or a CLI) with zero FastAPI
dependency, while backend/ owns all the "web API" concerns: request
validation, HTTP status codes, response shaping.
"""
import logging
from functools import lru_cache

from src.rag_pipeline import RAGPipeline
from src.embedding import get_chroma_collection
from backend.schemas import AskRequest, AskResponse, SourceChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """
    Load the pipeline once per process (embedding model + Chroma client are
    expensive to initialize) and reuse it across requests.
    """
    logger.info("Initializing RAG pipeline (embedding model + vector store)...")
    return RAGPipeline()


def _flatten_source(s: dict) -> dict:
    """
    Lift metadata fields (complaint_id, product_category, issue) up to the
    top level of the source dict. The raw retriever output nests these under
    "metadata", but the frontend's SourcesPanel expects them flat -- this is
    the single place that shape conversion happens, so both the sync and
    streaming endpoints stay consistent.
    """
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
    """
    Yields SSE-ready dicts from the pipeline's streaming generator, with
    source chunks flattened to match the same shape ask() returns above --
    otherwise the frontend reads undefined for complaint_id/product_category/issue.
    """
    pipeline = get_pipeline()
    for event in pipeline.ask_stream(request.question, product_filter=request.product_filter):
        if event["type"] == "sources":
            yield {"type": "sources", "sources": [_flatten_source(s) for s in event["sources"]]}
        else:
            yield event


def health_check() -> dict:
    collection = get_chroma_collection()
    return {"status": "ok", "collection_count": collection.count()}