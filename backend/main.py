"""
FastAPI app exposing the RAG pipeline to the React frontend.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /health        -> liveness + vector store sanity check
    POST /ask            -> single-shot question, full answer + sources
    POST /ask/stream      -> Server-Sent Events, tokens streamed as generated
"""
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.schemas import AskRequest, AskResponse, HealthResponse
from backend.services import rag_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CrediTrust Complaint Intelligence API",
    description="RAG-powered API for querying customer complaint trends.",
    version="1.0.0",
)

# In production, replace "*" with the deployed frontend's exact origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        result = rag_service.health_check()
        return HealthResponse(status=result["status"], collection_count=result["collection_count"])
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        return rag_service.ask(request)
    except RuntimeError as e:
        # e.g. missing GROQ_API_KEY
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("RAG pipeline error")
        raise HTTPException(status_code=500, detail="Something went wrong answering your question.")


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    def event_generator():
        try:
            for event in rag_service.ask_stream(request):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception("Streaming RAG pipeline error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
