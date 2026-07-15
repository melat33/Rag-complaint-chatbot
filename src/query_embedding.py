"""
Lightweight, API-based embedding for the DEPLOYED backend's query path.

Uses Hugging Face's official InferenceClient rather than hand-rolled HTTP
calls to a raw URL -- HF has restructured their Inference API more than
once, and the client library abstracts over exactly which endpoint/provider
is current, which is far more durable than us guessing at URLs.
"""
import logging
import numpy as np
from huggingface_hub import InferenceClient

from src import config

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_client = None


def get_client() -> InferenceClient:
    global _client
    if _client is None:
        if not config.HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is not set. Add it to your .env file (see .env.example)."
            )
        _client = InferenceClient(provider="hf-inference", api_key=config.HF_TOKEN)
    return _client


def embed_query(text: str) -> list[float]:
    """Embed a single query string via Hugging Face's Inference API."""
    client = get_client()
    result = np.array(client.feature_extraction(text, model=EMBEDDING_MODEL_ID))

    if result.ndim == 1:
        # Already a single pooled sentence vector (384,)
        embedding = result
    else:
        # Per-token vectors (num_tokens, 384) -- mean-pool to match how
        # sentence-transformers produced the vectors used to build the index.
        embedding = result.mean(axis=0)

    return embedding.tolist()