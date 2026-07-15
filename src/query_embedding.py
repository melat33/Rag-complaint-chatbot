"""
Lightweight, API-based embedding for the DEPLOYED backend's query path.

Why this exists separately from src/embedding.py: that module loads torch +
sentence-transformers locally, which is fine for offline bulk indexing on
your own machine (src/build_index.py), but far too heavy (300-400MB+) for a
memory-constrained deployment that only ever needs to embed ONE short
question at a time. This module calls Hugging Face's hosted Inference API
for that single embedding instead, using the exact same model
(all-MiniLM-L6-v2) so vectors stay compatible with an index that was built
locally with sentence-transformers.
"""
import logging
import requests

from src import config

logger = logging.getLogger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def embed_query(text: str) -> list[float]:
    """Embed a single query string via Hugging Face's Inference API."""
    if not config.HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to your .env file (see .env.example)."
        )

    response = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {config.HF_TOKEN}"},
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30,
    )
    response.raise_for_status()
    embedding = response.json()

    if isinstance(embedding[0], list):
        embedding = embedding[0]
    return embedding