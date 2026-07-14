"""
Task 3: Generator.

LLM choice: Groq's Llama 3.3 70B Versatile.
Rationale (documented for the report):
- Groq's inference is extremely fast (LPU hardware), which matters for a
  chat UI where Asha expects an answer in seconds, not "let me get back to you".
- Free tier is generous enough for a portfolio/demo deployment - no
  per-request billing risk if the project gets shared publicly.
- 70B-class open-weight model gives strong instruction-following for the
  "stay grounded in context, admit uncertainty" behavior the prompt requires.
- Swappable: `generate()` is the only function that touches the LLM SDK, so
  switching to OpenAI/Anthropic later is a one-function change.
"""
import logging
from typing import List

from groq import Groq

from src import config
from src.prompt_templates import build_prompt

logger = logging.getLogger(__name__)

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def generate(question: str, context_chunks: List[str]) -> str:
    """Send the assembled prompt to the LLM and return its text answer."""
    prompt = build_prompt(question, context_chunks)
    client = get_client()

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def generate_stream(question: str, context_chunks: List[str]):
    """Same as generate(), but yields tokens as they arrive for streaming UIs."""
    prompt = build_prompt(question, context_chunks)
    client = get_client()

    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
