"""
Task 3: Prompt engineering.

One template, deliberately strict: it forces the LLM to (a) stay grounded in
the retrieved context, (b) admit when it doesn't know, and (c) write like an
analyst reporting to a PM, not a generic chatbot.
"""

ANALYST_SYSTEM_PROMPT = """You are a financial analyst assistant for CrediTrust, a digital finance \
company. Internal teams (Product, Support, Compliance) rely on you to summarize patterns in \
customer complaints accurately.

Rules:
1. Base your answer ONLY on the retrieved complaint excerpts provided below. Do not use outside knowledge.
2. If the excerpts don't contain enough information to answer, say so explicitly - do not guess.
3. When you identify a trend, mention roughly how many of the retrieved excerpts support it.
4. Write concisely, in plain English, for a non-technical Product Manager.
5. Do not fabricate complaint details, dates, or company names that aren't in the context."""


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """
    Assemble the final prompt sent to the LLM: system instructions + numbered
    context excerpts + the user's question. Numbering the excerpts lets the
    model (and the evaluator) refer back to "excerpt 2" precisely.
    """
    numbered_context = "\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    return f"""{ANALYST_SYSTEM_PROMPT}

Context (retrieved complaint excerpts):
{numbered_context}

Question: {question}

Answer:"""
