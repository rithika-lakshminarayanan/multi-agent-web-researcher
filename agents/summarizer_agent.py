"""Summarizer agent for extracting relevant evidence from page text."""

from __future__ import annotations

from tools.llm_client import LLMError, llm_complete


def summarize_agent(query: str, text: str) -> str:
    """Summarize extracted page text relative to the user query."""
    prompt = f"""
You are a scientific web research assistant.
Given the user question and page text, extract concise evidence-focused notes.
Prioritize factual claims, numbers, dates, and caveats.
Return 4-8 bullet points.

User question:
{query}

Page text:
{text}
""".strip()

    try:
        return llm_complete(prompt)
    except LLMError:
        fallback = text[:800]
        return f"- Fallback summary (LLM unavailable): {fallback}"
