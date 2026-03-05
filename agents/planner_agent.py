"""Planner agent: derive targeted search queries from a user question."""

from __future__ import annotations

from typing import List

from tools.llm_client import LLMError, llm_complete


def planner_agent(query: str) -> List[str]:
    """Generate exactly three web-search queries for a research question."""
    prompt = f"""
You are a research planning agent.
Given the user question, produce exactly 3 diverse web search queries.
Each query should target a different angle (definition, evidence, comparison/policy where relevant).
Return ONLY 3 lines, one query per line, no numbering.

User question: {query}
""".strip()

    try:
        raw = llm_complete(prompt)
        lines = [line.strip(" -\t") for line in raw.splitlines() if line.strip()]
        queries = [line for line in lines if len(line.split()) >= 3][:3]
        if len(queries) == 3:
            return queries
    except LLMError:
        pass

    # Fallback keeps system operational when local LLM is unavailable.
    return [
        query,
        f"{query} recent evidence",
        f"{query} comparison",
    ]
