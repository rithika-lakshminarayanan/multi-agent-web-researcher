"""Critic agent: evaluate answer quality and determine reflection need."""

from __future__ import annotations

import re
from typing import Dict

from tools.llm_client import LLMError, llm_complete


def critic_agent(answer: str) -> Dict[str, object]:
    """Return score (1-10) and short review of answer quality."""
    prompt = f"""
You are a strict research evaluator.
Score the answer from 1 to 10 on factuality, completeness, clarity, and use of evidence.
Return exactly this format:
Score: <integer 1-10>
Review: <2-4 sentence critique with specific improvement advice>

Answer:
{answer}
""".strip()

    try:
        raw = llm_complete(prompt)
        match = re.search(r"Score:\s*(\d+)", raw)
        score = int(match.group(1)) if match else 6
        score = max(1, min(10, score))
        review_match = re.search(r"Review:\s*(.*)", raw, flags=re.DOTALL)
        review = review_match.group(1).strip() if review_match else raw.strip()
        return {"score": score, "review": review}
    except LLMError:
        return {
            "score": 6,
            "review": "LLM critic unavailable. Conservative default score assigned.",
        }
