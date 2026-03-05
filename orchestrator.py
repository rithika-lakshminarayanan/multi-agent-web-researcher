"""Main multi-agent orchestration logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from agents.browser_agent import open_page
from agents.critic_agent import critic_agent
from agents.planner_agent import planner_agent
from agents.summarizer_agent import summarize_agent
from tools.llm_client import LLMError, llm_complete
from tools.search import search


@dataclass
class RunTrace:
    sources: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)
    reflections: int = 0


def _compose_answer(query: str, notes: List[str], sources: List[str]) -> str:
    context = "\n\n".join(notes[:8]) if notes else "No evidence collected."
    prompt = f"""
You are the final synthesis agent.
Produce a structured final answer with sections:
1) Direct Answer
2) Key Evidence
3) Caveats and Uncertainty
4) Sources Used

Question: {query}

Evidence notes:
{context}

Sources:
{sources}
""".strip()
    try:
        return llm_complete(prompt)
    except LLMError:
        return (
            "1) Direct Answer\n"
            "Unable to generate a high-quality synthesis because the local LLM endpoint is unavailable.\n\n"
            "2) Key Evidence\n"
            f"Collected {len(notes)} evidence note(s).\n\n"
            "3) Caveats and Uncertainty\n"
            "Quality may be limited without model-based synthesis.\n\n"
            "4) Sources Used\n"
            + "\n".join(f"- {s}" for s in sources)
        )


def _search_and_summarize(query: str, trace: RunTrace, use_browser: bool) -> None:
    planned_queries = planner_agent(query)
    for planned in planned_queries:
        try:
            urls = search(planned, top_k=3)
        except Exception:
            urls = []

        for url in urls:
            if url in trace.sources:
                continue
            trace.sources.add(url)
            if not use_browser:
                trace.notes.append(f"Query '{planned}' returned source: {url}")
                continue
            try:
                text = open_page(url)
            except Exception:
                continue
            summary = summarize_agent(query, text)
            trace.notes.append(f"Source: {url}\n{summary}")


def run_agent(
    query: str,
    mode: str = "full",
    reflection_threshold: int = 7,
    max_reflections: int = 2,
) -> Tuple[str, Dict[str, object]]:
    """
    Run the research agent pipeline.

    Modes:
    - planner_only: no browsing/summarization; uses planned queries + URLs only
    - planner_browser: browsing enabled, no critic reflection loop
    - full: all agents + reflection loop
    """
    trace = RunTrace()

    if mode == "planner_only":
        _search_and_summarize(query, trace, use_browser=False)
        answer = _compose_answer(query, trace.notes, sorted(trace.sources))
        review = critic_agent(answer)
        review.update(
            {
                "num_sources": len(trace.sources),
                "num_reflections": trace.reflections,
                "mode": mode,
            }
        )
        return answer, review

    if mode in {"planner_browser", "full"}:
        _search_and_summarize(query, trace, use_browser=True)
    else:
        raise ValueError("mode must be one of: planner_only, planner_browser, full")

    answer = _compose_answer(query, trace.notes, sorted(trace.sources))
    review = critic_agent(answer)

    if mode == "full":
        while (
            int(review.get("score", 0)) < reflection_threshold
            and trace.reflections < max_reflections
        ):
            trace.reflections += 1
            reflection_query = (
                f"{query}. Improve based on critic feedback: {review.get('review', '')}"
            )
            _search_and_summarize(reflection_query, trace, use_browser=True)
            answer = _compose_answer(query, trace.notes, sorted(trace.sources))
            review = critic_agent(answer)

    review.update(
        {
            "num_sources": len(trace.sources),
            "num_reflections": trace.reflections,
            "mode": mode,
        }
    )
    return answer, review


if __name__ == "__main__":
    sample_q = "What are the key methods for reducing hallucinations in LLMs?"
    final_answer, final_review = run_agent(sample_q, mode="full")
    print(final_answer)
    print("\n---\n")
    print(final_review)
