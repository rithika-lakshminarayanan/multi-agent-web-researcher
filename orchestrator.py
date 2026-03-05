"""Main multi-agent orchestration logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from agents.browser_agent import open_page
from agents.critic_agent import critic_agent
from agents.multihop_reasoner import decompose_question, multihop_reason, validate_chain
from agents.planner_agent import planner_agent
from agents.summarizer_agent import summarize_agent
from tools.llm_client import LLMError, llm_complete
from tools.search import search
from tools.vector_memory import format_memory_context, retrieve_memory, save_memory


@dataclass
class RunTrace:
    sources: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)
    reflections: int = 0
    multihop_enabled: bool = False
    sub_questions: List[str] = field(default_factory=list)
    sub_answers: List[str] = field(default_factory=list)


def _compose_answer(
    query: str,
    notes: List[str],
    sources: List[str],
    memory_context: str = "",
) -> str:
    context = "\n\n".join(notes[:8]) if notes else "No evidence collected."
    memory_section = memory_context if memory_context else "No prior memory context."
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

Related memory from prior runs:
{memory_section}

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


def _multihop_search_and_summarize(
    sub_questions: List[str], trace: RunTrace, use_browser: bool
) -> List[str]:
    """
    Search and summarize for each sub-question in a multi-hop chain.
    Returns list of answers corresponding to sub-questions.
    """
    sub_answers = []
    for i, sub_q in enumerate(sub_questions):
        sub_trace_notes = []
        planned_queries = planner_agent(sub_q)
        
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
                    sub_trace_notes.append(f"Query '{planned}' returned: {url}")
                    continue
                try:
                    text = open_page(url)
                except Exception:
                    continue
                summary = summarize_agent(sub_q, text)
                sub_trace_notes.append(f"Source: {url}\n{summary}")
        
        # Compose sub-answer
        sub_answer = _compose_answer(
            sub_q,
            sub_trace_notes,
            sorted(list(trace.sources)[-3:]),  # Last 3 sources
            memory_context="",
        )
        sub_answers.append(sub_answer)
        trace.notes.append(f"[Sub-question {i+1}] {sub_q}\n{sub_answer}\n")
    
    return sub_answers


def run_agent(
    query: str,
    mode: str = "full",
    reflection_threshold: int = 7,
    max_reflections: int = 2,
    use_memory: bool = True,
    memory_top_k: int = 3,
    memory_min_score: float = 0.2,
    enable_multihop: bool = True,
) -> Tuple[str, Dict[str, object]]:
    """
    Run the research agent pipeline.

    Modes:
    - planner_only: no browsing/summarization; uses planned queries + URLs only
    - planner_browser: browsing enabled, no critic reflection loop
    - full: all agents + reflection loop
    
    Args:
        query: research question
        mode: execution mode
        reflection_threshold: critic score threshold for reflection (1-10)
        max_reflections: maximum reflection iterations
        use_memory: use vector memory for context
        memory_top_k: retrieve top-k memory hits
        memory_min_score: minimum similarity threshold for memory
        enable_multihop: enable multi-hop reasoning for complex questions
    """
    trace = RunTrace()
    memory_hits = (
        retrieve_memory(query, top_k=memory_top_k, min_score=memory_min_score)
        if use_memory
        else []
    )
    memory_context = format_memory_context(memory_hits)

    # Analyze if question needs multi-hop reasoning
    analysis = decompose_question(query) if enable_multihop else {}
    requires_multihop = analysis.get("requires_multihop", False)
    
    if requires_multihop and mode in {"planner_browser", "full"}:
        # Multi-hop reasoning path
        trace.multihop_enabled = True
        trace.sub_questions = analysis.get("sub_questions", [])
        
        # Search and answer each sub-question
        trace.sub_answers = _multihop_search_and_summarize(
            trace.sub_questions, trace, use_browser=True
        )
        
        # Synthesize multi-hop answer
        multihop_answer = multihop_reason(query, trace.sub_answers)
        
        # Compose final answer incorporating multi-hop synthesis
        answer = _compose_answer(
            query,
            trace.notes + [f"Multi-hop synthesis:\n{multihop_answer}"],
            sorted(trace.sources),
            memory_context=memory_context,
        )
        
        # Validate reasoning chain
        chain_validation = validate_chain(query, trace.sub_questions, trace.sub_answers)
        
    else:
        # Standard single-hop reasoning path
        if mode == "planner_only":
            _search_and_summarize(query, trace, use_browser=False)
            answer = _compose_answer(
                query,
                trace.notes,
                sorted(trace.sources),
                memory_context=memory_context,
            )
            review = critic_agent(answer)
            if use_memory:
                save_memory(query, answer, trace.notes, sorted(trace.sources))
            review.update(
                {
                    "num_sources": len(trace.sources),
                    "num_reflections": trace.reflections,
                    "mode": mode,
                    "num_memory_hits": len(memory_hits),
                    "multihop_enabled": trace.multihop_enabled,
                }
            )
            return answer, review

        if mode in {"planner_browser", "full"}:
            _search_and_summarize(query, trace, use_browser=True)
        else:
            raise ValueError("mode must be one of: planner_only, planner_browser, full")

        answer = _compose_answer(
            query,
            trace.notes,
            sorted(trace.sources),
            memory_context=memory_context,
        )
        chain_validation = {}

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
            answer = _compose_answer(
                query,
                trace.notes,
                sorted(trace.sources),
                memory_context=memory_context,
            )
            review = critic_agent(answer)

    if use_memory:
        save_memory(query, answer, trace.notes, sorted(trace.sources))

    review.update(
        {
            "num_sources": len(trace.sources),
            "num_reflections": trace.reflections,
            "mode": mode,
            "num_memory_hits": len(memory_hits),
            "multihop_enabled": trace.multihop_enabled,
            "sub_questions": trace.sub_questions,
            "chain_validation": chain_validation,
            "difficulty_analysis": analysis,
        }
    )
    return answer, review


if __name__ == "__main__":
    sample_q = "What are the key methods for reducing hallucinations in LLMs?"
    final_answer, final_review = run_agent(sample_q, mode="full")
    print(final_answer)
    print("\n---\n")
    print(final_review)
