"""Multi-hop reasoning agent: decompose complex questions into reasoning chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from tools.llm_client import LLMError, llm_complete


@dataclass
class ReasoningHop:
    """Represents a single reasoning step in a multi-hop chain."""

    step_num: int
    sub_question: str
    reasoning: str
    evidence: str = ""


def decompose_question(query: str) -> Dict[str, object]:
    """
    Analyze a question to determine if multi-hop reasoning is needed.
    Returns: {
        'requires_multihop': bool,
        'sub_questions': List[str],
        'reasoning_chain': str,
        'difficulty_level': str  # 'simple', 'moderate', 'complex'
    }
    """
    prompt = f"""
You are a research question analyzer.
Analyze whether the following question requires multi-hop reasoning (multiple reasoning steps with information chaining).

Classification rules:
- Simple: single-fact lookup (e.g., "What is X?", "Who discovered Y?")
- Moderate: requires 2 decomposition steps (e.g., "How does X affect Y?")
- Complex: requires 3+ chained reasoning steps (e.g., "Why does X cause Y, and how does that impact Z?")

For complex/moderate questions, decompose into 2-4 focused sub-questions.

Question: {query}

Respond in this format:
Difficulty: <simple|moderate|complex>
Requires Multihop: <yes|no>
Sub-questions:
1. <first sub-question>
2. <second sub-question>
...
Reasoning Chain: <brief explanation of how sub-answers chain together>
""".strip()

    try:
        raw = llm_complete(prompt)
        
        # Parse response
        difficulty = "simple"
        requires_multihop = False
        sub_questions = []
        reasoning_chain = ""
        
        lines = raw.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Difficulty:"):
                difficulty = line.split(":", 1)[1].strip().lower()
                requires_multihop = difficulty in ["moderate", "complex"]
            elif line.startswith("Requires Multihop:"):
                requires_multihop = "yes" in line.lower()
            elif line.startswith(("1.", "2.", "3.", "4.")):
                sub_q = line.split(".", 1)[1].strip()
                if sub_q:
                    sub_questions.append(sub_q)
            elif line.startswith("Reasoning Chain:"):
                reasoning_chain = line.split(":", 1)[1].strip()
        
        return {
            "requires_multihop": requires_multihop,
            "sub_questions": sub_questions[:4],  # Max 4 sub-questions
            "reasoning_chain": reasoning_chain,
            "difficulty_level": difficulty,
        }
    except LLMError:
        # Fallback for simple questions
        return {
            "requires_multihop": False,
            "sub_questions": [query],
            "reasoning_chain": "LLM unavailable; treating as simple query.",
            "difficulty_level": "simple",
        }


def multihop_reason(
    query: str,
    sub_answers: List[str],
) -> str:
    """
    Chain together answers from multiple sub-questions into a cohesive explanation.
    
    Args:
        query: original user question
        sub_answers: list of answers to sub-questions, in order
    
    Returns:
        Synthesized answer that chains reasoning across hops
    """
    if not sub_answers:
        return "No evidence available for multi-hop reasoning."
    
    sub_answer_text = "\n".join(
        f"Sub-answer {i+1}: {ans}" for i, ans in enumerate(sub_answers)
    )
    
    prompt = f"""
You are a research synthesis agent specializing in multi-hop reasoning.
Given a complex question and answers to its decomposed sub-questions, synthesize a coherent answer that chains the reasoning together.

Original Question: {query}

Sub-question Answers:
{sub_answer_text}

Task: Produce a unified answer that shows HOW the sub-answers logically connect and build on each other to address the original question.
Structure your response as:
1) Main Reasoning Chain: [explain how sub-answers build on each other]
2) Integrated Answer: [synthesized answer to the original question]
3) Key Connections: [bullet points linking sub-answers]
""".strip()

    try:
        return llm_complete(prompt)
    except LLMError:
        return (
            "Unable to synthesize multi-hop reasoning due to LLM unavailability. "
            "Raw sub-answers:\n" + sub_answer_text
        )


def validate_chain(
    original_query: str,
    sub_questions: List[str],
    sub_answers: List[str],
) -> Dict[str, object]:
    """
    Validate that the reasoning chain is logically sound and complete.
    
    Returns: {
        'is_valid': bool,
        'gaps': List[str],
        'suggestions': List[str],
        'confidence': float  # 0.0-1.0
    }
    """
    if len(sub_questions) != len(sub_answers):
        return {
            "is_valid": False,
            "gaps": ["Mismatch between sub-questions and answers"],
            "suggestions": ["Collect missing answers"],
            "confidence": 0.0,
        }
    
    sub_qa_text = "\n".join(
        f"Q{i+1}: {q}\nA{i+1}: {a}"
        for i, (q, a) in enumerate(zip(sub_questions, sub_answers))
    )
    
    prompt = f"""
You are a research quality auditor.
Validate whether the chain of reasoning is logically sound and answers the original question completely.

Original Question: {original_query}

Sub-Q&A Chain:
{sub_qa_text}

Assess:
1. Are there logical gaps in the reasoning chain?
2. Are important aspects of the original question left unanswered?
3. Is the evidence sufficient to support the conclusions?

Respond in format:
Valid: <yes|no>
Confidence: <0.0-1.0>
Gaps: <list gaps, or "None">
Suggestions: <improvement suggestions>
""".strip()

    try:
        raw = llm_complete(prompt)
        
        is_valid = "yes" in raw.lower().split("Valid:")[1].split("\n")[0] if "Valid:" in raw else False
        
        confidence = 0.5
        if "Confidence:" in raw:
            try:
                conf_str = raw.split("Confidence:")[1].split("\n")[0].strip()
                confidence = float(conf_str)
            except (ValueError, IndexError):
                pass
        
        gaps = []
        suggestions = []
        
        if "Gaps:" in raw:
            gaps_section = raw.split("Gaps:")[1].split("Suggestions:")[0].strip()
            gaps = [g.strip() for g in gaps_section.split("\n") if g.strip() and g.strip() != "None"]
        
        if "Suggestions:" in raw:
            sugg_section = raw.split("Suggestions:")[1].strip()
            suggestions = [s.strip() for s in sugg_section.split("\n") if s.strip()]
        
        return {
            "is_valid": is_valid,
            "gaps": gaps,
            "suggestions": suggestions,
            "confidence": confidence,
        }
    except LLMError:
        return {
            "is_valid": True,  # Assume valid if LLM unavailable
            "gaps": [],
            "suggestions": [],
            "confidence": 0.5,
        }
