"""Minimal Ollama-compatible LLM client with deterministic fallbacks."""

from __future__ import annotations

import json
from typing import Any, Dict

import requests

DEFAULT_MODEL = "phi"
DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"


class LLMError(RuntimeError):
    """Raised when the local LLM backend is not reachable or returns invalid data."""


def llm_complete(
    prompt: str,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: int = 120,
) -> str:
    """Call an Ollama-compatible generation endpoint and return plain text response."""
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("LLM response was not valid JSON.") from exc

    text = data.get("response", "")
    if not isinstance(text, str):
        raise LLMError("LLM response missing text field 'response'.")
    return text.strip()
