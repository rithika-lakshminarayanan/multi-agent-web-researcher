"""HTML cleaning and extraction helpers."""

from __future__ import annotations

import re
from bs4 import BeautifulSoup


def clean_html_text(html: str, max_chars: int = 12000) -> str:
    """Strip scripts/styles/navigation noise and return compact text."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "svg", "footer", "header", "nav", "form"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]
