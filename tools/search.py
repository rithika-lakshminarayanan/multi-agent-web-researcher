"""Web search utilities using DuckDuckGo HTML results."""

from __future__ import annotations

from typing import List
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

DUCKDUCKGO_HTML = "https://duckduckgo.com/html/?q={query}"


def search(query: str, top_k: int = 3, timeout: int = 20) -> List[str]:
    """Return top result URLs from DuckDuckGo's HTML endpoint."""
    url = DUCKDUCKGO_HTML.format(query=quote_plus(query))
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    links = []
    for a_tag in soup.select("a.result__a"):
        href = a_tag.get("href")
        if not href:
            continue
        if href.startswith("http"):
            links.append(href)
        if len(links) >= top_k:
            break
    return links
