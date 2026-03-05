"""Browser agent using Playwright to fetch rendered page content."""

from __future__ import annotations

from playwright.sync_api import sync_playwright

from tools.scraper import clean_html_text


def open_page(url: str, timeout_ms: int = 30000) -> str:
    """Open a URL and return cleaned textual content from fully rendered HTML."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        html = page.content()
        browser.close()
    return clean_html_text(html)
