"""Web search fallback. Swappable: DuckDuckGo (no key), Tavily (TAVILY_API_KEY), or disabled."""

from __future__ import annotations

import os
from typing import Protocol

from crag.store import Doc


class WebSearch(Protocol):
    def search(self, query: str, n: int) -> list[Doc]: ...


class DuckDuckGoSearch:
    def search(self, query: str, n: int) -> list[Doc]:
        from ddgs import DDGS

        try:
            hits = DDGS().text(query, max_results=n)
        except Exception as exc:  # rate limits / network: degrade to "no web evidence", never crash
            print(f"[websearch] ddgs failed: {exc}")
            return []
        return [
            Doc(id=f"web{i}", text=f"{h.get('title', '')}: {h.get('body', '')}", source=h.get("href", ""), kind="web")
            for i, h in enumerate(hits)
        ]


class TavilySearch:
    def search(self, query: str, n: int) -> list[Doc]:
        import httpx

        r = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": n},
            timeout=30,
        )
        r.raise_for_status()
        return [
            Doc(id=f"web{i}", text=f"{h['title']}: {h['content']}", source=h["url"], kind="web")
            for i, h in enumerate(r.json().get("results", []))
        ]


class NoWebSearch:
    def search(self, query: str, n: int) -> list[Doc]:
        return []


def make_websearch(backend: str) -> WebSearch:
    return {"ddgs": DuckDuckGoSearch, "tavily": TavilySearch, "none": NoWebSearch}[backend]()
