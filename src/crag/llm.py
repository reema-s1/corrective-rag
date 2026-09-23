"""Provider-agnostic LLM interface. The pipeline only depends on `LLM.complete`."""

from __future__ import annotations

import json
import re
from typing import Protocol

try:  # use the OS certificate store (works behind antivirus / corporate TLS inspection)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


class LLM(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str):
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"{self.model} refused the request")
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAILLM:
    def __init__(self, model: str):
        import openai

        # OPENAI_BASE_URL (read by the SDK) points this at any OpenAI-compatible host: Groq, Gemini, Ollama.
        # Extra retries ride out free-tier 429s.
        self._client = openai.OpenAI(max_retries=8)
        self.model = model

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            # Reasoning models (gpt-oss, o-series) spend completion tokens on thinking; leave headroom.
            max_completion_tokens=max_tokens + 4096,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


def make_llm(provider: str, model: str) -> LLM:
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAILLM(model)
    raise ValueError(f"unknown provider {provider!r}")


def parse_json(text: str):
    """Extract the first JSON object/array from a model reply (tolerates ```json fences and prose)."""
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in model output: {text[:200]!r}")
    return json.loads(match.group(1))
