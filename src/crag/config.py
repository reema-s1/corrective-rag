"""Runtime settings, read from environment variables (prefix CRAG_)."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

# Export .env (e.g. ANTHROPIC_API_KEY) to the process so the provider SDKs pick it up; real env vars win.
if (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and value.strip() and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.split("#")[0].strip())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRAG_", env_file=ROOT / ".env", extra="ignore")

    provider: str = "anthropic"  # anthropic | openai
    generator_model: str = "claude-opus-5"
    grader_model: str = "claude-haiku-4-5"  # cheap model for the evaluator / refiner / judge

    kb_dir: Path = ROOT / "data" / "kb"
    run_log: Path = ROOT / "runs" / "runs.jsonl"
    embedder: str = "tfidf"  # or a sentence-transformers model name, e.g. "all-MiniLM-L6-v2"

    top_k: int = 4
    # Paper-style thresholds on the evaluator's relevance score (here 0-10, paper uses [-1, 1]).
    upper_threshold: float = 7.0
    lower_threshold: float = 3.0
    web_results: int = 4
    web_backend: str = "ddgs"  # ddgs | tavily | none


settings = Settings()
