import re

import pytest

from crag.config import Settings
from crag.pipeline import build_pipelines
from crag.store import Doc


class FakeLLM:
    """Deterministic stand-in. Grades by keyword overlap so tests exercise real branching logic."""

    def __init__(self, grades: dict[str, float] | None = None):
        self.grades = grades or {}
        self.calls: list[str] = []

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        self.calls.append(system)
        if "grade whether retrieved passages" in system:
            query = prompt.split("\n", 1)[0]
            n = len(re.findall(r"^\[\d+\]", prompt, re.M))
            score = next((s for k, s in self.grades.items() if k in query), 0)
            return '{"scores": [' + ",".join(f'{{"id": {i}, "score": {score}}}' for i in range(n)) + "]}"
        if "select the sentences" in system:
            n = len(re.findall(r"^\[\d+\]", prompt, re.M))
            return '{"keep": [' + ",".join(map(str, range(n))) + "]}"
        if "web-search queries" in system:
            return "search terms"
        return "Answer from sources [D1]." if "[D1]" in prompt else "Answer from web [W1]."


class FakeWeb:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, n):
        self.queries.append(query)
        return [Doc(id="web0", text="Web page: a relevant web fact.", source="https://example.com", kind="web")]


@pytest.fixture
def make(tmp_path):
    def _make(grades):
        cfg = Settings(embedder="tfidf", run_log=tmp_path / "runs.jsonl")
        llm, web = FakeLLM(grades), FakeWeb()
        crag, plain = build_pipelines(cfg, grader=llm, generator=llm, web=web)
        return crag, plain, llm, web

    return _make
