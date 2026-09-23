"""Retrieval evaluator: scores each retrieved doc, then maps the scores to a corrective action.

Paper: a fine-tuned T5-large scores each (query, doc) pair in [-1, 1]; two thresholds pick the action.
Here: implementation A (default) is an LLM-as-judge that scores all docs in ONE call on 0-10;
implementation B is a zero-cost embedding-similarity threshold, kept as a comparison baseline.
The action rule (`decide_action`) is the paper's and is shared by both.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from crag.llm import LLM, parse_json
from crag.store import Doc


class Action(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


def decide_action(scores: list[float], upper: float, lower: float) -> Action:
    # Paper rule: any doc confidently relevant -> Correct; every doc confidently irrelevant -> Incorrect;
    # everything in between -> Ambiguous (hedge by using both sources).
    if any(s >= upper for s in scores):
        return Action.CORRECT
    if all(s <= lower for s in scores):
        return Action.INCORRECT
    return Action.AMBIGUOUS


class RetrievalEvaluator(Protocol):
    upper: float
    lower: float

    def score(self, query: str, docs: list[Doc]) -> list[float]: ...


GRADER_SYSTEM = (
    "You grade whether retrieved passages contain information that answers a user's question. "
    "You are strict: topical overlap is not enough — the passage must contain the specific fact asked for. "
    "Reply with JSON only."
)

GRADER_PROMPT = """Question: {query}

Passages:
{passages}

For each passage give a relevance score from 0 to 10:
- 10 = passage directly contains the answer
- 5 = related and partially helpful, but the answer is incomplete or needs outside knowledge
- 0 = irrelevant, or about a different product/topic than the question
Return JSON: {{"scores": [{{"id": <passage number>, "score": <0-10>, "reason": "<8 words max>"}}, ...]}}"""


class LLMJudgeEvaluator:
    def __init__(self, llm: LLM, upper: float = 7.0, lower: float = 3.0):
        self.llm, self.upper, self.lower = llm, upper, lower

    def score(self, query: str, docs: list[Doc]) -> list[float]:
        if not docs:
            return []
        passages = "\n\n".join(f"[{i}] {d.text}" for i, d in enumerate(docs))
        # One batched call for all k docs instead of k calls: ~k x fewer round trips on the hot path.
        out = parse_json(self.llm.complete(GRADER_SYSTEM, GRADER_PROMPT.format(query=query, passages=passages), 512))
        by_id = {int(s["id"]): float(s["score"]) for s in out["scores"]}
        for i, d in enumerate(docs):
            d.meta["grade_reason"] = next((s.get("reason", "") for s in out["scores"] if int(s["id"]) == i), "")
        return [by_id.get(i, 0.0) for i in range(len(docs))]  # a doc the judge skipped counts as irrelevant


class EmbeddingThresholdEvaluator:
    """Uses the retriever's own cosine similarity. Free and instant, but blind to 'similar topic, wrong fact'."""

    def __init__(self, upper: float = 0.6, lower: float = 0.35):
        self.upper, self.lower = upper, lower

    def score(self, query: str, docs: list[Doc]) -> list[float]:
        return [d.score for d in docs]
