"""LLM grounding judge. Scores an answer against the exact sources the pipeline gave the generator."""

from __future__ import annotations

import re

from crag.llm import LLM, parse_json

JUDGE_SYSTEM = (
    "You are a strict evaluator of retrieval-augmented answers. You check claims against sources only, "
    "never against your own knowledge. Reply with JSON only."
)

JUDGE_PROMPT = """Question: {query}
Reference answer (for correctness only): {reference}

Sources given to the system:
{sources}

System answer:
{answer}

Tasks:
1. Split the system answer into atomic factual claims (ignore citation tags and pleasantries).
   For each, decide if it is SUPPORTED by the sources above (not by the reference, not by your knowledge).
2. abstained: true if the answer declines to answer / says it lacks information.
3. correct: true if the answer's substance agrees with the reference answer (false if it abstained).

Return JSON: {{"claims": [{{"claim": "...", "supported": true}}], "abstained": false, "correct": true}}"""


def judge(llm: LLM, query: str, reference: str, sources: list[dict], answer: str) -> dict:
    src = "\n\n".join(f"[{s['tag']}] {s['text']}" for s in sources) or "(no sources)"
    out = parse_json(llm.complete(JUDGE_SYSTEM, JUDGE_PROMPT.format(
        query=query, reference=reference, sources=src, answer=answer), 1500))
    return finalize(out)


ABSTAIN = re.compile(r"(don't|do not) have enough information", re.I)


def finalize(out: dict) -> dict:
    # The judge sometimes lists the refusal sentence itself as an "unsupported claim"; it isn't a factual claim.
    claims = [c for c in out.get("claims", []) if not ABSTAIN.search(c.get("claim", ""))]
    out["unsupported"] = [c["claim"] for c in claims if not c.get("supported")]
    # An abstention makes no claims, so it is trivially grounded — reported separately as abstain rate.
    out["grounded"] = not out["unsupported"]
    return out
