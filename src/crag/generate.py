"""Grounded generation with inline citations: [D#] = internal doc, [W#] = web result."""

from __future__ import annotations

from crag.llm import LLM
from crag.store import Doc

GEN_SYSTEM = (
    "You are a customer-support assistant. Answer ONLY from the provided sources. "
    "Every sentence must end with a citation like [D1] or [W2] naming the source that supports it. "
    "If the sources do not contain the answer, reply exactly: I don't have enough information to answer that."
)


def label(docs: list[Doc]) -> list[tuple[str, Doc]]:
    counters = {"internal": 0, "web": 0}
    out = []
    for d in docs:
        counters[d.kind] += 1
        out.append((f"{'D' if d.kind == 'internal' else 'W'}{counters[d.kind]}", d))
    return out


def format_sources(labeled: list[tuple[str, Doc]]) -> str:
    return "\n\n".join(f"[{tag}] (source: {d.source})\n{d.text}" for tag, d in labeled)


# The "typical tutorial RAG" prompt: no abstain instruction. Used only as a second baseline in the eval.
NAIVE_SYSTEM = "You are a customer-support assistant. Use the sources to answer the question, citing them like [D1]."


def generate(llm: LLM, query: str, docs: list[Doc], strict: bool = True) -> tuple[str, list[tuple[str, Doc]]]:
    labeled = label(docs)
    if not labeled and strict:
        return "I don't have enough information to answer that.", []
    prompt = f"Sources:\n{format_sources(labeled) or '(none)'}\n\nQuestion: {query}\nAnswer:"
    return llm.complete(GEN_SYSTEM if strict else NAIVE_SYSTEM, prompt, 1024).strip(), labeled
