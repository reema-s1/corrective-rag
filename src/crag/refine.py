"""Knowledge refinement ("decompose-then-recompose" in the paper).

Split each doc into sentence-level strips, keep only the strips that bear on the question, and
concatenate them. The paper scores every strip with the T5 evaluator; we ask the LLM to select strip
ids in a single call.
"""

from __future__ import annotations

import re

from crag.llm import LLM, parse_json
from crag.store import Doc

REFINE_SYSTEM = "You select the sentences needed to answer a question. Reply with JSON only."
REFINE_PROMPT = """Question: {query}

Numbered sentences:
{strips}

Return the ids of every sentence that helps answer the question (include needed context, drop the rest):
{{"keep": [<ids>]}}"""


def decompose(doc: Doc) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.text) if len(s.strip()) > 3]


def refine(llm: LLM, query: str, docs: list[Doc]) -> list[Doc]:
    strips = [(d, s) for d in docs for s in decompose(d)]
    if not strips:
        return []
    listing = "\n".join(f"[{i}] {s}" for i, (_, s) in enumerate(strips))
    keep = set(parse_json(llm.complete(REFINE_SYSTEM, REFINE_PROMPT.format(query=query, strips=listing), 256))["keep"])
    refined: dict[str, Doc] = {}
    for i, (doc, strip) in enumerate(strips):
        if i in keep:  # recompose: kept strips are regrouped under their source doc so citations still work
            refined.setdefault(doc.id, Doc(id=doc.id, text="", source=doc.source, kind=doc.kind, score=doc.score))
            refined[doc.id].text += strip + " "
    return [Doc(**{**d.__dict__, "text": d.text.strip()}) for d in refined.values()]


REWRITE_SYSTEM = "You turn questions into short web-search queries. Reply with the query only."


def rewrite_for_web(llm: LLM, query: str) -> str:
    # Paper rewrites the question into keywords before searching; conversational phrasing hurts search.
    return llm.complete(REWRITE_SYSTEM, f"Question: {query}\nSearch query (max 10 words):", 64).strip().strip('"')
