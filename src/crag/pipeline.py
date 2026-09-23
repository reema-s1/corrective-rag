"""CRAG pipeline (retrieve -> evaluate -> correct -> generate) and the plain-RAG baseline."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict

from crag.config import Settings, settings as default_settings
from crag.evaluator import Action, LLMJudgeEvaluator, RetrievalEvaluator, decide_action
from crag.generate import generate
from crag.llm import LLM, make_llm
from crag.refine import refine, rewrite_for_web
from crag.runlog import RunLog
from crag.store import Doc, Embedder, VectorStore, load_kb
from crag.websearch import WebSearch, make_websearch


def _doc_summary(labeled: list[tuple[str, Doc]]) -> list[dict]:
    return [{"tag": t, "id": d.id, "kind": d.kind, "source": d.source, "text": d.text} for t, d in labeled]


class _Timer:
    def __init__(self):
        self.t0, self.marks = time.perf_counter(), {}

    def mark(self, name: str):
        now = time.perf_counter()
        self.marks[name] = round((now - self.t0) * 1000)
        self.t0 = now


class PlainRAGPipeline:
    """Baseline: same retriever, same generator prompt, no evaluation or correction."""

    def __init__(self, store: VectorStore, generator: LLM, log: RunLog, top_k: int, strict: bool = True):
        self.store, self.generator, self.log, self.top_k, self.strict = store, generator, log, top_k, strict
        self.name = "plain_rag" if strict else "plain_rag_naive"

    def run(self, query: str) -> dict:
        timer = _Timer()
        docs = self.store.search(query, self.top_k)
        timer.mark("retrieve_ms")
        answer, labeled = generate(self.generator, query, docs, strict=self.strict)
        timer.mark("generate_ms")
        record = {
            "run_id": uuid.uuid4().hex[:12], "pipeline": self.name, "query": query, "action": None,
            "retrieved": [{"id": d.id, "sim": round(d.score, 3)} for d in docs],
            "sources": _doc_summary(labeled), "answer": answer,
            "timings": {**timer.marks, "total_ms": sum(timer.marks.values())},
        }
        self.log.write(record)
        return record


class CRAGPipeline:
    name = "crag"

    def __init__(self, store: VectorStore, evaluator: RetrievalEvaluator, grader: LLM, generator: LLM,
                 web: WebSearch, log: RunLog, top_k: int, web_results: int):
        self.store, self.evaluator, self.grader, self.generator = store, evaluator, grader, generator
        self.web, self.log, self.top_k, self.web_results = web, log, top_k, web_results

    def _web_knowledge(self, query: str) -> tuple[str, list[Doc]]:
        web_query = rewrite_for_web(self.grader, query)
        hits = self.web.search(web_query, self.web_results)
        return web_query, refine(self.grader, query, hits) if hits else []

    def run(self, query: str) -> dict:
        timer = _Timer()
        docs = self.store.search(query, self.top_k)
        timer.mark("retrieve_ms")

        scores = self.evaluator.score(query, docs)
        action = decide_action(scores, self.evaluator.upper, self.evaluator.lower)
        timer.mark("evaluate_ms")

        web_query = None
        if action is Action.CORRECT:
            # Only refine docs that cleared the lower bar; low scorers are noise even on the Correct path.
            context = refine(self.grader, query, [d for d, s in zip(docs, scores) if s > self.evaluator.lower])
        elif action is Action.INCORRECT:
            # Internal retrieval is judged useless: discard it entirely rather than risk grounding on it.
            web_query, context = self._web_knowledge(query)
        else:  # AMBIGUOUS: hedge — refined internal strips + refined web evidence
            internal = refine(self.grader, query, [d for d, s in zip(docs, scores) if s > self.evaluator.lower])
            web_query, web_docs = self._web_knowledge(query)
            context = internal + web_docs
        timer.mark("correct_ms")

        answer, labeled = generate(self.generator, query, context)
        timer.mark("generate_ms")

        record = {
            "run_id": uuid.uuid4().hex[:12], "pipeline": self.name, "query": query, "action": action.value,
            "retrieved": [
                {"id": d.id, "sim": round(d.score, 3), "grade": s, "reason": d.meta.get("grade_reason", "")}
                for d, s in zip(docs, scores)
            ],
            "web_query": web_query, "sources": _doc_summary(labeled), "answer": answer,
            "timings": {**timer.marks, "total_ms": sum(timer.marks.values())},
        }
        self.log.write(record)
        return record


def build_pipelines(cfg: Settings = default_settings, grader: LLM | None = None, generator: LLM | None = None,
                    web: WebSearch | None = None) -> tuple[CRAGPipeline, PlainRAGPipeline]:
    """Wire both pipelines on a shared index. LLMs / web are injectable so tests can pass fakes."""
    docs = load_kb(cfg.kb_dir)
    store = VectorStore(docs, Embedder(cfg.embedder, [d.text for d in docs]))
    grader = grader or make_llm(cfg.provider, cfg.grader_model)
    generator = generator or make_llm(cfg.provider, cfg.generator_model)
    web = web or make_websearch(cfg.web_backend)
    log = RunLog(cfg.run_log)
    evaluator = LLMJudgeEvaluator(grader, cfg.upper_threshold, cfg.lower_threshold)
    crag = CRAGPipeline(store, evaluator, grader, generator, web, log, cfg.top_k, cfg.web_results)
    return crag, PlainRAGPipeline(store, generator, log, cfg.top_k)
