"""FastAPI service: `uvicorn crag.api:app --reload` (run from src/)."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from crag.config import settings
from crag.pipeline import build_pipelines
from crag.runlog import RunLog

app = FastAPI(title="Corrective RAG")
_crag, _plain = build_pipelines()


class Query(BaseModel):
    question: str
    baseline: bool = False  # true -> run plain RAG instead, for side-by-side comparison


@app.post("/query")
def query(q: Query) -> dict:
    return (_plain if q.baseline else _crag).run(q.question)


@app.get("/runs/{run_id}")
def run(run_id: str) -> dict:
    rec = RunLog(settings.run_log).get(run_id)
    if rec is None:
        raise HTTPException(404, "run not found")
    return rec
