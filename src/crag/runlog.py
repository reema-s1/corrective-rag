"""Append-only JSONL run log: every query's scores, action, sources and timings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get(self, run_id: str) -> dict | None:
        if not self.path.exists():
            return None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("run_id") == run_id:
                return rec
        return None
