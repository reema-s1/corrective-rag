"""Eval harness: runs CRAG and plain-RAG baselines over queries.jsonl and writes report.md.

    python eval/run_eval.py                       # full run: crag + both baselines, 20 queries
    python eval/run_eval.py --pipelines crag --limit 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from crag.config import settings  # noqa: E402
from crag.llm import make_llm  # noqa: E402
from crag.pipeline import PlainRAGPipeline, build_pipelines  # noqa: E402
from judge import finalize, judge  # noqa: E402

WEB_ACTIONS = {"incorrect", "ambiguous"}


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}% ({n}/{d})" if d else "n/a"


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    lat = [r["timings"]["total_ms"] for r in rows]
    return {
        "n": n,
        "grounded": sum(r["judge"]["grounded"] for r in rows),
        "correct": sum(r["judge"]["correct"] for r in rows),
        "correct_and_grounded": sum(r["judge"]["correct"] and r["judge"]["grounded"] for r in rows),
        "abstained": sum(r["judge"]["abstained"] for r in rows),
        "hallucinated": sum(not r["judge"]["grounded"] for r in rows),
        "lat_mean": statistics.mean(lat), "lat_p50": statistics.median(lat),
    }


def routing(rows: list[dict]) -> dict:
    exact = sum(r["action"] == r["expected"] for r in rows)
    should = [r for r in rows if r["expected"] in WEB_ACTIONS]
    did = [r for r in rows if r["action"] in WEB_ACTIONS]
    tp = sum(r["action"] in WEB_ACTIONS for r in should)
    confusion = Counter((r["expected"], r["action"]) for r in rows)
    return {"exact": exact, "n": len(rows), "fallback_tp": tp, "should": len(should), "did": len(did),
            "confusion": confusion}


def render(results: dict[str, list[dict]], out: Path) -> str:
    L = [f"# CRAG eval report ({date.today()})", "",
         f"- Queries: {len(next(iter(results.values())))} "
         f"({dict(Counter(r['expected'] for r in next(iter(results.values()))))})",
         f"- Generator: `{settings.generator_model}` · grader: `{settings.grader_model}` · "
         f"judge: `{settings.generator_model}` · embedder: `{settings.embedder}` · web: `{settings.web_backend}`",
         f"- Thresholds: upper={settings.upper_threshold}, lower={settings.lower_threshold}, top_k={settings.top_k}", "",
         "## Answer quality", "",
         "| Pipeline | Grounded (no unsupported claims) | Correct | Correct **and** grounded | Abstained | Mean latency | p50 latency |",
         "|---|---|---|---|---|---|---|"]
    sums = {}
    for name, rows in results.items():
        s = sums[name] = summarize(rows)
        L.append(f"| `{name}` | {pct(s['grounded'], s['n'])} | {pct(s['correct'], s['n'])} | "
                 f"{pct(s['correct_and_grounded'], s['n'])} | {pct(s['abstained'], s['n'])} | "
                 f"{s['lat_mean']:.0f} ms | {s['lat_p50']:.0f} ms |")
    if "crag" in results:
        r = routing(results["crag"])
        L += ["", "## CRAG routing", "",
              f"- Branch accuracy (exact Correct/Incorrect/Ambiguous match): **{pct(r['exact'], r['n'])}**",
              f"- Web-fallback recall (should use web -> did): **{pct(r['fallback_tp'], r['should'])}**",
              f"- Web-fallback precision (used web -> should have): **{pct(r['fallback_tp'], r['did'])}**", "",
              "| expected (rows) vs actual (cols) | correct | ambiguous | incorrect |", "|---|---|---|---|"]
        for e in ("correct", "ambiguous", "incorrect"):
            L.append(f"| {e} | " + " | ".join(str(r["confusion"][(e, a)]) for a in ("correct", "ambiguous", "incorrect")) + " |")
        for base in ("plain_rag", "plain_rag_naive"):
            if base in sums:
                d = sums["crag"]["lat_mean"] - sums[base]["lat_mean"]
                L.append(f"\nLatency cost of correction vs `{base}`: **{d:+.0f} ms** mean per query.")
    L += ["", "## Per-query detail", "", "| id | expected | " + " | ".join(results) + " |",
          "|---|---|" + "---|" * len(results)]
    rows_by_id = {name: {r["id"]: r for r in rows} for name, rows in results.items()}
    for qid in rows_by_id[next(iter(results))]:
        cells = []
        for name in results:
            r = rows_by_id[name][qid]
            j = r["judge"]
            flag = "abstain" if j["abstained"] else ("ok" if j["correct"] else "wrong")
            flag += "" if j["grounded"] else " / **UNGROUNDED**"
            cells.append((f"{r['action']}: " if r.get("action") else "") + flag)
        L.append(f"| {qid} | {rows_by_id[next(iter(results))][qid]['expected']} | " + " | ".join(cells) + " |")
    unsupported = [(n, r["id"], c) for n, rows in results.items() for r in rows for c in r["judge"]["unsupported"]]
    if unsupported:
        L += ["", "## Unsupported claims found by the judge", ""]
        L += [f"- `{n}` {qid}: {c}" for n, qid, c in unsupported]
    L += ["", "Raw per-query records (scores, sources, answers, judge output): `eval/results.jsonl`."]
    text = "\n".join(L) + "\n"
    out.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", default="crag,plain_rag,plain_rag_naive")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rerender", action="store_true", help="re-score saved results.jsonl without API calls")
    args = ap.parse_args()

    if args.rerender:
        results: dict[str, list[dict]] = {}
        for line in (HERE / "results.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            rec["judge"] = finalize(rec["judge"])
            results.setdefault(rec["pipeline"], []).append(rec)
        print(render(results, HERE / "report.md"))
        return

    queries = [json.loads(l) for l in (HERE / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    queries = queries[: args.limit] if args.limit else queries
    crag, plain = build_pipelines()
    naive = PlainRAGPipeline(plain.store, plain.generator, plain.log, plain.top_k, strict=False)
    pipes = {p.name: p for p in (crag, plain, naive)}
    judge_llm = make_llm(settings.provider, settings.generator_model)

    results: dict[str, list[dict]] = {}
    with (HERE / "results.jsonl").open("w", encoding="utf-8") as f:
        for name in args.pipelines.split(","):
            results[name] = []
            for q in queries:
                rec = pipes[name].run(q["query"])
                rec.update(id=q["id"], expected=q["expected"], reference=q["reference"])
                rec["judge"] = judge(judge_llm, q["query"], q["reference"], rec["sources"], rec["answer"])
                results[name].append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{name}] {q['id']:>3} action={rec['action']!s:<9} grounded={rec['judge']['grounded']!s:<5} "
                      f"correct={rec['judge']['correct']!s:<5} {rec['timings']['total_ms']} ms")
    print("\n" + render(results, HERE / "report.md"))


if __name__ == "__main__":
    main()
