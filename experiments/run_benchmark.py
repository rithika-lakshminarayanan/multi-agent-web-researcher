"""Run benchmark and ablation experiments across agent configurations."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orchestrator import run_agent

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "benchmark_queries.json"
RESULTS_PATH = ROOT / "experiments" / "results.json"

CONFIGS = ["planner_only", "planner_browser", "full"]


def load_queries() -> List[Dict[str, object]]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def run() -> None:
    rows = []
    queries = load_queries()

    for cfg in CONFIGS:
        for item in queries:
            qid = item["id"]
            query = item["query"]
            print(f"Running {cfg} | q{qid}: {query}")
            start = time.time()
            answer, review = run_agent(str(query), mode=cfg)
            elapsed = time.time() - start

            rows.append(
                {
                    "config": cfg,
                    "id": qid,
                    "query": query,
                    "score": int(review.get("score", 0)),
                    "review": review.get("review", ""),
                    "num_sources": int(review.get("num_sources", 0)),
                    "num_reflections": int(review.get("num_reflections", 0)),
                    "runtime_sec": round(elapsed, 2),
                    "answer": answer,
                }
            )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved benchmark results to {RESULTS_PATH}")


if __name__ == "__main__":
    run()
