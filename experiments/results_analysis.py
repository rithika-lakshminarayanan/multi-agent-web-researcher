"""Analyze experiment outputs and generate summary metrics + plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "experiments" / "results.json"
SUMMARY_PATH = ROOT / "experiments" / "summary.csv"
PLOT_PATH = ROOT / "experiments" / "score_by_config.png"


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing results file: {RESULTS_PATH}. Run run_benchmark.py first."
        )

    df = pd.read_json(RESULTS_PATH)
    summary = (
        df.groupby("config", as_index=False)
        .agg(
            avg_score=("score", "mean"),
            avg_sources=("num_sources", "mean"),
            avg_reflections=("num_reflections", "mean"),
            avg_runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values("avg_score", ascending=False)
    )

    summary.to_csv(SUMMARY_PATH, index=False)
    print("Summary metrics:")
    print(summary)

    plt.figure(figsize=(8, 5))
    plt.bar(summary["config"], summary["avg_score"], color=["#8da0cb", "#66c2a5", "#fc8d62"])
    plt.title("Average Critic Score by Configuration")
    plt.ylabel("Average Score (1-10)")
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
