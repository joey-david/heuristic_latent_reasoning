#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import yaml


def load_results(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_accuracies(records: List[Dict], models: List[str]) -> Dict[str, float]:
    totals = {name: 0 for name in models}
    correct = {name: 0 for name in models}
    for record in records:
        model_stats = record.get("models", {})
        for name in models:
            if name not in model_stats:
                continue
            totals[name] += 1
            correct[name] += 1 if model_stats[name].get("correct") else 0
    return {name: (correct[name] / totals[name]) if totals[name] else 0.0 for name in models}


def plot_histogram(accuracies: Dict[str, float], output: Path) -> None:
    labels = list(accuracies.keys())
    values = [accuracies[label] * 100 for label in labels]
    colors = ["#4E79A7", "#F28E2B", "#59A14F"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="#1b1b1d", linewidth=0.8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model Accuracies", pad=14, fontsize=13, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model accuracies from a YAML results file.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("exps/results.yaml"),
        help="Path to the evaluation results YAML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/model_accuracy_histogram.png"),
        help="Destination for the generated plot.",
    )
    args = parser.parse_args()

    models = ["cot", "coconut", "faiss_augmented"]
    records = load_results(args.results)
    accuracies = compute_accuracies(records, models)
    plot_histogram(accuracies, args.output)


if __name__ == "__main__":
    main()

