from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import yaml

from . import utils


@dataclass
class RunEntry:
    """Describes a single run participating in a report."""
    label: str
    metrics_path: Path
    flips_path: Optional[Path] = None
    name: Optional[str] = None


@dataclass
class DatasetReport:
    """Captures reporting targets for one dataset."""
    dataset: str
    runs: List[RunEntry]
    table_path: Path
    plot_path: Path


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def load_report_plan(path: Path) -> List[DatasetReport]:
    """Parses the global reporting configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Report config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    plans: List[DatasetReport] = []
    for item in payload.get("reports", []):
        dataset = str(item["dataset"])
        table_path = _as_path(item["table_path"])
        plot_path = _as_path(item["plot_path"])
        run_entries: List[RunEntry] = []
        for run in item.get("runs", []):
            run_entries.append(
                RunEntry(
                    label=str(run["label"]),
                    metrics_path=_as_path(run["metrics_path"]),
                    flips_path=(
                        None if run.get("flips_path") is None else _as_path(run["flips_path"])
                    ),
                    name=run.get("name"),
                )
            )
        plans.append(DatasetReport(dataset=dataset, runs=run_entries, table_path=table_path, plot_path=plot_path))
    return plans


def _resolve_accuracy(metrics: Dict[str, Any]) -> float:
    for key in ("knot_accuracy", "accuracy", "base_accuracy"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError("Metrics file missing accuracy fields.")


def _resolve_tokens(metrics: Dict[str, Any]) -> Optional[float]:
    value = metrics.get("avg_tokens") or metrics.get("tokens")
    return None if value is None else float(value)


def _summarize_flips(flips_path: Optional[Path], metrics: Dict[str, Any]) -> Dict[str, int]:
    if flips_path and flips_path.exists():
        wrong_to_right = 0
        right_to_wrong = 0
        for record in utils.read_jsonl(flips_path):
            base_correct = bool(record.get("base_correct"))
            final_correct = bool(record.get("final_correct"))
            if base_correct and not final_correct:
                right_to_wrong += 1
            elif (not base_correct) and final_correct:
                wrong_to_right += 1
        return {
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
        }
    return {
        "wrong_to_right": int(metrics.get("improved", 0)),
        "right_to_wrong": int(metrics.get("regressed", 0)),
    }


def _write_table(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    utils.ensure_directory(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_accuracy(dataset: str, rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    labels = [row["run"] for row in rows]
    accuracies = [row["accuracy_pct"] for row in rows]
    tokens = [row.get("avg_tokens") for row in rows]

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accuracies, color="#4f81bd")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{dataset} accuracy comparison")
    ax.set_ylim(0, max(accuracies + [0]) * 1.1 if accuracies else 1)

    for bar, token in zip(bars, tokens):
        if token is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{token:.1f} toks",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    utils.ensure_directory(out_path)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_report(plan: DatasetReport) -> None:
    """Produces a CSV table and accuracy plot for a dataset."""
    rows: List[Dict[str, Any]] = []
    for run in plan.runs:
        metrics = utils.load_json(run.metrics_path)
        accuracy = _resolve_accuracy(metrics)
        tokens = _resolve_tokens(metrics)
        flips = _summarize_flips(run.flips_path, metrics)
        name = run.label
        row = {
            "run": name,
            "accuracy": accuracy,
            "accuracy_pct": accuracy * 100.0,
            "avg_tokens": tokens,
            "wrong_to_right": flips["wrong_to_right"],
            "right_to_wrong": flips["right_to_wrong"],
            "total": int(metrics.get("total", 0)),
        }
        net_gain = flips["wrong_to_right"] - flips["right_to_wrong"]
        row["net_flip"] = net_gain
        rows.append(row)

    fieldnames = ["run", "accuracy", "accuracy_pct", "avg_tokens", "wrong_to_right", "right_to_wrong", "net_flip", "total"]
    _write_table(plan.table_path, rows, fieldnames)
    _plot_accuracy(plan.dataset, rows, plan.plot_path)
    print(f"[kNoT-report] Wrote table to {plan.table_path}")
    print(f"[kNoT-report] Saved plot to {plan.plot_path}")


def generate_reports(config_path: Path) -> None:
    """Generates reports for all datasets listed in the config."""
    plans = load_report_plan(config_path)
    if not plans:
        print("[kNoT-report] No report entries found.")
        return
    for plan in plans:
        generate_report(plan)
