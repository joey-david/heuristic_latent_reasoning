import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_results(model_paths: Dict[str, Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for model, path in model_paths.items():
        if not path.exists():
            print(f"[plots] skipping missing log for {model}: {path}")
            continue

        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("model_name") != model:
                    payload["model_name"] = model
                records.append(payload)

        frames.append(pd.DataFrame.from_records(records))

    if not frames:
        raise RuntimeError("No result logs were loaded; ensure paths are correct.")
    return pd.concat(frames, ignore_index=True)


def ensure_plots_dir() -> Path:
    plots_dir = Path(__file__).resolve().parent / "_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def accuracy_tokens_table(df: pd.DataFrame, plots_dir: Path) -> None:
    summary = (
        df.groupby("model_name")
        .agg(
            accuracy=("is_correct", "mean"),
            mean_tokens=("num_generated_tokens", "mean"),
        )
        .reset_index()
    )
    summary["accuracy_percent"] = summary["accuracy"] * 100.0
    summary = summary[["model_name", "accuracy_percent", "mean_tokens"]]

    table_path = plots_dir / "accuracy_tokens_table.csv"
    summary.to_csv(table_path, index=False)
    print(f"Saved accuracy vs tokens table to {table_path}")


def accuracy_vs_thoughts(df: pd.DataFrame, plots_dir: Path) -> None:
    focus = df[df["model_name"].isin(["coconut", "knot"])].copy()
    if focus.empty:
        print("No Coconut or kNoT results found; skipping accuracy vs thoughts plot.")
        return

    plt.figure(figsize=(6, 4))
    for model, group in focus.groupby("model_name"):
        stats = (
            group.groupby("num_latent_thoughts")
            .agg(
                accuracy=("is_correct", "mean"),
                count=("is_correct", "count"),
            )
            .reset_index()
            .sort_values("num_latent_thoughts")
        )
        if stats.empty:
            continue

        ci = 1.96 * (stats["accuracy"] * (1 - stats["accuracy"]) / stats["count"]).pow(
            0.5
        )

        plt.plot(
            stats["num_latent_thoughts"],
            stats["accuracy"] * 100.0,
            marker="o",
            label=model,
        )
        plt.fill_between(
            stats["num_latent_thoughts"],
            (stats["accuracy"] - ci) * 100.0,
            (stats["accuracy"] + ci) * 100.0,
            alpha=0.2,
        )

    plt.xlabel("# Thoughts per step")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    out_path = plots_dir / "accuracy_vs_thoughts.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved accuracy vs thoughts plot to {out_path}")


def accuracy_vs_inference_time(df: pd.DataFrame, plots_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    colors = {"cot": "#1f77b4", "coconut": "#ff7f0e", "knot": "#2ca02c"}
    for model, group in df.groupby("model_name"):
        plt.scatter(
            group["inference_time_ms"],
            group["is_correct"],
            label=model,
            alpha=0.6,
            s=18,
            color=colors.get(model),
        )
    plt.xlabel("Inference time (ms)")
    plt.ylabel("Accuracy (0/1)")
    plt.legend()
    plt.tight_layout()
    out_path = plots_dir / "accuracy_vs_inference_time.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved accuracy vs inference time scatter to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate experiment plots by aggregating model result logs."
    )
    parser.add_argument(
        "--cot-path",
        type=Path,
        default=Path(__file__).resolve().parent / "cot" / "results.jsonl",
    )
    parser.add_argument(
        "--coconut-path",
        type=Path,
        default=Path(__file__).resolve().parent / "coconut" / "results.jsonl",
    )
    parser.add_argument(
        "--knot-path",
        type=Path,
        default=Path(__file__).resolve().parent / "knot" / "results.jsonl",
    )
    args = parser.parse_args()

    df = load_results(
        {
            "cot": args.cot_path,
            "coconut": args.coconut_path,
            "knot": args.knot_path,
        }
    )

    plots_dir = ensure_plots_dir()
    accuracy_tokens_table(df, plots_dir)
    accuracy_vs_thoughts(df, plots_dir)
    accuracy_vs_inference_time(df, plots_dir)


if __name__ == "__main__":
    main()
