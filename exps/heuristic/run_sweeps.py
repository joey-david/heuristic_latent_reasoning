#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "plots" / "heuristic"


BASE_CONFIG: Dict[str, object] = {
    "dataset": "data/gsm_train.json",
    "model_checkpoint": "data/checkpoints/gsm/gsm-coconut",
    "model_id": "openai-community/gpt2",
    "max_examples": 5000,
    "max_new_tokens": 128,
    "num_latent_thoughts": 8,
    "dry_run": False,
    "train_mode": True,
    "live_plot": True,
    "live_plot_interactive": False,
    "live_plot_save_every": 100,
    "heuristic_memory": "args/gsm_heuristic.yaml",
    "memory_max_entries": 128,
    "retrieval_threshold": 0.95,
    "nudge_min_prob": 0.5,
    "nudge_lr": 0.0003,
    "nudge_inactivity_penalty": 0.2,
}


SWEEPS: Dict[str, List[float | int]] = {
    "memory_max_entries": [32, 128, 512],
    "retrieval_threshold": [0.8, 0.88, 0.97],
    "nudge_min_prob": [0.0, 0.25, 0.6],
    "nudge_lr": [0.0001, 0.0008, 0.003],
}


def _format_value(value: float | int) -> str:
    text = f"{value}"
    return text.replace(".", "p").replace("-", "m")


def _build_runs() -> Iterable[Tuple[str, Dict[str, float | int]]]:
    yield ("heuristic_base", {})
    for param, values in SWEEPS.items():
        for value in values:
            yield (f"sweep_{param}_{_format_value(value)}", {param: value})


def _prepare_config(run_id: str, overrides: Dict[str, float | int]) -> Path:
    config = deepcopy(BASE_CONFIG)
    config.update(overrides)
    config["run_id"] = run_id
    run_dir = PLOTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    relative_run_dir = Path("plots") / "heuristic" / run_id
    config["live_plot_path"] = str(relative_run_dir / "plot.png")
    config["faiss_index_path"] = str(relative_run_dir / "index.faiss")
    config["metadata_path"] = str(relative_run_dir / "meta.pkl")
    config["nudge_weights_path"] = str(relative_run_dir / "nudge.pt")

    config_path = run_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


def _cleanup_run_dir(run_dir: Path) -> None:
    for name in ("index.faiss", "meta.pkl", "nudge.pt"):
        path = run_dir / name
        if path.exists():
            path.unlink()


def _run_experiment(config_path: Path) -> None:
    rel_path = config_path.relative_to(ROOT)
    subprocess.run(
        ["python3", "exps/heuristic/run_experiment.py", str(rel_path)],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for run_id, overrides in _build_runs():
        print(f"[heuristic-sweep] starting {run_id}")
        config_path = _prepare_config(run_id, overrides)
        try:
            _run_experiment(config_path)
        finally:
            _cleanup_run_dir(config_path.parent)
            print(f"[heuristic-sweep] finished {run_id}")


if __name__ == "__main__":
    main()
