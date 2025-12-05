import argparse
import itertools
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    alpha: float
    temperature: float
    k: int

    def tag(self) -> str:
        t = f"t{self.threshold:.3f}".replace(".", "p")
        a = f"a{self.alpha:.2f}".replace(".", "p")
        temp = f"temp{self.temperature:.2f}".replace(".", "p")
        k = f"k{self.k}"
        return f"{t}-{a}-{temp}-{k}"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run_experiment(config_path: Path) -> None:
    cmd = [sys.executable, str(Path(__file__).with_name("run_experiment.py")), str(config_path)]
    subprocess.run(cmd, check=True)


def move_results(dst_path: Path) -> None:
    src = Path(__file__).with_name("results.jsonl")
    if not src.exists():
        raise FileNotFoundError(f"Expected results at {src}, but not found")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst_path))


def summarize_results(path: Path) -> Dict[str, float]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not records:
        return {"n": 0, "acc": 0.0, "avg_tokens": 0.0, "nudge_rate": 0.0, "acc_when_nudged": 0.0, "mean_top_sim": 0.0}

    n = len(records)
    acc = mean(1.0 if r.get("is_correct") else 0.0 for r in records)
    avg_tokens = mean(float(r.get("num_generated_tokens") or 0.0) for r in records)
    nudged = [r for r in records if r.get("retrieval_nudge_applied")]
    nudge_rate = (len(nudged) / n) if n else 0.0
    acc_when_nudged = mean(1.0 if r.get("is_correct") else 0.0 for r in nudged) if nudged else 0.0
    sims = [float(r.get("retrieval_top_similarity") or 0.0) for r in records]
    mean_top_sim = mean(sims) if sims else 0.0
    return {
        "n": float(n),
        "acc": float(acc),
        "avg_tokens": float(avg_tokens),
        "nudge_rate": float(nudge_rate),
        "acc_when_nudged": float(acc_when_nudged),
        "mean_top_sim": float(mean_top_sim),
    }


def make_sweep_points(
    thresholds: Sequence[float],
    alphas: Sequence[float],
    temperatures: Sequence[float],
    ks: Sequence[int],
) -> List[SweepPoint]:
    pts: List[SweepPoint] = []
    for t, a, temp, k in itertools.product(thresholds, alphas, temperatures, ks):
        pts.append(SweepPoint(threshold=t, alpha=a, temperature=temp, k=k))
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Coconut retrieval params")
    parser.add_argument("--base-config", type=Path, default=Path(__file__).with_name("config.retrieval.40k.1k.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("sweeps"))
    parser.add_argument("--max-examples", type=int, default=None, help="Override max_examples in config for speed")
    parser.add_argument("--quick", action="store_true", help="Run a small, fast sweep (default grid)")
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    base_retrieval = base_cfg.get("retrieval") or {}

    # Default quick sweep: vary threshold and alpha; keep k and temperature fixed
    thresholds = [0.990, 0.992, 0.995, 0.997]
    alphas = [0.5, 1.0]
    temperatures = [base_retrieval.get("temperature", 0.3)]
    ks = [base_retrieval.get("k", 10)]

    if not args.quick:
        # Expand a bit when not quick
        thresholds = [0.988, 0.990, 0.992, 0.995, 0.997]
        alphas = [0.25, 0.5, 1.0]
        temperatures = [0.0, float(base_retrieval.get("temperature", 0.3))]
        ks = [5, int(base_retrieval.get("k", 10))]

    points = make_sweep_points(thresholds, alphas, temperatures, ks)

    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    tmp_cfg_dir = Path(__file__).with_name(".sweep")
    tmp_cfg_dir.mkdir(exist_ok=True)
    results_dir = args.out_dir
    results_dir.mkdir(exist_ok=True)

    summaries: List[Tuple[SweepPoint, Dict[str, float], Path]] = []

    for i, pt in enumerate(points, 1):
        tag = pt.tag()
        cfg = json.loads(json.dumps(base_cfg))  # deep copy via JSON
        cfg.setdefault("run_id", f"sweep-{stamp}-{tag}")
        if args.max_examples is not None:
            cfg["max_examples"] = int(args.max_examples)
        # Apply retrieval overrides
        cfg.setdefault("retrieval", {})
        cfg["retrieval"]["enabled"] = True
        cfg["retrieval"]["threshold"] = float(pt.threshold)
        cfg["retrieval"]["alpha"] = float(pt.alpha)
        cfg["retrieval"]["temperature"] = float(pt.temperature)
        cfg["retrieval"]["k"] = int(pt.k)

        cfg_path = tmp_cfg_dir / f"config.{tag}.yaml"
        dump_yaml(cfg, cfg_path)

        print(f"\n=== [{i}/{len(points)}] Running {tag} ===")
        run_experiment(cfg_path)

        result_path = results_dir / f"results.{tag}.jsonl"
        move_results(result_path)
        stats = summarize_results(result_path)
        print(
            f"   n={int(stats['n'])}, acc={stats['acc']:.4f}, avg_tokens={stats['avg_tokens']:.1f}, "
            f"nudge_rate={stats['nudge_rate']:.3f}, acc_when_nudged={stats['acc_when_nudged']:.4f}, mean_top_sim={stats['mean_top_sim']:.3f}"
        )
        summaries.append((pt, stats, result_path))

    # Rank by accuracy
    print("\n=== Sweep summary (best by acc) ===")
    summaries.sort(key=lambda x: x[1].get("acc", 0.0), reverse=True)
    for rank, (pt, stats, path) in enumerate(summaries, 1):
        print(
            f"{rank:>2}. {pt.tag()} -> acc={stats['acc']:.4f}, nudge_rate={stats['nudge_rate']:.3f}, acc_when_nudged={stats['acc_when_nudged']:.4f} [{path}]"
        )


if __name__ == "__main__":
    main()

