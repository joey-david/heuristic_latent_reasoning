import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import yaml

from exps.logger import ExperimentLogger
from heuristic import HeuristicMemory


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(candidate: Any) -> Path:
    if candidate in (None, "None"):
        raise ValueError("A dataset path must be provided in the config file.")
    return Path(candidate)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data["data"] if isinstance(data, dict) and "data" in data else data
    raise ValueError(f"Unsupported dataset format: {path}")


def ensure_engine(
    module_name: str,
    model_checkpoint: Any,
    num_latent_thoughts: int,
    heuristic_memory: Optional[HeuristicMemory] = None,
) -> Tuple[Any, Any]:
    module = importlib.import_module(module_name)

    if not hasattr(module, "load_engine") or not hasattr(module, "infer"):
        raise AttributeError(
            f"Module {module_name!r} must define load_engine() and infer()."
        )

    load_kwargs = {
        "model_checkpoint": model_checkpoint,
        "num_latent_thoughts": num_latent_thoughts,
    }
    if heuristic_memory is not None:
        load_kwargs["heuristic_memory"] = heuristic_memory

    try:
        engine = module.load_engine(**load_kwargs)
    except TypeError:
        # Backwards compatibility for engines that do not yet accept heuristic memory.
        load_kwargs.pop("heuristic_memory", None)
        engine = module.load_engine(**load_kwargs)
    return module, engine


def simulate_answer(num_latent_thoughts: int, problem: Dict[str, Any]) -> Dict[str, Any]:
    prompt = problem.get("question") or problem.get("prompt") or ""
    base = len(prompt) + num_latent_thoughts * 3
    predicted = str((base % 91) + 9)
    return {
        "predicted_answer": predicted,
        "num_generated_tokens": 35 + num_latent_thoughts,
        "num_latent_thoughts": num_latent_thoughts,
        "retrieval_success": num_latent_thoughts > 0,
        "retrieved_neighbor_id": f"train_{base % 50}" if num_latent_thoughts else None,
        "retrieval_similarity_score": 0.65 + (num_latent_thoughts * 0.05),
        "nudge_applied": num_latent_thoughts > 0,
        "nudge_magnitude": 0.8 + num_latent_thoughts * 0.1,
        "raw_completion": predicted,
        "faiss_index_size": 200 + base % 80,
        "training_problems_processed": base,
    }


def compare_answers(predicted: Any, ground_truth: Any) -> bool:
    if ground_truth is None:
        return False
    if isinstance(predicted, str) and isinstance(ground_truth, str):
        return predicted.strip().lower() == ground_truth.strip().lower()
    return predicted == ground_truth


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)

    dataset_path = resolve_path(config.get("dataset"))
    dataset = load_dataset(dataset_path)
    max_examples = config.get("max_examples")
    if max_examples not in (None, "None"):
        dataset = dataset[: int(max_examples)]

    run_id = config.get("run_id")
    log_path = Path(__file__).resolve().parent / "results.jsonl"
    logger = ExperimentLogger(log_path, model_name="heuristic", run_id=run_id)

    num_latent_thoughts = int(config.get("num_latent_thoughts", 1))
    max_new_tokens = int(config.get("max_new_tokens", 128))
    temperature = float(config.get("temperature", 0.7))
    do_sample = bool(config.get("do_sample", False))
    dry_run = bool(config.get("dry_run", False))
    engine_module_name = config.get("engine_module", "heuristic.engine")
    model_checkpoint = config.get("model_checkpoint")

    heuristic_memory_cfg = config.get("heuristic_memory") if not dry_run else None
    heuristic_memory = (
        HeuristicMemory(heuristic_memory_cfg) if heuristic_memory_cfg else None
    )

    engine_module = None
    engine = None
    if not dry_run:
        engine_module, engine = ensure_engine(
            engine_module_name,
            model_checkpoint,
            num_latent_thoughts,
            heuristic_memory=heuristic_memory,
        )

    for idx, problem in enumerate(dataset):
        question = (
            problem.get("question")
            or problem.get("prompt")
            or problem.get("input")
            or ""
        )
        ground_truth = problem.get("answer")
        problem_id = problem.get("problem_id") or problem.get("id") or str(idx)

        start = time.perf_counter()
        if dry_run:
            result = simulate_answer(num_latent_thoughts, problem)
        else:
            infer_kwargs = {
                "engine": engine,
                "question": question,
                "ground_truth": ground_truth,
                "num_latent_thoughts": num_latent_thoughts,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
            }
            if heuristic_memory is not None:
                infer_kwargs["heuristic_memory"] = heuristic_memory
            try:
                result = engine_module.infer(**infer_kwargs)
            except TypeError:
                infer_kwargs.pop("heuristic_memory", None)
                result = engine_module.infer(**infer_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        heuristic_payload = {
            "retrieval_success": result.get("retrieval_success", False),
            "retrieved_neighbor_id": result.get("retrieved_neighbor_id"),
            "retrieval_similarity_score": result.get("retrieval_similarity_score"),
            "nudge_applied": result.get("nudge_applied", False),
            "nudge_magnitude": result.get("nudge_magnitude"),
        }

        logger.log_inference(
            problem_id=str(problem_id),
            predicted_answer=result.get("predicted_answer"),
            ground_truth_answer=ground_truth,
            is_correct=compare_answers(result.get("predicted_answer"), ground_truth),
            num_generated_tokens=result.get("num_generated_tokens", 0),
            num_latent_thoughts=result.get("num_latent_thoughts", 0),
            inference_time_ms=elapsed_ms,
            heuristic_metrics=heuristic_payload,
            extra_metrics={
                k: result.get(k)
                for k in (
                    "raw_completion",
                    "faiss_index_size",
                    "training_problems_processed",
                )
                if result.get(k) is not None
            },
        )


if __name__ == "__main__":
    main()
