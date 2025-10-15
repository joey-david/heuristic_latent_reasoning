import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from exps.logger import ExperimentLogger


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


def ensure_engine(args: argparse.Namespace) -> Tuple[Any, Any]:
    module_name = args.engine_module or "heuristic.engine"
    module = importlib.import_module(module_name)

    if not hasattr(module, "load_engine") or not hasattr(module, "infer"):
        raise AttributeError(
            f"Module {module_name!r} must define load_engine() and infer()."
        )

    engine = module.load_engine(
        model_checkpoint=args.model_checkpoint,
        num_latent_thoughts=args.num_latent_thoughts,
    )
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
    parser = argparse.ArgumentParser(
        description="Run the heuristic GPT-2 nudging model and log metrics."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-checkpoint", type=str)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--num-latent-thoughts", type=int, default=1)
    parser.add_argument("--engine-module", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.max_examples:
        dataset = dataset[: args.max_examples]

    log_path = Path(__file__).resolve().parent / "results.jsonl"
    logger = ExperimentLogger(log_path, model_name="heuristic", run_id=args.run_id)

    engine_module = None
    engine = None
    if not args.dry_run:
        engine_module, engine = ensure_engine(args)

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
        if args.dry_run:
            result = simulate_answer(args.num_latent_thoughts, problem)
        else:
            result = engine_module.infer(
                engine=engine,
                question=question,
                ground_truth=ground_truth,
                num_latent_thoughts=args.num_latent_thoughts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
            )
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
