import argparse
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


def ensure_model(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.model_checkpoint is None:
        raise ValueError("--model-checkpoint is required unless --dry-run is used.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_answer(
    model_bundle: Tuple[Any, Any, Any],
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> Tuple[str, int, str]:
    model, tokenizer, device = model_bundle
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return extract_final_answer(decoded), int(new_tokens.size(0)), decoded


def simulate_answer(prompt: str) -> Tuple[str, int, str]:
    fake_answer = str(len(prompt) % 17)
    fake_tokens = len(prompt.split()) + 20
    return fake_answer, fake_tokens, fake_answer


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.lower().startswith("answer"):
            return line.split(":", 1)[-1].strip()
    return lines[-1] if lines else text.strip()


def compare_answers(predicted: Any, ground_truth: Any) -> bool:
    if ground_truth is None:
        return False
    if isinstance(predicted, str) and isinstance(ground_truth, str):
        return predicted.strip().lower() == ground_truth.strip().lower()
    return predicted == ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the GPT-2 CoT baseline and log per-problem metrics."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-checkpoint", type=str)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.max_examples:
        dataset = dataset[: args.max_examples]

    log_path = Path(__file__).resolve().parent / "results.jsonl"
    logger = ExperimentLogger(log_path, model_name="cot", run_id=args.run_id)

    model_bundle = None
    if not args.dry_run:
        model_bundle = ensure_model(args)

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
            predicted, tokens, raw_completion = simulate_answer(question)
        else:
            predicted, tokens, raw_completion = generate_answer(
                model_bundle,
                question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.log_inference(
            problem_id=str(problem_id),
            predicted_answer=predicted,
            ground_truth_answer=ground_truth,
            is_correct=compare_answers(predicted, ground_truth),
            num_generated_tokens=tokens,
            num_latent_thoughts=0,
            inference_time_ms=elapsed_ms,
            extra_metrics={"raw_completion": raw_completion},
        )


if __name__ == "__main__":
    main()
