import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import yaml

from utils import ensure_transformers_no_torchvision

ensure_transformers_no_torchvision()

from transformers import AutoModelForCausalLM, AutoTokenizer

from exps.logger import ExperimentLogger


DEFAULT_DATASET = Path("data/gsm_test.json")
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def resolve_path(path_value: Any, default: Path) -> Path:
    if path_value in (None, "None"):
        return default
    return Path(path_value)


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


def load_model_and_tokenizer(
    checkpoint: Path, model_id: str
) -> Tuple[Any, Any, torch.device]:
    if checkpoint is None:
        raise ValueError("A --model-checkpoint path is required for evaluation.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_answer(
    model_bundle: Tuple[Any, Any, torch.device],
    question: str,
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> Tuple[str, int, str]:
    model, tokenizer, device = model_bundle
    prompt = question.rstrip() + "\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)

    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    predicted = extract_final_answer(full_text)
    return predicted, int(new_tokens.size(0)), completion


def simulate_answer(question: str) -> Tuple[str, int, str]:
    fake_answer = str(len(question) % 17)
    fake_tokens = len(question.split()) + 20
    return fake_answer, fake_tokens, fake_answer


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    if "###" in text:
        answer = text.split("###")[-1]
    else:
        answer = text.split("#")[-1]

    return answer.replace(",", "").strip()


def compare_answers(predicted: Any, ground_truth: Any) -> bool:
    if ground_truth is None:
        return False
    if isinstance(predicted, str) and isinstance(ground_truth, str):
        return predicted.strip().lower() == ground_truth.strip().lower()
    return predicted == ground_truth


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)

    dataset_path = resolve_path(config.get("dataset"), DEFAULT_DATASET)
    dataset = load_dataset(dataset_path)
    max_examples = config.get("max_examples")
    if max_examples not in (None, "None"):
        dataset = dataset[: int(max_examples)]

    checkpoint = resolve_path(
        config.get("model_checkpoint"), Path("data/checkpoints/gsm/gsm-cot")
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint}. "
            "Run dl_datasets_and_chkpts.bash to download it."
        )

    log_path = Path(__file__).resolve().parent / "results.jsonl"
    logger = ExperimentLogger(log_path, model_name="cot", run_id=config.get("run_id"))

    model_id = config.get("model_id", "openai-community/gpt2")

    dry_run = bool(config.get("dry_run", False))
    generation_cfg = config.get("generation", {})
    max_new_tokens = int(config.get("max_new_tokens", generation_cfg.get("max_new_tokens", 160)))
    do_sample = bool(config.get("do_sample", generation_cfg.get("do_sample", False)))
    temperature = float(
        config.get("temperature", generation_cfg.get("temperature", 0.0))
    )

    if dry_run:
        model_bundle = None
    else:
        model_bundle = load_model_and_tokenizer(checkpoint, model_id)

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
            predicted, tokens, raw_completion = simulate_answer(question)
        else:
            predicted, tokens, raw_completion = generate_answer(
                model_bundle,
                question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
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
