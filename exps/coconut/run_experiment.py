import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

import torch
import yaml
import numpy as np

from utils import ensure_transformers_no_torchvision

ensure_transformers_no_torchvision()

from transformers import AutoModelForCausalLM, AutoTokenizer

from knot.retrieval import LatentIndex
from src.model import Coconut
from exps.logger import ExperimentLogger


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(candidate: Any, default: Path) -> Path:
    if candidate in (None, "None"):
        return default
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


def load_coconut_model(
    checkpoint: Path, model_id: str
) -> Tuple[Coconut, AutoTokenizer, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    base_model.resize_token_embeddings(len(tokenizer))

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    coconut = Coconut(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    state_dict = torch.load(checkpoint, map_location="cpu")
    coconut.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coconut.to(device)
    coconut.eval()
    return coconut, tokenizer, device


def compute_latent_tokens(
    problem: Dict[str, Any],
    *,
    c_thought: int,
    max_latent_stage: int,
    override_stage: Optional[int],
    override_tokens: Optional[int],
) -> Tuple[int, Optional[int]]:
    if override_tokens is not None:
        return max(0, override_tokens), override_stage

    if override_stage is not None:
        stage = max(0, override_stage)
    else:
        steps = problem.get("steps") or []
        stage = min(max_latent_stage, len(steps))

    return stage * c_thought, stage


def generate_answer(
    model_bundle: Tuple[Coconut, AutoTokenizer, torch.device],
    question: str,
    *,
    latent_tokens: int,
    max_new_tokens: int,
) -> Tuple[str, int, str]:
    model, tokenizer, device = model_bundle
    prompt = question.rstrip() + "\n"
    if latent_tokens > 0:
        prompt += (
            "<|start-latent|>"
            + ("<|latent|>" * latent_tokens)
            + "<|end-latent|>"
            + "\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predicted = extract_final_answer(full_text)
    return predicted, int(new_tokens.size(0)), completion


def simulate_answer(question: str, latent_tokens: int) -> Tuple[str, int, str]:
    fake_answer = str((len(question) + latent_tokens) % 23)
    fake_tokens = len(question.split()) + 25 + latent_tokens
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


def t_softmax(values: List[float], temperature: float) -> List[float]:
    if not values:
        return []
    if temperature <= 0:
        return [1.0 / len(values)] * len(values)
    scaled = [v / temperature for v in values]
    max_scaled = max(scaled)
    exps = [math.exp(v - max_scaled) for v in scaled]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in exps]


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)

    dataset_path = resolve_path(config.get("dataset"), Path("data/gsm_test.json"))
    dataset = load_dataset(dataset_path)
    max_examples = config.get("max_examples")
    if max_examples not in (None, "None"):
        dataset = dataset[: int(max_examples)]

    checkpoint = resolve_path(
        config.get("model_checkpoint"), Path("data/checkpoints/gsm/gsm-coconut")
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint}. "
            "Run dl_datasets_and_chkpts.bash to download it."
        )

    model_id = config.get("model_id", "openai-community/gpt2")
    c_thought = int(config.get("c_thought", 1))
    max_latent_stage = int(config.get("max_latent_stage", 0))
    override_stage = (
        None
        if config.get("latent_stage") in (None, "None")
        else int(config.get("latent_stage"))
    )
    override_tokens = (
        None
        if config.get("latent_tokens") in (None, "None")
        else int(config.get("latent_tokens"))
    )
    max_new_tokens = int(config.get("max_new_tokens", 160))
    dry_run = bool(config.get("dry_run", False))

    retrieval_cfg = config.get("retrieval") or {}
    retrieval_enabled = bool(retrieval_cfg.get("enabled", False))
    retrieval_k = int(retrieval_cfg.get("k", 5))
    retrieval_temperature = float(retrieval_cfg.get("temperature", 0.0))
    retrieval_alpha = float(retrieval_cfg.get("alpha", 0.0))
    retrieval_threshold = float(retrieval_cfg.get("threshold", 0.0))
    retrieval_cache = resolve_path(
        retrieval_cfg.get("train_cache"),
        Path("outputs/coconut_train_latents.jsonl"),
    )
    retrieval_index_path = resolve_path(
        retrieval_cfg.get("index_path"),
        Path("outputs/coconut_train.faiss"),
    )
    retrieval_metadata_path = resolve_path(
        retrieval_cfg.get("metadata_path"),
        Path("outputs/coconut_train.faiss.meta.json"),
    )

    log_path = Path(__file__).resolve().parent / "results.jsonl"
    logger = ExperimentLogger(log_path, model_name="coconut", run_id=config.get("run_id"))

    if dry_run:
        model_bundle = None
    else:
        model_bundle = load_coconut_model(checkpoint, model_id)
    index: Optional[LatentIndex] = None
    vec_by_id: Dict[int, np.ndarray] = {}
    if not dry_run and retrieval_enabled:
        index = LatentIndex.load(
            retrieval_cache,
            retrieval_index_path,
            retrieval_metadata_path,
        )
        vec_by_id = {rec.idx: rec.hidden_state for rec in index.records}

    for idx, problem in enumerate(dataset):
        question = (
            problem.get("question")
            or problem.get("prompt")
            or problem.get("input")
            or ""
        )
        ground_truth = problem.get("answer")
        problem_id = problem.get("problem_id") or problem.get("id") or str(idx)

        latent_tokens, stage_used = compute_latent_tokens(
            problem,
            c_thought=c_thought,
            max_latent_stage=max_latent_stage,
            override_stage=override_stage,
            override_tokens=override_tokens,
        )

        start = time.perf_counter()
        applied_nudge = False
        if dry_run:
            predicted, tokens, raw_completion = simulate_answer(
                question, latent_tokens
            )
            top_similarity = None
            neighbor_count = 0
        else:
            assert model_bundle is not None
            model, tokenizer, device = model_bundle
            prompt = question.rstrip() + "\n"
            if latent_tokens > 0:
                prompt += (
                    "<|start-latent|>"
                    + ("<|latent|>" * latent_tokens)
                    + "<|end-latent|>\n"
                )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            latent_nudge = None
            top_similarity = None
            neighbor_count = 0
            neighbors = []

            if index is not None:
                labels = inputs["input_ids"].clone()
                with torch.no_grad():
                    outputs = model.forward(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=labels,
                        position_ids=torch.arange(
                            0,
                            inputs["input_ids"].shape[1],
                            dtype=torch.long,
                            device=device,
                        ).unsqueeze(0),
                    )
                final_hidden = outputs.final_hidden[0].detach().cpu().numpy()
                neighbors = index.search(final_hidden, k=retrieval_k)
                neighbor_count = len(neighbors)
                sims = [n.similarity for n in neighbors]
                if neighbors:
                    weights = t_softmax(sims, retrieval_temperature)
                    neighbor_mean = np.zeros_like(final_hidden)
                    for weight, neighbor in zip(weights, neighbors):
                        neighbor_mean += weight * vec_by_id[neighbor.idx]
                    delta = neighbor_mean - final_hidden
                    if retrieval_alpha != 0.0:
                        latent_vector = retrieval_alpha * delta
                        latent_nudge = torch.tensor(
                            latent_vector, dtype=torch.float32, device=device
                        )
                    top_similarity = sims[0]
            applied_nudge = (
                latent_nudge is not None
                and top_similarity is not None
                and top_similarity >= retrieval_threshold
            )

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    latent_nudge=latent_nudge,
                    retrieval_similarity=top_similarity,
                    retrieval_threshold=retrieval_threshold,
                )

            new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
            raw_completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predicted = extract_final_answer(full_text)
            tokens = int(new_tokens.size(0))
        elapsed_ms = (time.perf_counter() - start) * 1000

        extras = {
            "raw_completion": raw_completion,
            "configured_latent_tokens": latent_tokens,
            "configured_latent_stage": stage_used,
        }
        if not dry_run and retrieval_enabled:
            extras["retrieval_top_similarity"] = top_similarity
            extras["retrieval_neighbor_count"] = neighbor_count
            extras["retrieval_nudge_applied"] = applied_nudge

        logger.log_inference(
            problem_id=str(problem_id),
            predicted_answer=predicted,
            ground_truth_answer=ground_truth,
            is_correct=compare_answers(predicted, ground_truth),
            num_generated_tokens=tokens,
            num_latent_thoughts=latent_tokens,
            inference_time_ms=elapsed_ms,
            extra_metrics=extras,
        )


if __name__ == "__main__":
    main()
