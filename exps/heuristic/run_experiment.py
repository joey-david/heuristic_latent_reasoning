import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import yaml

from exps.logger import ExperimentLogger
from heuristic import HeuristicMemory, RetrievedHeuristic
from exps.coconut.run_experiment import (
    load_coconut_model,
    extract_final_answer,
    compare_answers,
)
from exps.heuristic.live_plot import LivePlot


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
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["data"] if isinstance(data, dict) and "data" in data else data


def encode_question(tokenizer, question: str, latent_tokens: int, device: torch.device) -> Dict[str, torch.Tensor]:
    prompt = question.rstrip() + "\n"
    if latent_tokens > 0:
        prompt += (
            "<|start-latent|>"
            + ("<|latent|>" * latent_tokens)
            + "<|end-latent|>\n"
        )
    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def first_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model.base_causallm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    return outputs.hidden_states[-1][0, 0, :].detach()


def final_hidden(model, token_ids: torch.Tensor) -> torch.Tensor:
    attn = torch.ones_like(token_ids, device=token_ids.device)
    with torch.no_grad():
        outputs = model.base_causallm(
            input_ids=token_ids,
            attention_mask=attn,
            output_hidden_states=True,
        )
    return outputs.hidden_states[-1][0, -1, :].detach().cpu()


def generate_with_heuristic(
    model_bundle: Tuple[Any, Any, torch.device],
    question: str,
    *,
    num_latent_thoughts: int,
    max_new_tokens: int,
    heuristic_memory: Optional[HeuristicMemory],
) -> Dict[str, Any]:
    model, tokenizer, device = model_bundle
    latent_tokens = max(0, int(num_latent_thoughts))
    inputs = encode_question(tokenizer, question, latent_tokens, device)

    observed_vec = first_hidden(model, inputs["input_ids"], inputs["attention_mask"])
    retrieval: Optional[RetrievedHeuristic] = None
    nudge_vec: Optional[torch.Tensor] = None
    log_info: Dict[str, Any] = {
        "retrieval_success": False,
        "retrieval_similarity_score": None,
        "retrieved_neighbor_id": None,
        "nudge_probability": None,
        "nudge_applied": False,
        "nudge_loss": None,
    }

    if heuristic_memory is not None and latent_tokens > 0:
        nudge_vec, log_info, retrieval = heuristic_memory.get_nudge(observed_vec)

    observed_k0 = observed_vec.detach().cpu()

    latent_nudge = (
        None if nudge_vec is None else nudge_vec.unsqueeze(0).to(device)
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            latent_nudge=latent_nudge,
        )

    output_ids = output_ids.to(device)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids.shape[1] - prompt_len
    completion = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predicted = extract_final_answer(full_text)

    return {
        "predicted": predicted,
        "num_tokens": int(new_tokens),
        "raw_completion": completion,
        "output_ids": output_ids,
        "observed_k0": observed_k0,
        "retrieval": retrieval,
        "log_info": log_info,
    }


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
    dry_run = bool(config.get("dry_run", False))
    if dry_run:
        raise ValueError("Set dry_run=false for heuristic experiments.")

    train_mode = bool(config.get("train_mode", True))
    live_plot = bool(config.get("live_plot", True))
    live_plot_path = config.get("live_plot_path")
    if live_plot_path in (None, "None"):
        live_plot_path = None
    live_plot_interactive = bool(config.get("live_plot_interactive", True))
    baseline_accuracy = float(config.get("baseline_accuracy", 0.341))

    model_checkpoint = config.get("model_checkpoint")
    model_id = config.get("model_id", "openai-community/gpt2")
    if model_checkpoint in (None, "None"):
        raise ValueError("Set `model_checkpoint` in the config file.")

    checkpoint_path = Path(model_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path.resolve()}"
        )

    model_bundle = load_coconut_model(checkpoint_path, model_id)
    heuristic_memory_cfg = config.get("heuristic_memory")
    if heuristic_memory_cfg is None:
        raise ValueError("heuristic_memory configuration is required.")
    if isinstance(heuristic_memory_cfg, (str, Path)):
        heuristic_memory_cfg = load_config(resolve_path(heuristic_memory_cfg))
    if isinstance(heuristic_memory_cfg, dict) and "heuristic_memory" in heuristic_memory_cfg:
        heuristic_memory_cfg = heuristic_memory_cfg["heuristic_memory"]
    heuristic_memory = HeuristicMemory(heuristic_memory_cfg)

    memory_warmup = max(0, int(config.get("memory_warmup", 75)))
    memory_stride = max(1, int(config.get("memory_stride", 2)))
    memory_max_entries = max(1, int(config.get("memory_max_entries", 3000)))
    novelty_cap = float(config.get("memory_novelty_cap", 0.95))
    novelty_cap = max(0.0, min(1.0, novelty_cap))

    stats = {
        "correct": 0,
        "seen": 0,
        "loss_sum": 0.0,
        "loss_count": 0,
        "retrieval_attempts": 0,
        "retrieval_successes": 0,
        "retrieval_success_correct": 0,
        "memory_additions": 0,
    }

    plotter = (
        LivePlot(
            "Heuristic Accuracy",
            baseline=baseline_accuracy,
            save_path=live_plot_path,
            interactive=live_plot_interactive,
        )
        if live_plot
        else None
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
        result = generate_with_heuristic(
            model_bundle,
            question,
            num_latent_thoughts=num_latent_thoughts,
            max_new_tokens=max_new_tokens,
            heuristic_memory=heuristic_memory,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        predicted = result["predicted"]
        tokens = result["num_tokens"]
        raw_completion = result["raw_completion"]
        observed_k0 = result["observed_k0"]
        retrieval = result["retrieval"]
        output_ids = result["output_ids"]
        log_info = result["log_info"]

        attempted_retrieval = heuristic_memory is not None and num_latent_thoughts > 0
        if attempted_retrieval:
            stats["retrieval_attempts"] += 1
            if log_info.get("retrieval_success"):
                stats["retrieval_successes"] += 1
                if is_correct:
                    stats["retrieval_success_correct"] += 1
        log_info["retrieval_attempted"] = attempted_retrieval

        is_correct = compare_answers(predicted, ground_truth)
        stats["seen"] += 1
        stats["correct"] += int(is_correct)
        accuracy = stats["correct"] / stats["seen"]

        loss_value: Optional[float] = None
        prob_value: Optional[float] = log_info.get("nudge_probability")

        if retrieval is not None:
            obs = observed_k0.to(heuristic_memory.device)
            target = torch.tensor([float(is_correct)], device=heuristic_memory.device)
            if train_mode:
                loss_value, prob_value = heuristic_memory.train_step(obs, retrieval, target)
            else:
                prob_value, loss_value = heuristic_memory.eval_step(obs, retrieval, target)
            log_info["nudge_loss"] = loss_value
            log_info["nudge_probability"] = prob_value

        if loss_value is not None:
            stats["loss_sum"] += loss_value
            stats["loss_count"] += 1
            rolling_loss = stats["loss_sum"] / stats["loss_count"]
        else:
            rolling_loss = None

        should_add = False
        if train_mode and heuristic_memory is not None:
            index_size = heuristic_memory.index.ntotal if heuristic_memory.index is not None else 0
            best_sim = float(log_info.get("retrieval_similarity_score") or 0.0)
            retrieval_success = bool(log_info.get("retrieval_success"))
            novel = (not retrieval_success) or (best_sim < novelty_cap)
            if (
                is_correct
                and stats["seen"] > memory_warmup
                and novel
                and (memory_stride <= 1 or (stats["correct"] % memory_stride == 0))
                and index_size < memory_max_entries
            ):
                should_add = True

        if should_add:
            final_latent = final_hidden(model_bundle[0], output_ids)
            heuristic_memory.add(
                observed_k0,
                final_latent,
                metadata={"problem_id": problem_id, "is_correct": bool(is_correct)},
            )
            stats["memory_additions"] += 1
        log_info["memory_candidate_added"] = should_add

        faiss_index_size = (
            heuristic_memory.index.ntotal if heuristic_memory.index is not None else 0
        )
        log_info["memory_index_size"] = faiss_index_size
        retrieval_success_rate = (
            stats["retrieval_successes"] / stats["retrieval_attempts"]
            if stats["retrieval_attempts"] > 0
            else None
        )
        retrieval_frequency = (
            stats["retrieval_attempts"] / stats["seen"] if stats["seen"] > 0 else None
        )
        retrieval_guidance_success = (
            stats["retrieval_success_correct"] / stats["retrieval_successes"]
            if stats["retrieval_successes"] > 0
            else None
        )

        if plotter is not None:
            plotter.update(
                stats["seen"],
                accuracy,
                rolling_loss,
                faiss_entries=faiss_index_size,
                retrieval_success_rate=retrieval_success_rate,
                retrieval_attempts=stats["retrieval_attempts"],
                retrieval_frequency=retrieval_frequency,
                retrieval_guidance_success=retrieval_guidance_success,
            )

        logger.log_inference(
            problem_id=str(problem_id),
            predicted_answer=predicted,
            ground_truth_answer=ground_truth,
            is_correct=is_correct,
            num_generated_tokens=tokens,
            num_latent_thoughts=int(num_latent_thoughts),
            inference_time_ms=elapsed_ms,
            heuristic_metrics={
                "retrieval_success": log_info.get("retrieval_success"),
                "retrieval_attempted": log_info.get("retrieval_attempted"),
                "retrieved_neighbor_id": log_info.get("retrieved_neighbor_id"),
                "retrieval_similarity_score": log_info.get("retrieval_similarity_score"),
                "nudge_probability": log_info.get("nudge_probability"),
                "nudge_applied": log_info.get("nudge_applied"),
                "nudge_loss": log_info.get("nudge_loss"),
                "memory_candidate_added": log_info.get("memory_candidate_added"),
                "memory_index_size": log_info.get("memory_index_size"),
            },
            extra_metrics={
                "raw_completion": raw_completion,
                "faiss_index_size": faiss_index_size,
                "training_problems_processed": stats["seen"],
                "rolling_accuracy": accuracy,
                "rolling_loss": rolling_loss,
                "retrieval_success_rate": retrieval_success_rate,
                "retrieval_attempts": stats["retrieval_attempts"],
                "retrieval_successes": stats["retrieval_successes"],
                "retrieval_frequency": retrieval_frequency,
                "retrieval_guidance_success": retrieval_guidance_success,
                "baseline_accuracy": baseline_accuracy,
                "mode": "train" if train_mode else "test",
                "memory_additions": stats["memory_additions"],
            },
        )

    if train_mode:
        heuristic_memory.save_index()
        heuristic_memory.save_nudge_net()


if __name__ == "__main__":
    main()
