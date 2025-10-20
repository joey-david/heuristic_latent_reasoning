from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exps.logger import ExperimentLogger
from heuristic import HeuristicMemory
from exps.coconut.run_experiment import (
    load_coconut_model,
    extract_final_answer,
    compare_answers,
)


DEFAULT_CONFIG = Path(__file__).with_name("config_test.yaml")


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return data["data"] if isinstance(data, dict) and "data" in data else data


def encode_question(tokenizer, question: str, latent_tokens: int, device: torch.device) -> Dict[str, torch.Tensor]:
    prompt = question.rstrip() + "\n"
    if latent_tokens > 0:
        prompt += "<|start-latent|>" + ("<|latent|>" * latent_tokens) + "<|end-latent|>\n"
    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def k0_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model.base_causallm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hidden = outputs.hidden_states[-1]
    token_ids = input_ids[0]
    start_latent_id = getattr(model, "start_latent_id", None)
    latent_positions = (token_ids == start_latent_id).nonzero(as_tuple=False) if start_latent_id else None
    idx = int(latent_positions[0]) if latent_positions is not None and latent_positions.numel() > 0 else int(attention_mask[0].sum().item() - 1)
    return hidden[0, idx, :].detach()


def final_hidden(model, token_ids: torch.Tensor) -> torch.Tensor:
    attn = torch.ones_like(token_ids, device=token_ids.device)
    with torch.no_grad():
        outputs = model.base_causallm(
            input_ids=token_ids,
            attention_mask=attn,
            output_hidden_states=True,
        )
    return outputs.hidden_states[-1][0, -1, :].detach()


def run_generation(
    model_bundle: Tuple[Any, Any, torch.device],
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    latent_nudge: torch.Tensor | None = None,
) -> Dict[str, Any]:
    model, tokenizer, device = model_bundle
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"].clone(),
            attention_mask=inputs["attention_mask"].clone(),
            max_new_tokens=max_new_tokens,
            latent_nudge=latent_nudge,
        )
    output_ids = output_ids.to(device)
    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = output_ids[0, prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predicted = extract_final_answer(full_text)
    return {
        "predicted": predicted,
        "num_tokens": int(max(0, completion_ids.shape[0])),
        "raw_completion": completion,
        "output_ids": output_ids.detach(),
    }


def main() -> None:
    config = load_config(DEFAULT_CONFIG)

    dataset_path = Path(config["dataset"])
    dataset = load_dataset(dataset_path)
    max_examples = min(int(config.get("max_examples", 30000)), 30000)
    dataset = dataset[:max_examples]

    model_bundle = load_coconut_model(Path(config["model_checkpoint"]), config["model_id"])
    model, tokenizer, device = model_bundle

    memory = HeuristicMemory(config["heuristic_memory"])
    num_latent = int(config.get("num_latent_thoughts", 0))
    max_new_tokens = int(config.get("max_new_tokens", 128))
    run_id = config.get("run_id")

    logger = ExperimentLogger(
        Path(__file__).resolve().parent / "results.jsonl",
        model_name="heuristic",
        run_id=run_id,
    )

    stats = {
        "seen": 0,
        "correct": 0,
        "baseline_correct": 0,
        "nudged_correct": 0,
        "improved": 0,
        "retrievals": 0,
        "retrieval_success": 0,
    }
    loss_track: List[float] = []
    loss_curve: List[Tuple[int, float]] = []

    total = len(dataset)

    for problem in tqdm(
        dataset,
        desc=run_id or "heuristic",
        unit="problem",
        dynamic_ncols=True,
        total=total,
    ):
        problem_id = str(problem.get("problem_id") or problem.get("id") or stats["seen"])
        question = problem.get("question") or problem.get("prompt") or problem.get("input") or ""
        ground_truth = problem.get("answer")

        start = time.time()
        encoded = encode_question(tokenizer, question, num_latent, device)
        observed = k0_hidden(model, encoded["input_ids"], encoded["attention_mask"])
        baseline = run_generation(model_bundle, encoded, max_new_tokens)
        baseline_correct = compare_answers(baseline["predicted"], ground_truth)
        baseline_final = final_hidden(model, baseline["output_ids"])

        match = memory.lookup(observed)
        nudged = None
        nudged_correct = False
        prob = None
        similarity = None
        applied = False
        utility = 0

        if match is not None:
            stats["retrievals"] += 1
            similarity = match.similarity
            nudge_vec, prob = memory.preview_nudge(observed, match)
            latent_nudge = nudge_vec.unsqueeze(0).to(device)
            nudged = run_generation(model_bundle, encoded, max_new_tokens, latent_nudge=latent_nudge)
            nudged_correct = compare_answers(nudged["predicted"], ground_truth)
            utility = int(nudged_correct) - int(baseline_correct)
            applied = prob >= memory.gate_threshold
            loss, train_prob = memory.train(observed, match, 1.0 if utility > 0 else 0.0)
            prob = train_prob
            loss_track.append(loss)
            if len(loss_track) % 100 == 0:
                window = loss_track[-100:]
                loss_curve.append((len(loss_track), sum(window) / len(window)))
            memory.update_utility(match.index, utility)
            if utility > 0:
                stats["improved"] += 1
                nudged_final = final_hidden(model, nudged["output_ids"])
                memory.add_example(
                    observed,
                    nudged_final,
                    meta={"problem_id": problem_id, "question": question},
                    utility=float(utility),
                )
            elif utility < 0:
                stats["improved"] += 0  # keeps intent explicit

        if match is None and memory.cfg.add_if_correct and baseline_correct:
            memory.add_example(
                observed,
                baseline_final,
                meta={"problem_id": problem_id, "question": question},
                utility=0.0,
            )

        final_output = nudged if (applied and nudged is not None) else baseline
        is_correct = nudged_correct if (applied and nudged is not None) else baseline_correct

        elapsed = (time.time() - start) * 1000
        stats["seen"] += 1
        stats["correct"] += int(is_correct)
        stats["baseline_correct"] += int(baseline_correct)
        stats["nudged_correct"] += int(nudged_correct)
        if match is not None and utility >= 0:
            stats["retrieval_success"] += int(nudged_correct)

        logger.log_inference(
            problem_id=problem_id,
            predicted_answer=final_output["predicted"],
            ground_truth_answer=ground_truth,
            is_correct=is_correct,
            num_generated_tokens=final_output["num_tokens"],
            num_latent_thoughts=num_latent,
            inference_time_ms=elapsed,
            heuristic_metrics={
                "retrieval_similarity": similarity,
                "nudge_probability": prob,
                "nudge_applied": applied,
                "nudge_utility": utility if match is not None else None,
            },
            extra_metrics={
                "baseline_correct": bool(baseline_correct),
                "nudged_correct": bool(nudged_correct),
                "memory_size": len(memory),
            },
        )

    memory.save()
    if loss_curve:
        xs, ys = zip(*loss_curve)
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Training updates")
        plt.ylabel("Avg loss (last 100)")
        plt.title("Nudging Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(plot_dir / "nudging_loss.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
