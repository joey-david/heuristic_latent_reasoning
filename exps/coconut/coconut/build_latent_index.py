import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

from knot.retrieval import LatentIndex  # noqa: E402
from knot.utils import canonicalize_answer, ensure_directory  # noqa: E402
from src.model import Coconut  # noqa: E402
from exps.coconut.run_experiment import (  # noqa: E402
    DEFAULT_CONFIG,
    compute_latent_tokens,
    load_config,
    resolve_path,
    generate_answer,
    load_coconut_model,
    load_dataset,
)


def _resolve_question(problem: Dict[str, Any]) -> str:
    return (
        problem.get("question")
        or problem.get("prompt")
        or problem.get("input")
        or ""
    )


def _compute_final_hidden(
    coconut: Coconut,
    inputs: Dict[str, torch.Tensor],
) -> np.ndarray:
    with torch.no_grad():
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        attention_mask = inputs["attention_mask"]
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        outputs = coconut.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
    return outputs.final_hidden[0].detach().cpu().numpy()


def _iter_dataset(
    dataset: List[Dict[str, Any]],
    max_examples: Optional[int],
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    limit = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    for idx in range(limit):
        yield idx, dataset[idx]


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)

    train_path = resolve_path(
        config.get("train_dataset"), Path("data/gsm_train.json")
    )
    dataset = load_dataset(train_path)

    checkpoint = resolve_path(
        config.get("model_checkpoint"),
        Path("data/checkpoints/gsm/gsm-coconut"),
    )
    model_id = config.get("model_id", "openai-community/gpt2")
    c_thought = int(config.get("c_thought", 1))
    max_latent_stage = int(config.get("max_latent_stage", 0))
    max_new_tokens = int(config.get("max_new_tokens", 160))

    retrieval_cfg = config.get("retrieval") or {}
    out_cache = resolve_path(
        retrieval_cfg.get("train_cache"),
        Path("outputs/coconut_train_latents.jsonl"),
    )
    index_path = resolve_path(
        retrieval_cfg.get("index_path"),
        Path("outputs/coconut_train.faiss"),
    )
    metadata_path = resolve_path(
        retrieval_cfg.get("metadata_path"),
        Path("outputs/coconut_train.faiss.meta.json"),
    )
    max_examples = retrieval_cfg.get("max_cache_examples")
    if max_examples not in (None, "None"):
        max_examples = int(max_examples)
    else:
        max_examples = None
    filter_correct = bool(retrieval_cfg.get("filter_correct", True))
    normalize = bool(retrieval_cfg.get("normalize", True))

    print(f"[builder] Loading Coconut model ({model_id}) and checkpoint: {checkpoint}")
    coconut, tokenizer, device = load_coconut_model(checkpoint, model_id)
    print("[builder] Model ready; starting cache build")
    model_bundle = (coconut, tokenizer, device)

    ensure_directory(out_cache)
    ensure_directory(index_path)
    ensure_directory(metadata_path)

    kept = 0
    total = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    with out_cache.open("w", encoding="utf-8") as cache_handle, tqdm(
        total=total, desc="[builder] Caching Coconut latents", unit="ex"
    ) as pbar:
        for idx, problem in _iter_dataset(dataset, max_examples):
            question = _resolve_question(problem)
            gold_answer = problem.get("answer") or problem.get("target") or ""

            latent_tokens, _ = compute_latent_tokens(
                problem,
                c_thought=c_thought,
                max_latent_stage=max_latent_stage,
                override_stage=None,
                override_tokens=None,
            )

            prompt = question.rstrip() + "\n"
            if latent_tokens > 0:
                prompt += (
                    "<|start-latent|>"
                    + ("<|latent|>" * latent_tokens)
                    + "<|end-latent|>\n"
                )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            final_hidden = _compute_final_hidden(coconut, inputs)

            predicted, num_tokens, _ = generate_answer(
                model_bundle,
                question,
                latent_tokens=latent_tokens,
                max_new_tokens=max_new_tokens,
            )
            canonical_base = canonicalize_answer(predicted or "")
            canonical_gold = canonicalize_answer(gold_answer or "")

            if not (filter_correct and canonical_base != canonical_gold):
                record = {
                    "id": idx,
                    "question": question,
                    "gold_answer": gold_answer,
                    "gold_canonical": canonical_gold,
                    "base_answer": predicted,
                    "canonical_base": canonical_base,
                    "hidden_state": final_hidden.tolist(),
                    "tokens": num_tokens,
                }
                cache_handle.write(json.dumps(record, ensure_ascii=True))
                cache_handle.write("\n")
                kept += 1
            pbar.set_postfix(kept=kept)
            pbar.update(1)

    if kept == 0:
        raise RuntimeError("No latent records written; aborting index build.")

    print("[builder] Building FAISS index from cache")
    index = LatentIndex.from_cache(out_cache, normalize=normalize)
    index.save(index_path, metadata_path)
    print(f"[builder] Saved index -> {index_path}\n[builder] Saved meta  -> {metadata_path}")


if __name__ == "__main__":
    main()
