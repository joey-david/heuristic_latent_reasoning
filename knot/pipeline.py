from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    Config,
    DataSplitConfig,
    GateConfig,
    IndexConfig,
    load_config,
)
from .retrieval import LatentIndex, Neighbor
from . import utils
from .gate import GateDatasetEntry, LogisticGate, write_gate_dataset


def _resolve_device(device: str) -> torch.device:
    """Selects execution device, defaulting to CUDA when available."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    """Maps string dtype identifiers to torch dtypes."""
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(dtype.lower(), torch.float32)


class ModelRunner:
    """Handles model/tokenizer setup and latent extraction."""
    def __init__(self, config: Config) -> None:
        """Initializes tokenizer/model pair according to config."""
        self.model_cfg = config.model
        self.gen_cfg = config.generation
        self.prompt_cfg = config.prompt
        self.device = _resolve_device(self.model_cfg.device)
        dtype = _resolve_dtype(self.model_cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.name, torch_dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, question: str) -> Tuple[str, np.ndarray, int]:
        """Runs single-pass generation and returns answer, latent, and token count."""
        prompt = self.prompt_cfg.template.format(question=question.strip())
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=self.gen_cfg.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        if self.gen_cfg.do_sample:
            gen_kwargs["temperature"] = max(self.gen_cfg.temperature, 1e-5)
            gen_kwargs["top_p"] = self.gen_cfg.top_p
        with torch.no_grad():
            generation = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        full_sequence = generation.sequences[0]
        generated_ids = full_sequence[prompt_len:]

        final_hidden = self._extract_final_hidden(generation)

        answer = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        if self.prompt_cfg.strip_response:
            answer = answer.split("\n")[0].strip()

        return answer, final_hidden, int(generated_ids.shape[0])

    def _extract_final_hidden(self, generation) -> np.ndarray:
        """Pulls the final token hidden state from the generate() output."""
        hidden_states = getattr(generation, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Generation did not return hidden states.")
        final_step = hidden_states[-1]
        if isinstance(final_step, (list, tuple)):
            final_layer = final_step[-1]
        else:
            final_layer = final_step
        if final_layer.dim() == 3:
            vec = final_layer[0, -1, :]
        elif final_layer.dim() == 2:
            vec = final_layer[0, :]
        else:
            raise RuntimeError("Unexpected hidden state shape.")
        return vec.detach().float().cpu().numpy()


def _load_dataset(split: DataSplitConfig) -> List[Dict[str, str]]:
    """Loads a JSON dataset and applies the optional limit."""
    data = utils.load_json(split.path)
    if split.limit is not None:
        data = data[: split.limit]
    items = []
    for idx, sample in enumerate(data):
        items.append(
            {
                "id": idx,
                "question": sample["question"],
                "answer": sample["answer"],
            }
        )
    return items


def extract_latents(runner: ModelRunner, split: DataSplitConfig) -> None:
    """Generates answers and caches latents for a dataset split."""
    records = []
    dataset = _load_dataset(split)
    iterator = tqdm(
        dataset,
        desc=f"[kNoT] Extracting latents -> {split.cache}",
        unit="record",
        total=len(dataset),
        leave=False,
    )
    for sample in iterator:
        answer, hidden, tokens = runner.generate(sample["question"])
        base_canonical = utils.canonicalize_answer(answer)
        gold_canonical = utils.canonicalize_answer(sample["answer"])
        records.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "gold_canonical": gold_canonical,
                "base_answer": answer,
                "canonical_base": base_canonical,
                "hidden_state": hidden.tolist(),
                "tokens": tokens,
            }
        )
    utils.write_jsonl(split.cache, records)


@dataclass
class VoteResult:
    """Tracks vote weights, similarities, and the chosen winner."""
    winner: str
    winner_raw: str
    winner_weight: float
    base_weight: float
    weights_by_answer: Dict[str, float]
    raw_by_answer: Dict[str, str]
    neighbor_counts: Dict[str, int]
    similarities: List[float]
    weights: List[float]
    max_similarity: float
    mean_similarity: float
    total_weight: float


def _vote(
    record: Dict[str, Any],
    neighbors: List[Neighbor],
    cfg: IndexConfig,
) -> VoteResult:
    """Aggregates neighbor votes using the configured weighting."""
    base_canonical = record["canonical_base"]
    base_answer = record["base_answer"]

    if not neighbors:
        weights_by_answer = {base_canonical: cfg.base_prior}
        return VoteResult(
            winner=base_canonical,
            winner_raw=base_answer,
            winner_weight=weights_by_answer[base_canonical],
            base_weight=weights_by_answer[base_canonical],
            weights_by_answer=weights_by_answer,
            raw_by_answer={base_canonical: base_answer},
            neighbor_counts={},
            similarities=[],
            weights=[],
            max_similarity=0.0,
            mean_similarity=0.0,
            total_weight=0.0,
        )

    sims = [neighbor.similarity for neighbor in neighbors]
    if cfg.weighting.lower() == "uniform":
        weights = [1.0 for _ in neighbors]
    else:
        weights = utils.softmax(sims, cfg.temperature)

    weights_by_answer: Dict[str, float] = {}
    raw_by_answer: Dict[str, str] = {base_canonical: base_answer}
    neighbor_counts: Dict[str, int] = {}

    for neighbor, weight in zip(neighbors, weights):
        key = neighbor.gold_canonical
        weights_by_answer[key] = weights_by_answer.get(key, 0.0) + weight
        neighbor_counts[key] = neighbor_counts.get(key, 0) + 1
        raw_by_answer.setdefault(key, neighbor.gold_answer)

    weights_by_answer[base_canonical] = weights_by_answer.get(base_canonical, 0.0) + cfg.base_prior
    neighbor_counts.setdefault(base_canonical, 0)

    winner, winner_weight = max(weights_by_answer.items(), key=lambda item: item[1])
    winner_raw = raw_by_answer.get(winner, base_answer)
    base_weight = weights_by_answer.get(base_canonical, 0.0)
    max_similarity = max(sims) if sims else 0.0
    mean_similarity = float(np.mean(sims)) if sims else 0.0
    total_weight = float(sum(weights))

    return VoteResult(
        winner=winner,
        winner_raw=winner_raw,
        winner_weight=winner_weight,
        base_weight=base_weight,
        weights_by_answer=weights_by_answer,
        raw_by_answer=raw_by_answer,
        neighbor_counts=neighbor_counts,
        similarities=sims,
        weights=weights,
        max_similarity=max_similarity,
        mean_similarity=mean_similarity,
        total_weight=total_weight,
    )


def _gate_features(
    vote: VoteResult,
    base_canonical: str,
) -> List[float]:
    """Computes feature vector consumed by the optional gate."""
    base_in_neighbors = 1.0 if vote.neighbor_counts.get(base_canonical, 0) > 0 else 0.0
    return [
        vote.max_similarity,
        vote.mean_similarity,
        vote.winner_weight,
        vote.base_weight,
        base_in_neighbors,
    ]


def _train_gate(
    index: LatentIndex,
    cfg: IndexConfig,
) -> Optional[LogisticGate]:
    """Trains the logistic gate on cached training records."""
    samples = index.records
    features: List[List[float]] = []
    labels: List[int] = []
    dataset_entries: List[GateDatasetEntry] = []
    for record in samples:
        neighbors = index.search(record.hidden_state, cfg.k, exclude_idx=record.idx)
        record_payload = {
            "canonical_base": record.base_canonical,
            "base_answer": record.base_answer,
        }
        vote = _vote(record_payload, neighbors, cfg)
        feats = _gate_features(vote, record.base_canonical)
        base_correct = 1 if record.base_canonical == record.gold_canonical else 0
        features.append(feats)
        labels.append(base_correct)
        dataset_entries.append(GateDatasetEntry(features=feats, label=base_correct))
    if not features:
        return None
    gate = LogisticGate(threshold=cfg.gate.threshold)
    gate.fit(features, labels)
    if cfg.gate.dataset_path:
        write_gate_dataset(cfg.gate.dataset_path, dataset_entries)
    if cfg.gate.state_path:
        gate.save(cfg.gate.state_path)
    return gate


def build_index(cfg: Config) -> LatentIndex:
    """Builds and persists the latent FAISS index."""
    print(f"[kNoT] Building FAISS index from {cfg.index.train_cache}")
    index = LatentIndex.from_cache(cfg.index.train_cache, normalize=cfg.index.normalize)
    index.save(cfg.index.index_path, cfg.index.metadata_path)
    print(f"[kNoT] Saved index to {cfg.index.index_path}")
    return index


def _resolve_index(cfg: Config) -> LatentIndex:
    """Loads a prebuilt index or creates one on demand."""
    index_path = cfg.index.index_path
    metadata_path = cfg.index.metadata_path
    if index_path.exists() and metadata_path.exists():
        return LatentIndex.load(cfg.index.train_cache, index_path, metadata_path)
    print(f"[kNoT] Prebuilt index not found; constructing from {cfg.index.train_cache}")
    return build_index(cfg)


@dataclass
class Decision:
    """Encapsulates the final answer selection and diagnostics."""
    final_canonical: str
    final_answer: str
    confidence: float
    reason: str
    override: bool
    gate_score: Optional[float]
    features: List[float]
    neighbors: List[Neighbor]
    vote: VoteResult


def _decide(
    record: Dict[str, Any],
    hidden: np.ndarray,
    index: LatentIndex,
    cfg: IndexConfig,
    gate: Optional[LogisticGate],
) -> Decision:
    """Selects the final canonical answer for a record."""
    neighbors = index.search(hidden, cfg.k)
    vote = _vote(record, neighbors, cfg)

    features = _gate_features(vote, record["canonical_base"])
    confidence = vote.winner_weight
    final_canonical = record["canonical_base"]
    final_answer = record["base_answer"]
    override = False
    reason = "winner_is_base"
    gate_score: Optional[float] = None

    if confidence < cfg.min_confidence:
        reason = "below_min_confidence"
    elif vote.winner == record["canonical_base"]:
        reason = "winner_is_base"
    elif vote.winner_weight < cfg.override_threshold:
        reason = "below_override_threshold"
    else:
        if gate is not None:
            gate_score = gate.predict(features)
            if gate_score <= cfg.gate.threshold:
                final_canonical = vote.winner
                final_answer = vote.winner_raw
                override = True
                reason = "gate_override"
            else:
                reason = "gate_block"
        else:
            final_canonical = vote.winner
            final_answer = vote.winner_raw
            override = True
            reason = "vote_override"

    return Decision(
        final_canonical=final_canonical,
        final_answer=final_answer,
        confidence=confidence,
        reason=reason,
        override=override,
        gate_score=gate_score,
        features=features,
        neighbors=neighbors,
        vote=vote,
    )


def evaluate(
    cfg: Config,
) -> None:
    """Runs kNoT evaluation using cached latents."""
    eval_records = list(utils.read_jsonl(cfg.data.eval.cache))
    index = _resolve_index(cfg)

    gate: Optional[LogisticGate] = None
    if cfg.index.gate.enabled:
        if cfg.index.gate.state_path and cfg.index.gate.state_path.exists():
            gate = LogisticGate.load(cfg.index.gate.state_path)
            gate.threshold = cfg.index.gate.threshold
        else:
            gate = _train_gate(index, cfg.index)
            if gate is not None:
                gate.threshold = cfg.index.gate.threshold

    predictions = []
    metrics_counter = Counter()
    base_correct = 0
    final_correct = 0
    total_tokens = 0
    for record in eval_records:
        hidden = np.array(record["hidden_state"], dtype=np.float32)
        decision = _decide(record, hidden, index, cfg.index, gate)

        base_is_correct = record["canonical_base"] == record["gold_canonical"]
        final_is_correct = decision.final_canonical == record["gold_canonical"]

        base_correct += int(base_is_correct)
        final_correct += int(final_is_correct)
        total_tokens += int(record.get("tokens", 0))

        transition = (
            "tp" if base_is_correct and final_is_correct
            else "tn" if (not base_is_correct) and (not final_is_correct)
            else "fp" if base_is_correct and not final_is_correct
            else "fn"
        )
        metrics_counter[transition] += 1

        predictions.append(
            {
                "id": record["id"],
                "question": record["question"],
                "gold_answer": record["gold_answer"],
                "gold_canonical": record["gold_canonical"],
                "base_answer": record["base_answer"],
                "base_canonical": record["canonical_base"],
                "final_canonical": decision.final_canonical,
                "base_correct": base_is_correct,
                "final_correct": final_is_correct,
                "override": decision.override,
                "reason": decision.reason,
                "vote_winner_weight": decision.vote.winner_weight,
                "vote_total_weight": decision.vote.total_weight,
                "gate_score": decision.gate_score,
                "vote_similarities": decision.vote.similarities,
                "vote_weights": decision.vote.weights,
                "gate_features": decision.features,
                "neighbors": [
                    {
                        "idx": neighbor.idx,
                        "similarity": neighbor.similarity,
                        "base_canonical": neighbor.base_canonical,
                        "gold_canonical": neighbor.gold_canonical,
                        "base_answer": neighbor.base_answer,
                        "gold_answer": neighbor.gold_answer,
                    }
                    for neighbor in decision.neighbors
                ],
            }
        )

    total = len(eval_records)
    metrics = {
        "total": total,
        "base_accuracy": base_correct / total if total else 0.0,
        "knot_accuracy": final_correct / total if total else 0.0,
        "avg_tokens": total_tokens / total if total else 0.0,
        "improved": metrics_counter["fn"],
        "regressed": metrics_counter["fp"],
        "unchanged_correct": metrics_counter["tp"],
        "unchanged_wrong": metrics_counter["tn"],
    }
    utils.write_json(cfg.output.metrics_path, metrics)
    flips = [
        pred
        for pred in predictions
        if pred["base_correct"] != pred["final_correct"]
    ]
    utils.write_jsonl(cfg.output.flips_path, flips)
    if cfg.output.predictions_path:
        utils.write_jsonl(cfg.output.predictions_path, predictions)


def run_pipeline(
    config_path: Path,
    override_steps: Optional[List[str]] = None,
) -> None:
    """Executes the configured pipeline steps sequentially."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    cfg = load_config(raw_cfg)
    utils.set_seed(cfg.pipeline.seed)

    steps = override_steps if override_steps is not None else (cfg.pipeline.steps or ["extract_train", "extract_eval", "evaluate"])

    requires_model = any(step in {"extract_train", "extract_eval"} for step in steps)
    runner: Optional[ModelRunner] = ModelRunner(cfg) if requires_model else None

    for step in steps:
        if step == "extract_train":
            if runner is None:
                raise RuntimeError("ModelRunner not initialized; cannot extract train latents.")
            extract_latents(runner, cfg.data.train)
        elif step == "extract_eval":
            if runner is None:
                raise RuntimeError("ModelRunner not initialized; cannot extract eval latents.")
            extract_latents(runner, cfg.data.eval)
        elif step == "build_index":
            build_index(cfg)
        elif step == "train_gate":
            index = _resolve_index(cfg)
            gate = _train_gate(index, cfg.index)
            if gate is None:
                print("[kNoT] Gate training skipped (no records available).")
            else:
                print("[kNoT] Gate trained successfully.")
        elif step == "evaluate":
            evaluate(cfg)
        else:
            raise ValueError(f"Unknown pipeline step: {step}")
