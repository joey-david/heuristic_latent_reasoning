"""
Heuristic memory components built on top of FAISS for latent-guided reasoning.

The module exposes three key classes:

- :class:`KeyProjector` projects language model latent states into a lower
  dimensional space that can be indexed efficiently by FAISS.
- :class:`NudgingNet` consumes projected latent states from the current problem
  and a retrieved heuristic to produce a guidance vector (``nudge``).
- :class:`HeuristicMemory` orchestrates FAISS indexing, neural projections, and
  nudge computation, providing both offline index-building and online lookup
  utilities.
"""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency handled at runtime
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Tensor = torch.Tensor


class KeyProjector(nn.Module):
    """
    Projects the high-dimensional LLM latent state to a lower-dimensional key space.

    The projector is implemented as a lightweight multi-layer perceptron (MLP)
    that operates on the latent representation of the first token (e.g. the BOS
    token). The resulting vectors are suitable for insertion into a FAISS index.
    """

    def __init__(self, input_dim: int, key_dim: int, hidden_dim: Optional[int] = None) -> None:
        """
        Args:
            input_dim: Dimensionality of the original latent state (e.g. 768).
            key_dim: Target dimensionality for the projected key/value.
            hidden_dim: Optional hidden size for the projection MLP. Defaults to
                ``max(key_dim * 2, input_dim)`` which provides sufficient capacity
                without over-parameterising the projector.
        """
        super().__init__()
        hidden = hidden_dim or max(key_dim * 2, input_dim)
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, key_dim),
        )

    def forward(self, latent_state: Tensor) -> Tensor:
        """
        Args:
            latent_state: Tensor of shape ``(batch, seq_len, input_dim)`` or
                ``(batch, input_dim)`` containing latent activations.

        Returns:
            Tensor of shape ``(batch, key_dim)`` with the projected vectors.
        """
        if latent_state.dim() == 3:
            first_token = latent_state[:, 0, :]
        elif latent_state.dim() == 2:
            first_token = latent_state
        else:
            raise ValueError(
                f"Expected latent_state with 2 or 3 dimensions, got {latent_state.shape}"
            )
        return self.mlp(first_token)


class NudgingNet(nn.Module):
    """
    Computes a nudge vector for the LLM based on a retrieved heuristic.
    """

    def __init__(self, key_dim: int, value_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        Args:
            key_dim: Dimensionality of the projected keys.
            value_dim: Dimensionality of the projected values.
            hidden_dim: Hidden size used inside the internal MLP.
            output_dim: Dimensionality of the resulting nudge vector, matching the
                base model's latent dimension.
        """
        super().__init__()
        mlp_input_dim = key_dim * 2 + value_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.GELU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.GELU(), # Let's start with a single hidden layer
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        observed_k0_proj: Tensor,
        retrieved_k0_proj: Tensor,
        retrieved_kn_proj: Tensor,
    ) -> Tensor:
        """
        Args:
            observed_k0_proj: Projected latent state of the current problem.
            retrieved_k0_proj: Projected latent state of the retrieved heuristic.
            retrieved_kn_proj: Projected final latent state (value) of the heuristic.

        Returns:
            Tensor of shape ``(batch, output_dim)`` containing the nudge vectors.
        """
        concatenated = torch.cat(
            (observed_k0_proj, retrieved_k0_proj, retrieved_kn_proj), dim=-1
        )
        return self.mlp(concatenated)


@dataclass
class RetrievedHeuristic:
    """Container for objects returned after a successful FAISS lookup."""

    index: int
    similarity: float
    key_proj: Tensor
    value_proj: Tensor
    extra: Optional[Dict[str, Any]]
    raw_k0: Optional[Tensor] = None
    raw_kn: Optional[Tensor] = None


class HeuristicMemory:
    """
    Manages the FAISS index, neural projectors, and the nudging logic.

    The class provides offline utilities to build and persist an index of
    heuristics, as well as online search and nudge computation facilities to be
    used during inference.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the FAISS index, projectors, and nudging network from a config.

        Args:
            config: Dictionary containing the parameters described in
                ``args/gsm_heuristic.yaml`` under ``heuristic_memory``.
        """
        if faiss is None:
            raise ImportError(
                "HeuristicMemory requires the 'faiss' Python package. Please install "
                "faiss-cpu or faiss-gpu before using heuristic memory."
            )
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        key_dim = int(config["key_dim"])
        value_dim = int(config["value_dim"])
        llm_latent_dim = int(config["llm_latent_dim"])
        hidden_dim = int(config["nudge_net_hidden_dim"])

        self.faiss_index_path = Path(config["faiss_index_path"])
        self.metadata_path = Path(config["metadata_path"])
        self.warmup_min_entries = int(config.get("warmup_min_entries", 64))
        self.duplicate_threshold = float(config.get("duplicate_threshold", 0.98))
        self.novelty_threshold = config.get("novelty_threshold", None)
        self.novelty_threshold = (
            float(self.novelty_threshold)
            if self.novelty_threshold not in (None, "None")
            else None
        )
        self.nudge_min_prob = float(config.get("nudge_min_prob", 0.6))
        self.nudge_scale_min = float(config.get("nudge_scale_min", 0.05))
        self.nudge_scale_min = max(0.0, min(1.0, self.nudge_scale_min))
        self.nudge_target_norm = float(config.get("nudge_target_norm", 1.9))
        self.nudge_target_norm_min = float(config.get("nudge_target_norm_min", 1.6))
        self.nudge_target_norm_max = float(config.get("nudge_target_norm_max", 2.1))
        self.max_nudge_norm = 1.9
        self.nudge_target_norm_max = min(self.nudge_target_norm_max, self.max_nudge_norm)
        self.nudge_target_norm_min = min(self.nudge_target_norm_min, self.nudge_target_norm_max)
        self.nudge_target_norm = min(self.nudge_target_norm, self.nudge_target_norm_max)
        self.nudge_target_norm = max(self.nudge_target_norm, self.nudge_target_norm_min)
        self.nudge_trust_region_width = max(
            0.0, float(config.get("nudge_trust_region_width", 0.15))
        )
        self.nudge_trust_region_enabled = bool(
            config.get("nudge_trust_region_enabled", True)
        )
        self.nudge_target_step = max(0.0, float(config.get("nudge_target_step", 0.05)))
        self.nudge_advantage_beta = float(config.get("nudge_advantage_beta", 0.1))
        self.nudge_advantage_threshold = float(
            config.get("nudge_advantage_threshold", 0.02)
        )
        self._advantage_ema: Optional[float] = None
        self.nudge_target_apply_rate = config.get("nudge_target_apply_rate", None)
        if self.nudge_target_apply_rate in (None, "None"):
            self.nudge_target_apply_rate = None
        else:
            self.nudge_target_apply_rate = float(self.nudge_target_apply_rate)
            if not 0.0 < self.nudge_target_apply_rate < 1.0:
                self.nudge_target_apply_rate = None
        self.nudge_prob_window = max(1, int(config.get("nudge_prob_window", 256)))
        self.nudge_prob_min_count = max(1, int(config.get("nudge_prob_min_count", 32)))
        self.add_correct_only = bool(config.get("add_correct_only", True))
        self.max_entries = int(config.get("max_entries", 0))
        if self.max_entries < 0:
            self.max_entries = 0
        self.prune_metric = str(config.get("prune_metric", "utility")).lower()
        self.memory_utility_beta = float(config.get("memory_utility_beta", 0.2))
        self.memory_utility_beta = min(max(self.memory_utility_beta, 0.0), 1.0)
        self.memory_utility_floor = float(config.get("memory_utility_floor", -1.0))
        self.memory_utility_ceiling = float(config.get("memory_utility_ceiling", 1.0))

        # FAISS index (cosine similarity via inner product on L2-normalised vectors).
        self.index: Optional[faiss.IndexFlatIP] = None
        self._init_index(key_dim)

        # Metadata aligned with FAISS entries (value projections, raw info).
        self.metadata: List[Dict[str, Any]] = []

        # Neural components
        self.key_projector = KeyProjector(llm_latent_dim, key_dim).to(self.device)
        self.value_projector = KeyProjector(llm_latent_dim, value_dim).to(self.device)
        self.nudging_net = NudgingNet(key_dim, value_dim, hidden_dim, llm_latent_dim).to(
            self.device
        )
        self.nudge_lr = float(config.get("nudge_lr", 1e-4))
        self.optimizer = optim.Adam(self.nudging_net.parameters(), lr=self.nudge_lr)
        default_nudge_weights = self.faiss_index_path.parent / "nudge_weights.pt"
        self.nudge_weights_path = Path(
            config.get(
                "nudge_weights_path",
                default_nudge_weights,
            )
        )

        self.k_neighbors = int(config.get("k_neighbors", 1))
        threshold = config.get("retrieval_threshold")
        if threshold in (None, "None"):
            threshold = config.get("cosine_similarity_threshold")
        if threshold in (None, "None"):
            threshold = 0.88
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = 0.88
        if threshold <= 0.0:
            threshold = 0.88
        self.retrieval_threshold = threshold
        self.dynamic_retrieval_threshold = bool(
            config.get("dynamic_retrieval_threshold", False)
        )
        self.retrieval_target_success = float(
            config.get("retrieval_target_success", 0.4)
        )
        self.retrieval_target_success = min(
            max(self.retrieval_target_success, 0.0), 1.0
        )
        self.retrieval_threshold_window = max(
            1, int(config.get("retrieval_threshold_window", 512))
        )
        self.retrieval_threshold_min_count = max(
            1, int(config.get("retrieval_threshold_min_count", 64))
        )
        self.retrieval_threshold_min = float(config.get("retrieval_threshold_min", 0.8))
        self.retrieval_threshold_max = float(config.get("retrieval_threshold_max", 0.99))
        if self.retrieval_threshold_min > self.retrieval_threshold_max:
            self.retrieval_threshold_min, self.retrieval_threshold_max = (
                self.retrieval_threshold_max,
                self.retrieval_threshold_min,
            )
        self._similarity_history: Optional[deque[float]] = (
            deque(maxlen=self.retrieval_threshold_window)
            if self.dynamic_retrieval_threshold
            else None
        )
        self._current_retrieval_threshold = float(self.retrieval_threshold)

        self._access_counter = 0

        self.load_index()
        self.load_nudge_net()

    # Offline utilities

    def build_index(self, reasoning_traces: Iterable[Dict[str, Tensor]]) -> None:
        """
        Builds (or rebuilds) the FAISS index using a collection of reasoning traces.

        Args:
            reasoning_traces: Iterable of dictionaries with keys ``k0`` and ``kn``
                (full-dimensional latent states). Optional additional metadata will
                be stored alongside the value.
        """
        projected_keys: List[Tensor] = []
        metadata: List[Dict[str, Any]] = []

        for trace in reasoning_traces:
            k0 = trace["k0"].to(self.device)
            kn = trace["kn"].to(self.device)
            extra_meta = {k: v for k, v in trace.items() if k not in {"k0", "kn"}}

            key_proj = self.key_projector(k0.unsqueeze(0)).squeeze(0)
            value_proj = self.value_projector(kn.unsqueeze(0)).squeeze(0)

            projected_keys.append(key_proj.detach().cpu())
            metadata.append(
                {
                    "key_proj": key_proj.detach().cpu(),
                    "value_proj": value_proj.detach().cpu(),
                    "extra": extra_meta,
                    "raw_k0": k0.detach().cpu(),
                    "raw_kn": kn.detach().cpu(),
                }
            )

        if not projected_keys:
            raise ValueError("No reasoning traces supplied to build_index.")

        key_matrix = torch.stack(projected_keys)
        self._init_index(key_matrix.size(-1))
        self._add_to_index(key_matrix)

        self.metadata = metadata
        self.save_index()

    def add(self, k0: Tensor, kn: Tensor, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Adds a new heuristic (key/value pair) to the FAISS index.

        Args:
            k0: First latent state tensor of shape ``(latent_dim,)`` or ``(1, latent_dim)``.
            kn: Final latent state tensor matching the same dimensionality.
            metadata: Optional dictionary with auxiliary information associated with
                the heuristic (e.g., problem identifiers).
        """
        if self.index is None:
            raise RuntimeError("FAISS index has not been initialised.")

        if self.add_correct_only:
            is_correct = bool(metadata and metadata.get("is_correct"))
            if not is_correct:
                return

        k0 = k0.to(self.device)
        kn = kn.to(self.device)

        key_proj = self.key_projector(k0.unsqueeze(0)).squeeze(0).detach()
        value_proj = self.value_projector(kn.unsqueeze(0)).squeeze(0).detach()

        best_sim: Optional[float] = None
        if self.index.ntotal > 0:
            query = F.normalize(key_proj.unsqueeze(0), dim=-1).cpu().numpy()
            similarities, _ = self.index.search(query, 1)
            if similarities.size > 0:
                best_sim = float(similarities[0][0])
                if best_sim >= self.duplicate_threshold:
                    return
                if self.novelty_threshold is not None and best_sim >= self.novelty_threshold:
                    return

        self._add_to_index(key_proj.unsqueeze(0))
        self._access_counter += 1
        extra_meta = dict(metadata) if metadata else {}
        self.metadata.append(
            {
                "key_proj": key_proj.detach().cpu(),
                "value_proj": value_proj.detach().cpu(),
                "extra": extra_meta,
                "raw_k0": k0.detach().cpu(),
                "raw_kn": kn.detach().cpu(),
                "usage_count": 0,
                "last_used_step": self._access_counter,
                "added_step": self._access_counter,
                "utility": float(extra_meta.get("utility", 0.0)),
            }
        )
        self._maybe_prune_index()

    # Online retrieval

    def search(self, observed_k0: Tensor) -> Optional[RetrievedHeuristic]:
        """
        Searches the FAISS index for the best matching heuristic.

        Args:
            observed_k0: Current problem's first latent state tensor of shape
                ``(latent_dim,)`` or ``(1, latent_dim)``.

        Returns:
            :class:`RetrievedHeuristic` if a suitable candidate was found, otherwise
            ``None``.
        """
        if self.index is None or self.index.ntotal < self.warmup_min_entries:
            return None

        observed_k0 = observed_k0.to(self.device)
        observed_proj = self.key_projector(observed_k0.unsqueeze(0))
        observed_proj_norm = F.normalize(observed_proj, dim=-1)

        similarities, indices = self.index.search(
            observed_proj_norm.detach().cpu().numpy(), self.k_neighbors
        )

        sim_score = float(similarities[0][0])
        threshold = self._get_retrieval_threshold()
        self._record_similarity(sim_score)
        if sim_score < threshold or indices[0][0] < 0:
            return None

        meta_idx = int(indices[0][0])
        meta = self.metadata[meta_idx]
        self._access_counter += 1
        meta["last_used_step"] = self._access_counter
        meta["usage_count"] = int(meta.get("usage_count", 0)) + 1
        retrieved_key = F.normalize(meta["key_proj"], dim=-1)
        retrieved_value = meta["value_proj"]
        raw_k0 = meta.get("raw_k0")
        raw_kn = meta.get("raw_kn")

        return RetrievedHeuristic(
            index=meta_idx,
            similarity=sim_score,
            key_proj=retrieved_key.to(self.device),
            value_proj=retrieved_value.to(self.device),
            extra=meta.get("extra"),
            raw_k0=raw_k0.to(self.device) if raw_k0 is not None else None,
            raw_kn=raw_kn.to(self.device) if raw_kn is not None else None,
        )

    def get_nudge(
        self, observed_k0: Tensor
    ) -> Tuple[Optional[Tensor], Dict[str, Any], Optional[RetrievedHeuristic]]:
        """
        Computes the nudge vector for the current problem if a suitable heuristic
        exists in the memory.

        Args:
            observed_k0: First latent state tensor for the current reasoning step.

        Returns:
            Tuple consisting of:
                - The nudge vector (or ``None`` if no heuristic qualifies).
                - A dictionary with logging information about the retrieval.
        """
        retrieval = self.search(observed_k0)
        if retrieval is None:
            return None, self._no_retrieval_log(), None

        observed_proj = self.key_projector(observed_k0.unsqueeze(0))
        nudge = self.nudging_net(
            observed_proj,
            retrieval.key_proj.unsqueeze(0),
            retrieval.value_proj.unsqueeze(0),
        )
        raw_nudge = nudge.squeeze(0)
        raw_norm = raw_nudge.detach().norm().item()

        prob = 1.0

        scaled_nudge, scale, scale_floor, scaled_norm = self._scale_nudge(prob, raw_nudge)

        prob_threshold = None
        apply_nudge = True

        trust_nudge = self._apply_trust_region(scaled_nudge)
        trust_norm = trust_nudge.detach().norm().item()
        gated_nudge = trust_nudge if apply_nudge else torch.zeros_like(trust_nudge)
        inference_nudge = gated_nudge.detach().cpu()
        if not apply_nudge:
            trust_norm = 0.0

        extra = retrieval.extra or {}

        log_info = {
            "retrieval_success": True,
            "retrieval_similarity_score": retrieval.similarity,
            "retrieved_neighbor_id": extra.get("problem_id"),
            "nudge_probability": prob,
            "nudge_probability_threshold": prob_threshold,
            "nudge_applied": apply_nudge,
            "nudge_scale": scale,
            "nudge_scale_floor": scale_floor,
            "nudge_norm_pre_gate": raw_norm,
            "nudge_norm_post_scale": scaled_norm,
            "nudge_norm_returned": trust_norm,
            "retrieval_threshold": float(self._current_retrieval_threshold),
            "retrieval_index": retrieval.index,
        }
        return inference_nudge, log_info, retrieval

    def _no_retrieval_log(self) -> Dict[str, Any]:
        return {
            "retrieval_success": False,
            "retrieval_similarity_score": None,
            "retrieved_neighbor_id": None,
            "nudge_probability": None,
            "nudge_probability_threshold": None,
            "nudge_applied": False,
            "nudge_scale": None,
            "nudge_scale_floor": self.nudge_scale_min,
            "nudge_norm_pre_gate": None,
            "nudge_norm_post_scale": None,
            "nudge_norm_returned": 0.0,
            "retrieval_threshold": float(self._current_retrieval_threshold),
            "retrieval_index": None,
        }

    def _scale_nudge(
        self, prob: float, raw_nudge: Tensor
    ) -> Tuple[Tensor, float, float, float]:
        scale_floor = self.nudge_scale_min
        scale = scale_floor + (1.0 - scale_floor) * prob
        scaled = self._cap_norm(raw_nudge * scale)
        scaled_norm = scaled.detach().norm().item()
        return scaled, scale, scale_floor, scaled_norm

    def train_step(
        self, observed_k0: Tensor, retrieved: RetrievedHeuristic, target: Tensor
    ) -> Tuple[float, float]:
        self.nudging_net.eval()
        return 0.0, 1.0

    def update_feedback(self, index: int, *, helpful: Optional[bool]) -> None:
        """
        Updates the stored utility score for a retrieved heuristic.

        Args:
            index: Index of the heuristic inside the metadata list.
            helpful: ``True`` if the nudge improved the outcome, ``False`` if it hurt,
                and ``None`` if the effect is neutral (e.g. gate skipped).
        """
        if index is None or index < 0 or index >= len(self.metadata):
            return

        entry = self.metadata[index]
        current = float(entry.get("utility", 0.0))
        beta = self.memory_utility_beta
        if helpful is None:
            updated = (1.0 - beta) * current
        else:
            reward = 1.0 if helpful else -1.0
            updated = (1.0 - beta) * current + beta * reward
        updated = max(self.memory_utility_floor, min(self.memory_utility_ceiling, updated))
        entry["utility"] = updated

    def update_norm_target(self, advantage: Optional[float]) -> None:
        """
        Anneals the trust-region target norm based on observed nudge advantage.

        Args:
            advantage: Difference in accuracy between nudged and non-nudged buckets.
        """
        if advantage is None:
            return

        beta = self.nudge_advantage_beta
        if not 0.0 < beta <= 1.0:
            beta = 0.1

        if self._advantage_ema is None:
            self._advantage_ema = float(advantage)
        else:
            self._advantage_ema = (1.0 - beta) * self._advantage_ema + beta * float(advantage)

        threshold = abs(self.nudge_advantage_threshold)
        if self._advantage_ema > threshold:
            self.nudge_target_norm = min(
                self.nudge_target_norm + self.nudge_target_step,
                self.nudge_target_norm_max,
            )
        elif self._advantage_ema < -threshold:
            self.nudge_target_norm = max(
                self.nudge_target_norm - self.nudge_target_step,
                self.nudge_target_norm_min,
            )

    def eval_step(
        self,
        observed_k0: Tensor,
        retrieved: RetrievedHeuristic,
        target: Optional[Tensor] = None,
    ) -> Tuple[float, Optional[float]]:
        self.nudging_net.eval()
        return 1.0, None

    # Persistence

    def save_index(self) -> None:
        """Saves the FAISS index and metadata to disk."""
        if self.index is None:
            return

        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.faiss_index_path))

        with self.metadata_path.open("wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self) -> None:
        """Loads the FAISS index and metadata from disk if they exist."""
        if not self.faiss_index_path.exists() or not self.metadata_path.exists():
            return

        self.index = faiss.read_index(str(self.faiss_index_path))
        with self.metadata_path.open("rb") as f:
            self.metadata = pickle.load(f)

    # Nudge network persistence

    def save_nudge_net(self) -> None:
        self.nudging_net.eval()
        self.nudge_weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "nudging_net": self.nudging_net.state_dict(),
            },
            self.nudge_weights_path,
        )

    def load_nudge_net(self) -> None:
        if self.nudge_weights_path.exists():
            state = torch.load(self.nudge_weights_path, map_location=self.device)
            if "nudging_net" in state:
                self.nudging_net.load_state_dict(state["nudging_net"])
            else:
                self.nudging_net.load_state_dict(state)

    # Internal helpers

    def _init_index(self, key_dim: int) -> None:
        """Initialises an empty FAISS index with the given dimensionality."""
        self.index = faiss.IndexFlatIP(key_dim)

    def _add_to_index(self, projected_keys: Tensor) -> None:
        """Normalises and adds keys to the FAISS index."""
        if self.index is None:
            raise RuntimeError("FAISS index has not been initialised.")

        keys = projected_keys.float().detach()
        keys = F.normalize(keys, dim=-1)
        self.index.add(keys.cpu().numpy())

    def _maybe_prune_index(self) -> None:
        """Keeps the index within the configured capacity."""
        if (
            self.max_entries <= 0
            or self.index is None
            or self.index.ntotal <= self.max_entries
        ):
            return

        removal_idx = self._select_removal_index()
        if removal_idx is None:
            return

        del self.metadata[removal_idx]
        self._rebuild_index_from_metadata()

    def _select_removal_index(self) -> Optional[int]:
        """Selects which metadata entry to evict under capacity pressure."""
        if not self.metadata:
            return None

        candidates = list(range(len(self.metadata)))

        if self.prune_metric == "utility":
            utilities = [float(self.metadata[i].get("utility", 0.0)) for i in candidates]
            min_utility = min(utilities)
            candidates = [
                idx for idx in candidates if float(self.metadata[idx].get("utility", 0.0)) == min_utility
            ]

        last_used = [int(self.metadata[i].get("last_used_step", 0)) for i in candidates]
        min_last_used = min(last_used)
        candidates = [
            idx
            for idx in candidates
            if int(self.metadata[idx].get("last_used_step", 0)) == min_last_used
        ]

        added_steps = [int(self.metadata[i].get("added_step", 0)) for i in candidates]
        min_added = min(added_steps)
        for idx in candidates:
            if int(self.metadata[idx].get("added_step", 0)) == min_added:
                return idx
        return candidates[0] if candidates else None

    def _rebuild_index_from_metadata(self) -> None:
        """Reconstructs the FAISS index from in-memory metadata."""
        key_dim = None
        if self.index is not None:
            key_dim = self.index.d
        elif self.metadata:
            key_dim = int(self.metadata[0]["key_proj"].shape[-1])
        if key_dim is None:
            return

        self._init_index(key_dim)
        if not self.metadata:
            return

        keys = torch.stack([entry["key_proj"] for entry in self.metadata])
        self._add_to_index(keys)

    def _record_similarity(self, similarity: float) -> None:
        """Stores similarity observations for dynamic thresholding."""
        if self._similarity_history is not None:
            self._similarity_history.append(float(similarity))

    def _get_retrieval_threshold(self) -> float:
        """Returns the current retrieval threshold, optionally using a quantile gate."""
        threshold = float(self.retrieval_threshold)
        if (
            self.dynamic_retrieval_threshold
            and self._similarity_history is not None
            and len(self._similarity_history) >= self.retrieval_threshold_min_count
        ):
            quantile = 1.0 - self.retrieval_target_success
            quantile = min(max(quantile, 0.0), 1.0)
            history_tensor = torch.tensor(
                list(self._similarity_history), dtype=torch.float32
            )
            dynamic_threshold = float(torch.quantile(history_tensor, quantile).item())
            dynamic_threshold = max(self.retrieval_threshold_min, dynamic_threshold)
            dynamic_threshold = min(self.retrieval_threshold_max, dynamic_threshold)
            threshold = dynamic_threshold
        self._current_retrieval_threshold = threshold
        return threshold

    def _apply_trust_region(self, nudge_vec: Tensor) -> Tensor:
        """Clamps the returned nudge norm to the configured trust region."""
        if not self.nudge_trust_region_enabled:
            return self._cap_norm(nudge_vec)

        norm = float(nudge_vec.norm().item())
        if norm == 0.0:
            return nudge_vec

        target = float(self.nudge_target_norm)
        target = min(max(target, self.nudge_target_norm_min), self.nudge_target_norm_max)
        upper = min(target + self.nudge_trust_region_width, self.max_nudge_norm)
        lower = max(0.0, min(target - self.nudge_trust_region_width, upper))
        desired_norm = min(upper, max(lower, norm))
        if abs(desired_norm - norm) < 1e-6:
            return self._cap_norm(nudge_vec)

        scaled = nudge_vec * (desired_norm / norm)
        return self._cap_norm(scaled)

    def _cap_norm(self, nudge_vec: Tensor) -> Tensor:
        """Caps the nudge norm at the configured maximum."""
        norm = float(nudge_vec.norm().item())
        if norm == 0.0 or norm <= self.max_nudge_norm:
            return nudge_vec
        return nudge_vec * (self.max_nudge_norm / norm)
