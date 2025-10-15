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
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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

    similarity: float
    key_proj: Tensor
    value_proj: Tensor
    metadata: Optional[Dict[str, Any]]


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

        self.faiss_index_path = Path(config["faiss_index_path"])
        self.metadata_path = Path(config["metadata_path"])

        self.k_neighbors = int(config.get("k_neighbors", 1))
        self.retrieval_threshold = float(config.get("retrieval_threshold", 0.0))

        self.load_index()

    # --------------------------------------------------------------------- #
    # Offline utilities                                                     #
    # --------------------------------------------------------------------- #

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

        k0 = k0.to(self.device)
        kn = kn.to(self.device)

        key_proj = self.key_projector(k0.unsqueeze(0)).squeeze(0)
        value_proj = self.value_projector(kn.unsqueeze(0)).squeeze(0)

        self._add_to_index(key_proj.unsqueeze(0).cpu())
        self.metadata.append(
            {
                "key_proj": key_proj.detach().cpu(),
                "value_proj": value_proj.detach().cpu(),
                "extra": metadata or {},
            }
        )

    # --------------------------------------------------------------------- #
    # Online retrieval                                                      #
    # --------------------------------------------------------------------- #

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
        if self.index is None or self.index.ntotal == 0:
            return None

        observed_k0 = observed_k0.to(self.device)
        observed_proj = self.key_projector(observed_k0.unsqueeze(0))
        observed_proj_norm = F.normalize(observed_proj, dim=-1)

        similarities, indices = self.index.search(
            observed_proj_norm.detach().cpu().numpy(), self.k_neighbors
        )

        sim_score = float(similarities[0][0])
        if sim_score < self.retrieval_threshold or indices[0][0] < 0:
            return None

        meta = self.metadata[indices[0][0]]
        retrieved_key = F.normalize(meta["key_proj"], dim=-1)
        retrieved_value = meta["value_proj"]

        return RetrievedHeuristic(
            similarity=sim_score,
            key_proj=retrieved_key.to(self.device),
            value_proj=retrieved_value.to(self.device),
            metadata=meta.get("extra"),
        )

    def get_nudge(self, observed_k0: Tensor) -> Tuple[Optional[Tensor], Dict[str, Any]]:
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
            return None, {
                "retrieval_success": False,
                "retrieval_similarity_score": None,
                "retrieved_neighbor_id": None,
            }

        observed_proj = self.key_projector(observed_k0.unsqueeze(0))
        nudge = self.nudging_net(
            observed_proj,
            retrieval.key_proj.unsqueeze(0),
            retrieval.value_proj.unsqueeze(0),
        )

        log_info = {
            "retrieval_success": True,
            "retrieval_similarity_score": retrieval.similarity,
            "retrieved_neighbor_id": retrieval.metadata.get("problem_id")
            if retrieval.metadata
            else None,
        }
        return nudge.squeeze(0), log_info

    # --------------------------------------------------------------------- #
    # Persistence                                                           #
    # --------------------------------------------------------------------- #

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

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _init_index(self, key_dim: int) -> None:
        """Initialises an empty FAISS index with the given dimensionality."""
        self.index = faiss.IndexFlatIP(key_dim)

    def _add_to_index(self, projected_keys: Tensor) -> None:
        """Normalises and adds keys to the FAISS index."""
        if self.index is None:
            raise RuntimeError("FAISS index has not been initialised.")

        keys = projected_keys.float()
        keys = F.normalize(keys, dim=-1)
        self.index.add(keys.cpu().numpy())
