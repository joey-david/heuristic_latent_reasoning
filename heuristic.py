"""
Lean FAISS-backed heuristic memory for latent nudging.

This module keeps the implementation compact on purpose. We assume configs are
well-formed and omit defensive checks. The memory stores projected keys/values,
retrieves the closest trace with FAISS, and trains a small nudging network using
an explicit utility signal (whether the nudge helped over the baseline run).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # Optional dependency resolved in the environment.
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install faiss-cpu or faiss-gpu to use heuristic memory.") from exc

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass
class HeuristicConfig:
    latent_dim: int
    key_dim: int
    value_dim: int
    index_path: Path
    metadata_path: Path
    weights_path: Path
    min_similarity: float = 0.9
    gate_threshold: float = 0.5
    lr: float = 3e-4
    max_entries: int = 0
    add_if_correct: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "HeuristicConfig":
        return cls(
            latent_dim=int(raw["latent_dim"]),
            key_dim=int(raw["key_dim"]),
            value_dim=int(raw["value_dim"]),
            index_path=Path(raw["index_path"]),
            metadata_path=Path(raw["metadata_path"]),
            weights_path=Path(raw["weights_path"]),
            min_similarity=float(raw.get("min_similarity", 0.9)),
            gate_threshold=float(raw.get("gate_threshold", 0.5)),
            lr=float(raw.get("lr", 3e-4)),
            max_entries=int(raw.get("max_entries", 0)),
            add_if_correct=bool(raw.get("add_if_correct", True)),
        )


@dataclass
class HeuristicMatch:
    index: int
    similarity: float
    key: Tensor
    value: Tensor
    final_state: Tensor
    meta: Dict[str, Any]


class KeyProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden = max(output_dim * 2, input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class NudgingNet(nn.Module):
    def __init__(self, key_dim: int, value_dim: int, latent_dim: int) -> None:
        super().__init__()
        hidden = max(latent_dim // 2, key_dim + value_dim)
        self.net = nn.Sequential(
            nn.Linear(key_dim * 2 + value_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, observed_key: Tensor, memory_key: Tensor, memory_value: Tensor) -> Tensor:
        x = torch.cat((observed_key, memory_key, memory_value), dim=-1)
        return self.net(x)


class HeuristicMemory:
    def __init__(self, raw_config: Dict[str, Any]) -> None:
        self.cfg = HeuristicConfig.from_dict(raw_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.key_projector = KeyProjector(self.cfg.latent_dim, self.cfg.key_dim).to(self.device)
        self.value_projector = KeyProjector(self.cfg.latent_dim, self.cfg.value_dim).to(self.device)
        self.nudging_net = NudgingNet(self.cfg.key_dim, self.cfg.value_dim, self.cfg.latent_dim).to(
            self.device
        )
        self.gate = nn.Linear(self.cfg.latent_dim, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.nudging_net.parameters()) + list(self.gate.parameters()),
            lr=self.cfg.lr,
        )
        self.bce = nn.BCEWithLogitsLoss()

        self.entries: List[Dict[str, Any]] = []
        self.index = faiss.IndexFlatIP(self.cfg.key_dim)

        self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.weights_path.parent.mkdir(parents=True, exist_ok=True)

        self._load_metadata()
        self._load_weights()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_metadata(self) -> None:
        if not self.cfg.metadata_path.exists():
            return
        payload = torch.load(self.cfg.metadata_path, map_location="cpu")
        self.entries = []
        for item in payload:
            entry = {
                "key": item["key"].to(self.device),
                "value": item["value"].to(self.device),
                "final": item["final"].to(self.device),
                "meta": dict(item.get("meta", {})),
                "utility": float(item.get("utility", 0.0)),
            }
            self.entries.append(entry)
        if self.entries:
            self._rebuild_index()

    def _load_weights(self) -> None:
        if not self.cfg.weights_path.exists():
            return
        payload = torch.load(self.cfg.weights_path, map_location=self.device)
        self.key_projector.load_state_dict(payload["key_projector"])
        self.value_projector.load_state_dict(payload["value_projector"])
        self.nudging_net.load_state_dict(payload["nudging_net"])
        self.gate.load_state_dict(payload["gate"])
        self.optimizer.load_state_dict(payload["optimizer"])

    def save(self) -> None:
        self._save_metadata()
        self._save_weights()

    def _save_metadata(self) -> None:
        payload = []
        for entry in self.entries:
            payload.append(
                {
                    "key": entry["key"].detach().cpu(),
                    "value": entry["value"].detach().cpu(),
                    "final": entry["final"].detach().cpu(),
                    "meta": entry["meta"],
                    "utility": float(entry["utility"]),
                }
            )
        torch.save(payload, self.cfg.metadata_path)

    def _save_weights(self) -> None:
        payload = {
            "key_projector": self.key_projector.state_dict(),
            "value_projector": self.value_projector.state_dict(),
            "nudging_net": self.nudging_net.state_dict(),
            "gate": self.gate.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(payload, self.cfg.weights_path)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def lookup(self, observed: Tensor) -> Optional[HeuristicMatch]:
        if not self.entries:
            return None
        self.key_projector.eval()
        observed = observed.to(self.device)
        key = self.key_projector(observed.unsqueeze(0))
        key = F.normalize(key, dim=-1)
        scores, indices = self.index.search(key.detach().cpu().numpy(), 1)
        idx = int(indices[0][0])
        similarity = float(scores[0][0])
        if idx < 0 or similarity < self.cfg.min_similarity:
            return None
        entry = self.entries[idx]
        return HeuristicMatch(
            index=idx,
            similarity=similarity,
            key=entry["key"],
            value=entry["value"],
            final_state=entry["final"],
            meta=entry["meta"],
        )

    @torch.no_grad()
    def preview_nudge(self, observed: Tensor, match: HeuristicMatch) -> Tuple[Tensor, float]:
        self.key_projector.eval()
        self.nudging_net.eval()
        self.gate.eval()
        observed_key = self.key_projector(observed.unsqueeze(0).to(self.device))
        observed_key = F.normalize(observed_key, dim=-1)
        memory_key = match.key.unsqueeze(0).to(self.device)
        memory_value = match.value.unsqueeze(0).to(self.device)
        nudge = self.nudging_net(observed_key, memory_key, memory_value)
        prob = torch.sigmoid(self.gate(nudge)).item()
        return nudge.squeeze(0).detach(), prob

    def train(self, observed: Tensor, match: HeuristicMatch, target: float) -> Tuple[float, float]:
        self.key_projector.train()
        self.value_projector.train()
        self.nudging_net.train()
        self.gate.train()

        observed = observed.to(self.device)
        memory_key = match.key.unsqueeze(0).to(self.device)
        memory_value = match.value.unsqueeze(0).to(self.device)
        target_tensor = torch.tensor([[target]], device=self.device)

        self.optimizer.zero_grad()
        observed_key = self.key_projector(observed.unsqueeze(0))
        observed_key = F.normalize(observed_key, dim=-1)
        nudge = self.nudging_net(observed_key, memory_key, memory_value)
        logit = self.gate(nudge)
        cls_loss = self.bce(logit, target_tensor)
        recon_loss = F.mse_loss(nudge, match.final_state.unsqueeze(0).to(self.device))
        loss = cls_loss + 0.05 * recon_loss
        loss.backward()
        self.optimizer.step()

        prob = torch.sigmoid(logit.detach()).item()
        return float(loss.item()), prob

    def update_utility(self, index: int, utility: float) -> None:
        if 0 <= index < len(self.entries):
            current = float(self.entries[index].get("utility", 0.0))
            self.entries[index]["utility"] = 0.8 * current + 0.2 * float(utility)

    def add_example(
        self,
        observed: Tensor,
        final_state: Tensor,
        *,
        meta: Dict[str, Any],
        utility: float = 0.0,
    ) -> None:
        observed = observed.to(self.device)
        final_state = final_state.to(self.device)

        with torch.no_grad():
            key = self.key_projector(observed.unsqueeze(0))
            key = F.normalize(key, dim=-1).squeeze(0).detach()
            value = self.value_projector(final_state.unsqueeze(0)).squeeze(0).detach()

        entry = {
            "key": key,
            "value": value,
            "final": final_state.detach(),
            "meta": dict(meta),
            "utility": float(utility),
        }
        self.entries.append(entry)
        self._add_to_index(key)
        self._prune_if_needed()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _add_to_index(self, key: Tensor) -> None:
        vec = key.detach().cpu().numpy().astype("float32")
        vec = vec.reshape(1, -1)
        self.index.add(vec)

    def _rebuild_index(self) -> None:
        self.index = faiss.IndexFlatIP(self.cfg.key_dim)
        if not self.entries:
            return
        matrix = torch.stack([entry["key"] for entry in self.entries])
        vec = matrix.detach().cpu().numpy().astype("float32")
        self.index.add(vec)

    def _prune_if_needed(self) -> None:
        limit = self.cfg.max_entries
        if limit <= 0 or len(self.entries) <= limit:
            return
        self.entries.sort(key=lambda item: item.get("utility", 0.0), reverse=True)
        del self.entries[limit:]
        self._rebuild_index()

    # Convenience
    @property
    def gate_threshold(self) -> float:
        return self.cfg.gate_threshold

    def __len__(self) -> int:
        return len(self.entries)

