from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression

from . import utils


@dataclass
class GateDatasetEntry:
    """Container for a single gate training example."""
    features: List[float]
    label: int


class LogisticGate:
    """Logistic-regression-based override gate."""
    def __init__(
        self,
        threshold: float = 0.5,
        model: LogisticRegression | None = None,
    ) -> None:
        self.threshold = threshold
        self.model = model or LogisticRegression(max_iter=1000, solver="lbfgs")

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
    ) -> None:
        """Trains the logistic regression model."""
        if not features:
            return
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32)
        self.model.fit(x, y)

    def predict(self, features: Sequence[float]) -> float:
        """Returns P(correct | features) for the base answer."""
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        proba = self.model.predict_proba(x)[0][1]
        return float(proba)

    def save(self, path: Path) -> None:
        """Persists the trained gate and metadata."""
        utils.ensure_directory(path)
        dump({"threshold": self.threshold, "model": self.model}, str(path))

    @classmethod
    def load(cls, path: Path) -> "LogisticGate":
        """Restores a saved gate from disk."""
        payload = load(str(path))
        model = payload["model"]
        threshold = float(payload.get("threshold", 0.5))
        return cls(threshold=threshold, model=model)


def write_gate_dataset(path: Path, entries: Iterable[GateDatasetEntry]) -> None:
    """Writes a gate training dataset to disk for inspection."""
    records = [
        {"features": entry.features, "label": entry.label}
        for entry in entries
    ]
    utils.write_jsonl(path, records)
