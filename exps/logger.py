import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """
    Minimal experiment logger used by all run scripts.

    Every record is appended to a JSONL file, and a short status line is printed
    to the console. The logger always stores ``run_id`` and ``model_name``.
    """

    def __init__(self, log_path: Path, model_name: str, run_id: Optional[str] = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    def log_inference(
        self,
        *,
        problem_id: str,
        predicted_answer: Any,
        ground_truth_answer: Any,
        is_correct: bool,
        num_generated_tokens: int,
        num_latent_thoughts: int,
        inference_time_ms: float,
        heuristic_metrics: Optional[Dict[str, Any]] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "model_name": self.model_name,
            "problem_id": problem_id,
            "predicted_answer": predicted_answer,
            "ground_truth_answer": ground_truth_answer,
            "is_correct": bool(is_correct),
            "num_generated_tokens": int(num_generated_tokens),
            "num_latent_thoughts": int(num_latent_thoughts),
            "inference_time_ms": float(inference_time_ms),
        }

        if heuristic_metrics:
            entry.update(heuristic_metrics)
        if extra_metrics:
            entry.update(extra_metrics)

        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

        print(
            f"[{self.model_name}] problem={problem_id} "
            f"correct={entry['is_correct']} "
            f"tokens={entry['num_generated_tokens']} "
            f"time_ms={entry['inference_time_ms']:.2f}"
        )
