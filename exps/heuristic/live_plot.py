from pathlib import Path
from typing import Dict, Optional, Union

import yaml

import matplotlib.pyplot as plt


class YamlMetricsRecorder:
    """Keeps a bounded YAML summary of recent metrics for LLM consumption."""

    def __init__(
        self,
        path: Path,
        *,
        max_entries: int = 96,
        metadata: Optional[Dict[str, float]] = None,
    ) -> None:
        self.path = path
        self.max_entries = max_entries
        self.metadata = metadata or {}
        self.entries: list[Dict[str, Union[float, int, str]]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: Dict[str, Optional[float]]) -> None:
        filtered: Dict[str, float | int | str] = {}
        for key, value in entry.items():
            if value is None:
                continue
            if isinstance(value, float):
                filtered[key] = float(value)
            elif isinstance(value, int):
                filtered[key] = int(value)
            else:
                filtered[key] = value
        self.entries.append(filtered)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        payload: Dict[str, Union[Dict[str, float], list[Dict[str, Union[float, int, str]]]]] = {
            "history": self.entries
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        with self.path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)


class LivePlot:
    def __init__(
        self,
        title: str,
        baseline: float | None = None,
        *,
        save_path: Optional[Union[str, Path]] = None,
        interactive: bool = True,
        save_every: int = 1,
    ) -> None:
        self.interactive = interactive

        if self.interactive:
            plt.ion()
        else:
            plt.ioff()
        self.fig, (self.ax_perf, self.ax_memory, self.ax_nudge) = plt.subplots(
            3, 1, sharex=True, figsize=(8, 11)
        )
        self.ax_perf.set_title(title)
        self.ax_perf.set_ylabel("accuracy (%)")
        self.ax_memory.set_ylabel("counts")
        self.ax_nudge.set_ylabel("nudge norm")
        self.ax_nudge.set_xlabel("examples")

        self.ax_loss = self.ax_perf.twinx()
        self.ax_loss.set_ylabel("avg loss")
        self.ax_success = self.ax_memory.twinx()
        self.ax_success.set_ylabel("rates (%)")
        self.ax_nudge_scale = self.ax_nudge.twinx()
        self.ax_nudge_scale.set_ylabel("scale / prob")

        (self.acc_line,) = self.ax_perf.plot(
            [], [], label="accuracy (%)", color="#1f77b4"
        )
        (self.nudged_acc_line,) = self.ax_perf.plot(
            [], [], label="nudged acc (%)", color="#bcbd22", linestyle="--"
        )
        (self.non_nudged_acc_line,) = self.ax_perf.plot(
            [], [], label="non-nudged acc (%)", color="#ff9896", linestyle=":"
        )
        (self.loss_line,) = self.ax_loss.plot(
            [], [], label="avg loss", color="#ff7f0e"
        )
        (self.entry_line,) = self.ax_memory.plot(
            [], [], label="faiss entries", color="#2ca02c"
        )
        (self.success_line,) = self.ax_success.plot(
            [], [], label="retrieval success (%)", color="#d62728"
        )
        (self.guidance_line,) = self.ax_success.plot(
            [], [], label="guided accuracy (%)", color="#8c564b"
        )
        (self.applied_line,) = self.ax_success.plot(
            [], [], label="nudge applied (%)", color="#7f7f7f", linestyle="--"
        )
        (self.nudge_norm_line,) = self.ax_nudge.plot(
            [], [], label="mean nudge norm (last5)", color="#17becf"
        )
        (self.nudge_scale_line,) = self.ax_nudge_scale.plot(
            [], [], label="mean nudge scale", color="#d62728", linestyle="--"
        )
        (self.nudge_prob_line,) = self.ax_nudge_scale.plot(
            [], [], label="mean nudge prob", color="#8c564b", linestyle=":"
        )

        self.perf_steps: list[int] = []
        self.acc_vals: list[float] = []
        self.nudged_acc_steps: list[int] = []
        self.nudged_acc_vals: list[float] = []
        self.non_nudged_acc_steps: list[int] = []
        self.non_nudged_acc_vals: list[float] = []
        self.loss_steps: list[int] = []
        self.loss_vals: list[float] = []
        self.memory_steps: list[int] = []
        self.entry_vals: list[int] = []
        self.success_steps: list[int] = []
        self.success_vals: list[float] = []
        self.guidance_steps: list[int] = []
        self.guidance_vals: list[float] = []
        self.applied_steps: list[int] = []
        self.applied_vals: list[float] = []
        self.nudge_norm_steps: list[int] = []
        self.nudge_norm_vals: list[float] = []
        self.nudge_scale_steps: list[int] = []
        self.nudge_scale_vals: list[float] = []
        self.nudge_prob_steps: list[int] = []
        self.nudge_prob_vals: list[float] = []

        self.baseline = baseline
        self.baseline_line = None
        if baseline is not None:
            self.baseline_line = self.ax_perf.axhline(
                baseline * 100.0,
                color="gray",
                linestyle="--",
                linewidth=1,
                label="baseline",
            )

        self._refresh_legends()

        self.yaml_recorder: Optional[YamlMetricsRecorder]

        if save_path is not None:
            self.save_path: Optional[Path] = Path(save_path)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_metadata = {}
            if baseline is not None:
                yaml_metadata["baseline_pct"] = baseline * 100.0
            self.yaml_recorder = YamlMetricsRecorder(
                self.save_path.with_suffix(".yaml"), metadata=yaml_metadata
            )
        else:
            self.save_path = None
            self.yaml_recorder = None

        self.save_every: Optional[int] = None
        if save_every is None:
            self.save_every = 1 if self.save_path is not None else None
        elif save_every > 0:
            self.save_every = int(save_every)

    def _refresh_legends(self) -> None:
        perf_handles = [self.acc_line, self.nudged_acc_line, self.non_nudged_acc_line, self.loss_line]
        if self.baseline_line is not None:
            perf_handles.append(self.baseline_line)
        self.ax_perf.legend(
            perf_handles, [handle.get_label() for handle in perf_handles], loc="lower right"
        )
        memory_handles = [
            self.entry_line,
            self.success_line,
            self.guidance_line,
            self.applied_line,
        ]
        self.ax_memory.legend(
            memory_handles,
            [handle.get_label() for handle in memory_handles],
            loc="upper left",
        )
        nudge_handles = [
            self.nudge_norm_line,
            self.nudge_scale_line,
            self.nudge_prob_line,
        ]
        self.ax_nudge.legend(
            nudge_handles,
            [handle.get_label() for handle in nudge_handles],
            loc="upper right",
        )

    def update(
        self,
        step: int,
        accuracy: float,
        avg_loss: float | None = None,
        *,
        faiss_entries: int | None = None,
        retrieval_success_rate: float | None = None,
        retrieval_guidance_success: float | None = None,
        nudge_norm_window_mean: float | None = None,
        nudge_scale_mean: float | None = None,
        nudge_prob_mean: float | None = None,
        nudge_applied_rate: float | None = None,
        nudged_accuracy: float | None = None,
        non_nudged_accuracy: float | None = None,
    ) -> None:
        self.perf_steps.append(step)
        self.acc_vals.append(accuracy * 100.0)
        self.acc_line.set_data(self.perf_steps, self.acc_vals)

        if nudged_accuracy is not None:
            self.nudged_acc_steps.append(step)
            self.nudged_acc_vals.append(nudged_accuracy * 100.0)
        self.nudged_acc_line.set_data(self.nudged_acc_steps, self.nudged_acc_vals)

        if non_nudged_accuracy is not None:
            self.non_nudged_acc_steps.append(step)
            self.non_nudged_acc_vals.append(non_nudged_accuracy * 100.0)
        self.non_nudged_acc_line.set_data(
            self.non_nudged_acc_steps, self.non_nudged_acc_vals
        )

        if avg_loss is not None:
            self.loss_steps.append(step)
            self.loss_vals.append(avg_loss)
        self.loss_line.set_data(self.loss_steps, self.loss_vals)

        if faiss_entries is not None:
            self.memory_steps.append(step)
            self.entry_vals.append(faiss_entries)
        self.entry_line.set_data(self.memory_steps, self.entry_vals)

        if retrieval_success_rate is not None:
            self.success_steps.append(step)
            self.success_vals.append(retrieval_success_rate * 100.0)
        self.success_line.set_data(self.success_steps, self.success_vals)

        if retrieval_guidance_success is not None:
            self.guidance_steps.append(step)
            self.guidance_vals.append(retrieval_guidance_success * 100.0)
        self.guidance_line.set_data(self.guidance_steps, self.guidance_vals)

        if nudge_applied_rate is not None:
            self.applied_steps.append(step)
            self.applied_vals.append(nudge_applied_rate * 100.0)
        self.applied_line.set_data(self.applied_steps, self.applied_vals)

        if nudge_norm_window_mean is not None:
            self.nudge_norm_steps.append(step)
            self.nudge_norm_vals.append(nudge_norm_window_mean)
        self.nudge_norm_line.set_data(self.nudge_norm_steps, self.nudge_norm_vals)

        if nudge_scale_mean is not None:
            self.nudge_scale_steps.append(step)
            self.nudge_scale_vals.append(nudge_scale_mean)
        self.nudge_scale_line.set_data(self.nudge_scale_steps, self.nudge_scale_vals)

        if nudge_prob_mean is not None:
            self.nudge_prob_steps.append(step)
            self.nudge_prob_vals.append(nudge_prob_mean)
        self.nudge_prob_line.set_data(self.nudge_prob_steps, self.nudge_prob_vals)

        self.ax_perf.relim()
        self.ax_perf.autoscale_view()
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.ax_memory.relim()
        self.ax_memory.autoscale_view()
        self.ax_success.relim()
        self.ax_success.autoscale_view()
        self.ax_nudge.relim()
        self.ax_nudge.autoscale_view()
        self.ax_nudge_scale.relim()
        self.ax_nudge_scale.autoscale_view()

        self.fig.canvas.draw()

        if self.interactive:
            self.fig.canvas.flush_events()
            plt.pause(0.01)

        if self.yaml_recorder is not None:
            self.yaml_recorder.record(
                {
                    "step": int(step),
                    "accuracy_pct": self.acc_vals[-1] if self.acc_vals else None,
                    "nudged_accuracy_pct": self.nudged_acc_vals[-1] if self.nudged_acc_vals else None,
                    "non_nudged_accuracy_pct": self.non_nudged_acc_vals[-1] if self.non_nudged_acc_vals else None,
                    "avg_loss": self.loss_vals[-1] if self.loss_vals else None,
                    "faiss_entries": self.entry_vals[-1] if self.entry_vals else None,
                    "retrieval_success_pct": self.success_vals[-1] if self.success_vals else None,
                    "guided_accuracy_pct": self.guidance_vals[-1] if self.guidance_vals else None,
                    "nudge_applied_pct": self.applied_vals[-1] if self.applied_vals else None,
                    "nudge_norm_last5": self.nudge_norm_vals[-1] if self.nudge_norm_vals else None,
                    "nudge_scale_mean": self.nudge_scale_vals[-1] if self.nudge_scale_vals else None,
                    "nudge_prob_mean": self.nudge_prob_vals[-1] if self.nudge_prob_vals else None,
                }
            )

        if self.save_path is not None and self.save_every is not None:
            if len(self.perf_steps) % self.save_every == 0:
                self.fig.savefig(self.save_path, bbox_inches="tight")
