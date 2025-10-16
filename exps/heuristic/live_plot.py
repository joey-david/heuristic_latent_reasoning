from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt


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
        self.fig, (self.ax_perf, self.ax_memory) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 9)
        )
        self.ax_perf.set_title(title)
        self.ax_memory.set_xlabel("examples")
        self.ax_perf.set_ylabel("accuracy (%)")
        self.ax_memory.set_ylabel("counts")

        self.ax_loss = self.ax_perf.twinx()
        self.ax_loss.set_ylabel("avg loss")
        self.ax_success = self.ax_memory.twinx()
        self.ax_success.set_ylabel("rates (%)")

        (self.acc_line,) = self.ax_perf.plot(
            [], [], label="accuracy (%)", color="#1f77b4"
        )
        (self.loss_line,) = self.ax_loss.plot(
            [], [], label="avg loss", color="#ff7f0e"
        )
        (self.entry_line,) = self.ax_memory.plot(
            [], [], label="faiss entries", color="#2ca02c"
        )
        (self.attempt_line,) = self.ax_memory.plot(
            [], [], label="retrieval attempts", color="#17becf"
        )
        (self.success_line,) = self.ax_success.plot(
            [], [], label="retrieval success (%)", color="#d62728"
        )
        (self.freq_line,) = self.ax_success.plot(
            [], [], label="retrieval frequency (%)", color="#9467bd"
        )
        (self.guidance_line,) = self.ax_success.plot(
            [], [], label="guided accuracy (%)", color="#8c564b"
        )

        self.perf_steps: list[int] = []
        self.acc_vals: list[float] = []
        self.loss_steps: list[int] = []
        self.loss_vals: list[float] = []
        self.memory_steps: list[int] = []
        self.entry_vals: list[int] = []
        self.attempt_steps: list[int] = []
        self.attempt_vals: list[int] = []
        self.success_steps: list[int] = []
        self.success_vals: list[float] = []
        self.freq_steps: list[int] = []
        self.freq_vals: list[float] = []
        self.guidance_steps: list[int] = []
        self.guidance_vals: list[float] = []

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

        if save_path is not None:
            self.save_path: Optional[Path] = Path(save_path)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.save_path = None

        self.save_every: Optional[int] = None
        if save_every is None:
            self.save_every = 1 if self.save_path is not None else None
        elif save_every > 0:
            self.save_every = int(save_every)

    def _refresh_legends(self) -> None:
        perf_handles = [self.acc_line, self.loss_line]
        if self.baseline_line is not None:
            perf_handles.append(self.baseline_line)
        self.ax_perf.legend(
            perf_handles, [handle.get_label() for handle in perf_handles], loc="lower right"
        )
        memory_handles = [
            self.entry_line,
            self.attempt_line,
            self.success_line,
            self.freq_line,
            self.guidance_line,
        ]
        self.ax_memory.legend(
            memory_handles,
            [handle.get_label() for handle in memory_handles],
            loc="upper left",
        )

    def update(
        self,
        step: int,
        accuracy: float,
        avg_loss: float | None = None,
        *,
        faiss_entries: int | None = None,
        retrieval_success_rate: float | None = None,
        retrieval_attempts: int | None = None,
        retrieval_frequency: float | None = None,
        retrieval_guidance_success: float | None = None,
    ) -> None:
        self.perf_steps.append(step)
        self.acc_vals.append(accuracy * 100.0)
        self.acc_line.set_data(self.perf_steps, self.acc_vals)

        if avg_loss is not None:
            self.loss_steps.append(step)
            self.loss_vals.append(avg_loss)
        self.loss_line.set_data(self.loss_steps, self.loss_vals)

        if faiss_entries is not None:
            self.memory_steps.append(step)
            self.entry_vals.append(faiss_entries)
        self.entry_line.set_data(self.memory_steps, self.entry_vals)

        if retrieval_attempts is not None:
            self.attempt_steps.append(step)
            self.attempt_vals.append(retrieval_attempts)
        self.attempt_line.set_data(self.attempt_steps, self.attempt_vals)

        if retrieval_success_rate is not None:
            self.success_steps.append(step)
            self.success_vals.append(retrieval_success_rate * 100.0)
        self.success_line.set_data(self.success_steps, self.success_vals)

        if retrieval_frequency is not None:
            self.freq_steps.append(step)
            self.freq_vals.append(retrieval_frequency * 100.0)
        self.freq_line.set_data(self.freq_steps, self.freq_vals)

        if retrieval_guidance_success is not None:
            self.guidance_steps.append(step)
            self.guidance_vals.append(retrieval_guidance_success * 100.0)
        self.guidance_line.set_data(self.guidance_steps, self.guidance_vals)

        self.ax_perf.relim()
        self.ax_perf.autoscale_view()
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.ax_memory.relim()
        self.ax_memory.autoscale_view()
        self.ax_success.relim()
        self.ax_success.autoscale_view()

        self.fig.canvas.draw()

        if self.interactive:
            self.fig.canvas.flush_events()
            plt.pause(0.01)

        if self.save_path is not None and self.save_every is not None:
            if len(self.perf_steps) % self.save_every == 0:
                self.fig.savefig(self.save_path, bbox_inches="tight")
