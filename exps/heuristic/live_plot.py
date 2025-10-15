import matplotlib.pyplot as plt


class LivePlot:
    def __init__(self, title: str, baseline: float | None = None) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel("examples")
        self.ax.set_ylabel("metric")
        (self.acc_line,) = self.ax.plot([], [], label="accuracy")
        (self.loss_line,) = self.ax.plot([], [], label="avg_loss")
        self.ax.legend(loc="lower right")
        self.steps: list[int] = []
        self.acc_vals: list[float] = []
        self.loss_vals: list[float] = []
        self.baseline = baseline
        if baseline is not None:
            self.ax.axhline(baseline * 100.0, color="gray", linestyle="--", linewidth=1)

    def update(self, step: int, accuracy: float, avg_loss: float | None = None) -> None:
        from math import isnan

        loss = avg_loss if avg_loss is not None else float("nan")
        self.steps.append(step)
        self.acc_vals.append(accuracy * 100.0)
        self.loss_vals.append(loss)
        self.acc_line.set_data(self.steps, self.acc_vals)
        if not all(isnan(v) for v in self.loss_vals):
            self.loss_line.set_data(self.steps, self.loss_vals)
        else:
            self.loss_line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
