from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import yaml

from knot.pipeline import run_pipeline

RUNNER_CONFIG_PATH = Path("configs/runner.yaml")


def _load_runner_entries(path: Path) -> Iterator[Tuple[Path, Optional[List[str]]]]:
    """Yields (config_path, override_steps) pairs from the runner config."""
    if not path.exists():
        raise FileNotFoundError(f"Runner config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    entries = payload.get("configs", [])
    for entry in entries:
        if isinstance(entry, str):
            yield Path(entry), None
            continue
        if not isinstance(entry, dict):
            raise ValueError("Each runner entry must be a string path or a mapping.")
        if not entry.get("enabled", True):
            continue
        if "path" not in entry:
            raise ValueError("Runner entry missing required 'path' field.")
        config_path = Path(entry["path"])
        steps_raw = entry.get("steps")
        steps = list(steps_raw) if steps_raw else None
        yield config_path, steps


def main() -> None:
    """Loads the runner config and executes requested pipelines."""
    ran_any = False
    for config_path, steps in _load_runner_entries(RUNNER_CONFIG_PATH):
        print(f"[kNoT] Running config {config_path} with steps {steps or 'default'}")
        run_pipeline(config_path, override_steps=steps)
        ran_any = True
    if not ran_any:
        print("[kNoT] Runner config specified no active pipelines.")


if __name__ == "__main__":
    main()
