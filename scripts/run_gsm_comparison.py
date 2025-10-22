from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

COT_CONFIG = REPO_ROOT / "exps/cot/config.yaml"
COT_LOG = REPO_ROOT / "exps/cot/results.jsonl"
COT_METRICS = REPO_ROOT / "outputs/gsm_cot1_metrics.json"

COCONUT_CONFIG = REPO_ROOT / "exps/coconut/config.yaml"
COCONUT_LOG = REPO_ROOT / "exps/coconut/results.jsonl"
COCONUT_METRICS = REPO_ROOT / "outputs/gsm_coconut_metrics.json"

KNOT_RUNNER = REPO_ROOT / "knot.py"
KNOT_METRICS = REPO_ROOT / "outputs/gsm_metrics.json"
DIRECT_METRICS = REPO_ROOT / "outputs/gsm_direct_metrics.json"

REPORT_SCRIPT = REPO_ROOT / "knot_report.py"


def _run(cmd: Iterable[str]) -> None:
    cmd_display = " ".join(cmd)
    print(f"[kNoT-comparison] Running: {cmd_display}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _load_jsonl(path: Path) -> Tuple[int, float, float]:
    if not path.exists():
        raise FileNotFoundError(f"Expected log not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    total = len(records)
    accuracy = sum(1 for record in records if record.get("is_correct")) / total
    avg_tokens = sum(int(record.get("num_generated_tokens", 0)) for record in records) / total
    return total, accuracy, avg_tokens


def _write_metrics(path: Path, *, total: int, accuracy: float, avg_tokens: float) -> None:
    payload = {
        "total": total,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[kNoT-comparison] Wrote metrics to {path}")


def ensure_cot_metrics() -> None:
    if COT_METRICS.exists():
        print(f"[kNoT-comparison] Found existing CoT metrics at {COT_METRICS}, skipping eval.")
        return
    _run(["python", str(COT_CONFIG.parent / "run_experiment.py"), str(COT_CONFIG)])
    total, accuracy, avg_tokens = _load_jsonl(COT_LOG)
    _write_metrics(COT_METRICS, total=total, accuracy=accuracy, avg_tokens=avg_tokens)


def ensure_coconut_metrics() -> None:
    if COCONUT_METRICS.exists():
        print(f"[kNoT-comparison] Found existing Coconut metrics at {COCONUT_METRICS}, skipping eval.")
        return
    _run(["python", str(COCONUT_CONFIG.parent / "run_experiment.py"), str(COCONUT_CONFIG)])
    total, accuracy, avg_tokens = _load_jsonl(COCONUT_LOG)
    _write_metrics(COCONUT_METRICS, total=total, accuracy=accuracy, avg_tokens=avg_tokens)


def ensure_knot_metrics() -> None:
    if KNOT_METRICS.exists():
        print(f"[kNoT-comparison] Found existing kNoT metrics at {KNOT_METRICS}, skipping pipeline run.")
    else:
        _run(["python", str(KNOT_RUNNER)])

    if not KNOT_METRICS.exists():
        raise FileNotFoundError(
            f"Expected kNoT metrics at {KNOT_METRICS} after pipeline run."
        )
    knot_payload = json.loads(KNOT_METRICS.read_text(encoding="utf-8"))

    direct_stale = (
        (not DIRECT_METRICS.exists())
        or KNOT_METRICS.stat().st_mtime > DIRECT_METRICS.stat().st_mtime
    )
    if direct_stale:
        direct_payload = {
            "total": knot_payload.get("total", 0),
            "accuracy": knot_payload.get("base_accuracy", 0.0),
            "avg_tokens": knot_payload.get("avg_tokens", 0.0),
        }
        DIRECT_METRICS.write_text(json.dumps(direct_payload, indent=2), encoding="utf-8")
        print(f"[kNoT-comparison] Derived Direct metrics at {DIRECT_METRICS}")
    else:
        print(f"[kNoT-comparison] Direct metrics present at {DIRECT_METRICS}, leaving as-is.")


def generate_report() -> None:
    _run(["python", str(REPORT_SCRIPT)])


def main() -> None:
    ensure_cot_metrics()
    ensure_coconut_metrics()
    ensure_knot_metrics()
    generate_report()
    print("[kNoT-comparison] Completed GSM8K CoT / Coconut / kNoT comparison.")


if __name__ == "__main__":
    main()
