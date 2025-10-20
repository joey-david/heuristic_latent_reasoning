import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_jsonl(path: Path) -> Dict[str, Dict]:
    """Loads a JSONL file into a dict keyed by problem_id."""
    records: Dict[str, Dict] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation log: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            problem_id = str(entry["problem_id"])
            records[problem_id] = entry
    return records


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def build_report(
    dataset_path: Path,
    cot_path: Path,
    coconut_path: Path,
    heuristic_path: Path,
) -> List[Dict]:
    dataset = load_dataset(dataset_path)
    cot = load_jsonl(cot_path)
    coconut = load_jsonl(coconut_path)
    heuristic = load_jsonl(heuristic_path)

    report: List[Dict] = []
    for idx, item in enumerate(dataset):
        problem_id = str(item.get("problem_id") or item.get("id") or idx)
        question = item.get("question") or item.get("prompt") or item.get("input") or ""
        answer = item.get("answer")

        def extract(model_records: Dict[str, Dict], label: str) -> Tuple[bool, int]:
            record = model_records.get(problem_id)
            if record is None:
                raise KeyError(f"Problem {problem_id} missing from {label} results.")
            tokens = int(record.get("num_generated_tokens", 0))
            correct = bool(record.get("is_correct", False))
            return correct, tokens

        cot_correct, cot_tokens = extract(cot, "cot")
        coconut_correct, coconut_tokens = extract(coconut, "coconut")
        heuristic_correct, heuristic_tokens = extract(heuristic, "heuristic")

        report.append(
            {
                "problem_id": problem_id,
                "question": question,
                "answer": answer,
                "models": {
                    "cot": {"correct": cot_correct, "tokens": cot_tokens},
                    "coconut": {
                        "correct": coconut_correct,
                        "tokens": coconut_tokens,
                    },
                    "faiss_augmented": {
                        "correct": heuristic_correct,
                        "tokens": heuristic_tokens,
                    },
                },
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile GSM evaluation logs into a YAML report."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/gsm_test.json"),
        help="Dataset path used during evaluation.",
    )
    parser.add_argument(
        "--cot-log",
        type=Path,
        default=Path("exps/cot/results.jsonl"),
        help="JSONL log produced by the COT baseline.",
    )
    parser.add_argument(
        "--coconut-log",
        type=Path,
        default=Path("exps/coconut/results.jsonl"),
        help="JSONL log produced by the coconut baseline.",
    )
    parser.add_argument(
        "--faiss-log",
        type=Path,
        default=Path("exps/heuristic/results.jsonl"),
        help="JSONL log produced by the FAISS-augmented model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exps/results/gsm_eval.yaml"),
        help="Destination YAML file.",
    )
    args = parser.parse_args()
    report = build_report(args.dataset, args.cot_log, args.coconut_log, args.faiss_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)


if __name__ == "__main__":
    main()
