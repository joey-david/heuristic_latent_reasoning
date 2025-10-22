import json
import math
import os
import random
import re
from decimal import Decimal, InvalidOperation, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import numpy as np
import torch

getcontext().prec = 28


def set_seed(seed: int) -> None:
    """Seeds python, numpy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_directory(path: Path) -> None:
    """Creates parent directories for the provided path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    """Loads a JSON file into memory."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yields records from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Writes iterable records to a JSONL file."""
    ensure_directory(path)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")


def write_json(path: Path, payload: dict) -> None:
    """Writes a JSON payload to disk."""
    ensure_directory(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)


def softmax(values: Sequence[float], temperature: float) -> List[float]:
    """Computes a temperature-scaled softmax."""
    if not values:
        return []
    if temperature <= 0:
        max_val = max(values)
        exps = [math.exp(v - max_val) for v in values]
    else:
        scaled = [v / temperature for v in values]
        max_val = max(scaled)
        exps = [math.exp(v - max_val) for v in scaled]
    denom = sum(exps)
    if denom == 0:
        return [0.0 for _ in exps]
    return [v / denom for v in exps]


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """L2 normalizes a numpy vector."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def canonicalize_answer(answer: str) -> str:
    """Maps raw answers into normalized canonical forms."""
    answer = answer.strip()
    if not answer:
        return ""

    numeric = _extract_number(answer)
    if numeric is not None:
        return numeric

    lowered = answer.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)

    boolean_map = {
        "yes": "yes",
        "yeah": "yes",
        "y": "yes",
        "true": "yes",
        "no": "no",
        "nope": "no",
        "n": "no",
        "false": "no",
    }
    if lowered in boolean_map:
        return boolean_map[lowered]

    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    return cleaned


def _extract_number(text: str) -> str | None:
    """Extracts a canonical numeric string from text."""
    cleaned = text.replace(",", "").strip()
    # Fractions first (e.g., -3/7)
    fraction_match = re.search(r"-?\d+\s*/\s*-?\d+", cleaned)
    if fraction_match:
        value = fraction_match.group().replace(" ", "")
        try:
            frac = Fraction(value)
            frac = frac.limit_denominator()
            if frac.denominator == 1:
                return str(frac.numerator)
            return f"{frac.numerator}/{frac.denominator}"
        except ZeroDivisionError:
            pass

    number_match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not number_match:
        return None
    number_str = number_match.group()
    try:
        number = Decimal(number_str)
    except InvalidOperation:
        return None

    number = number.normalize()
    if number == number.to_integral():
        return str(number.to_integral())
    formatted = format(number, "f").rstrip("0").rstrip(".")
    return formatted if formatted else "0"
