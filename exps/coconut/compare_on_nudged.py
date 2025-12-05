import json
import sys
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    if len(sys.argv) < 3:
        print("Usage: python exps/coconut/compare_on_nudged.py <baseline.jsonl> <retrieval.jsonl>")
        sys.exit(1)

    base_path = Path(sys.argv[1])
    retr_path = Path(sys.argv[2])

    B = {x["problem_id"]: x for x in load_jsonl(base_path)}
    R = list(load_jsonl(retr_path))

    S = [r for r in R if r.get("retrieval_nudge_applied")]
    n = len(S)
    if n == 0:
        print("No nudged examples found.")
        return

    acc_r = sum(1 for s in S if s.get("is_correct")) / n
    acc_b = sum(1 for s in S if B.get(s["problem_id"], {}).get("is_correct")) / n
    improved = sum(
        1
        for s in S
        if not B.get(s["problem_id"], {}).get("is_correct") and s.get("is_correct")
    )
    regressed = sum(
        1
        for s in S
        if B.get(s["problem_id"], {}).get("is_correct") and not s.get("is_correct")
    )

    # McNemar continuity-corrected
    b, c = improved, regressed
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0

    print(
        "nudged_subset:",
        dict(
            n=n,
            acc_baseline=round(acc_b, 4),
            acc_retrieval=round(acc_r, 4),
            delta=round(acc_r - acc_b, 4),
            improved=b,
            regressed=c,
            mcnemar_chi2=round(chi2, 4),
        ),
    )


if __name__ == "__main__":
    main()

