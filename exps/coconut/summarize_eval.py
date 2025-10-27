import json
import sys
from statistics import mean
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python exps/coconut/summarize_eval.py <baseline.jsonl> <retrieval.jsonl>",
            file=sys.stderr,
        )
        sys.exit(2)

    baseline_path, retrieval_path = sys.argv[1], sys.argv[2]
    b = load_jsonl(baseline_path)
    r = load_jsonl(retrieval_path)

    nb, nr = len(b), len(r)
    acc_b = mean(int(x.get("is_correct", False)) for x in b) if b else 0.0
    acc_r = mean(int(x.get("is_correct", False)) for x in r) if r else 0.0
    tok_b = mean(int(x.get("num_generated_tokens", 0)) for x in b) if b else 0.0
    tok_r = mean(int(x.get("num_generated_tokens", 0)) for x in r) if r else 0.0

    by_id_b: Dict[str, dict] = {str(x.get("problem_id")): x for x in b}
    by_id_r: Dict[str, dict] = {str(x.get("problem_id")): x for x in r}

    common_ids = set(by_id_b) & set(by_id_r)
    improved = sum(
        1
        for k in common_ids
        if not by_id_b[k].get("is_correct") and by_id_r[k].get("is_correct")
    )
    regressed = sum(
        1
        for k in common_ids
        if by_id_b[k].get("is_correct") and not by_id_r[k].get("is_correct")
    )

    nudged = [x for x in r if x.get("retrieval_nudge_applied")]
    nudge_rate = (len(nudged) / nr) if nr else 0.0
    acc_when_nudged = (
        mean(int(x.get("is_correct", 0)) for x in nudged) if nudged else 0.0
    )
    mean_top_sim = (
        mean(float(x.get("retrieval_top_similarity") or 0.0) for x in r) if r else 0.0
    )

    print(f"baseline: n={nb}, acc={acc_b:.4f}, avg_tokens={tok_b:.1f}")
    print(
        f"retrieval: n={nr}, acc={acc_r:.4f}, avg_tokens={tok_r:.1f}, improved={improved}, regressed={regressed}"
    )
    print(
        f"   nudge_rate={nudge_rate:.3f}, acc_when_nudged={acc_when_nudged:.4f}, mean_top_sim={mean_top_sim:.3f}"
    )


if __name__ == "__main__":
    main()

