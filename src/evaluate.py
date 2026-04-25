from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import read_manifest


def cer(pred: str, truth: str) -> float:
    if not truth:
        return 0.0 if not pred else 1.0
    dp = [[0] * (len(truth) + 1) for _ in range(len(pred) + 1)]
    for i in range(len(pred) + 1):
        dp[i][0] = i
    for j in range(len(truth) + 1):
        dp[0][j] = j
    for i in range(1, len(pred) + 1):
        for j in range(1, len(truth) + 1):
            cost = 0 if pred[i - 1] == truth[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / max(1, len(truth))


def math_symbol_f1(pred: str, truth: str) -> float:
    symbols = set("+-=*/^()[]{}<>≤≥≠≈×÷")
    p = [c for c in pred if c in symbols]
    t = [c for c in truth if c in symbols]
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    overlap = sum(min(p.count(ch), t.count(ch)) for ch in set(p) | set(t))
    precision = overlap / max(1, len(p))
    recall = overlap / max(1, len(t))
    return 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)


def expression_exact_match(pred: str, truth: str) -> float:
    p = pred.replace(" ", "")
    t = truth.replace(" ", "")
    return 1.0 if p == t else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/val.jsonl")
    parser.add_argument("--predictions", default="output/predictions.jsonl")
    parser.add_argument("--report", default="output/eval_report.json")
    args = parser.parse_args()

    truth = {i: s for i, s in enumerate(read_manifest(args.manifest))}
    preds = {}
    pred_path = Path(args.predictions)
    if pred_path.exists():
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                preds[int(row["id"])] = row.get("prediction", "")

    cer_scores = []
    f1_scores = []
    expr_exact_scores = []
    for idx, sample in truth.items():
        pred = preds.get(idx, "")
        cer_scores.append(cer(pred, sample.text))
        if sample.mode == "math":
            f1_scores.append(math_symbol_f1(pred, sample.text))
            expr_exact_scores.append(expression_exact_match(pred, sample.text))

    report = {
        "samples": len(truth),
        "cer_mean": sum(cer_scores) / max(1, len(cer_scores)),
        "math_symbol_f1_mean": (sum(f1_scores) / len(f1_scores)) if f1_scores else None,
        "math_expression_exact_mean": (sum(expr_exact_scores) / len(expr_exact_scores)) if expr_exact_scores else None,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report: {args.report}")


if __name__ == "__main__":
    main()
