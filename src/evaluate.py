from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import read_manifest
from metrics import (
    cer,
    expression_exact_match,
    expression_normalized_exact_match,
    math_symbol_f1,
    wer,
)


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
    wer_scores = []
    f1_scores = []
    expr_exact_scores = []
    expr_exact_norm_scores = []
    by_mode_cer: dict[str, list[float]] = {}
    by_mode_wer: dict[str, list[float]] = {}
    by_mode_math_exact: dict[str, list[float]] = {}
    by_mode_math_exact_norm: dict[str, list[float]] = {}
    mode_counts: dict[str, int] = {}
    for idx, sample in truth.items():
        pred = preds.get(idx, "")
        score_cer = cer(pred, sample.text)
        score_wer = wer(pred, sample.text)
        cer_scores.append(score_cer)
        wer_scores.append(score_wer)
        key = (sample.mode or "auto").lower()
        by_mode_cer.setdefault(key, []).append(score_cer)
        by_mode_wer.setdefault(key, []).append(score_wer)
        mode_counts[key] = mode_counts.get(key, 0) + 1
        if sample.mode == "math":
            f1_scores.append(math_symbol_f1(pred, sample.text))
            expr_exact_scores.append(expression_exact_match(pred, sample.text))
            expr_exact_norm_scores.append(expression_normalized_exact_match(pred, sample.text))
            by_mode_math_exact.setdefault(key, []).append(expr_exact_scores[-1])
            by_mode_math_exact_norm.setdefault(key, []).append(expr_exact_norm_scores[-1])

    report = {
        "samples": len(truth),
        "mode_counts": mode_counts,
        "cer_mean": sum(cer_scores) / max(1, len(cer_scores)),
        "wer_mean": sum(wer_scores) / max(1, len(wer_scores)),
        "cer_by_mode": {k: sum(v) / len(v) for k, v in sorted(by_mode_cer.items())},
        "wer_by_mode": {k: sum(v) / len(v) for k, v in sorted(by_mode_wer.items())},
        "math_symbol_f1_mean": (sum(f1_scores) / len(f1_scores)) if f1_scores else None,
        "math_expression_exact_mean": (sum(expr_exact_scores) / len(expr_exact_scores)) if expr_exact_scores else None,
        "math_expression_normalized_exact_mean": (
            (sum(expr_exact_norm_scores) / len(expr_exact_norm_scores)) if expr_exact_norm_scores else None
        ),
        "math_expression_exact_by_mode": {
            k: sum(v) / len(v) for k, v in sorted(by_mode_math_exact.items())
        },
        "math_expression_normalized_exact_by_mode": {
            k: sum(v) / len(v) for k, v in sorted(by_mode_math_exact_norm.items())
        },
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report: {args.report}")


if __name__ == "__main__":
    main()
