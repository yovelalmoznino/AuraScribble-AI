from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import read_manifest
from metrics import cer, expression_exact_match, math_symbol_f1


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
    by_mode: dict[str, list[float]] = {}
    for idx, sample in truth.items():
        pred = preds.get(idx, "")
        score = cer(pred, sample.text)
        cer_scores.append(score)
        key = (sample.mode or "auto").lower()
        by_mode.setdefault(key, []).append(score)
        if sample.mode == "math":
            f1_scores.append(math_symbol_f1(pred, sample.text))
            expr_exact_scores.append(expression_exact_match(pred, sample.text))

    report = {
        "samples": len(truth),
        "cer_mean": sum(cer_scores) / max(1, len(cer_scores)),
        "cer_by_mode": {k: sum(v) / len(v) for k, v in sorted(by_mode.items())},
        "math_symbol_f1_mean": (sum(f1_scores) / len(f1_scores)) if f1_scores else None,
        "math_expression_exact_mean": (sum(expr_exact_scores) / len(expr_exact_scores)) if expr_exact_scores else None,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report: {args.report}")


if __name__ == "__main__":
    main()
