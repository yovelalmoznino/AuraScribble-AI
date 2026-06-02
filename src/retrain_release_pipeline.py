from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from decode_quality import passes_export_gate


def run(cmd: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stable retrain pipeline: train -> predict -> evaluate -> gate -> export -> upload."
    )
    parser.add_argument("--work-dir", default=".")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default="output/checkpoint_best.pt")
    parser.add_argument("--manifest", default="data/processed/val.jsonl")
    parser.add_argument("--predictions", default="output/predictions.release.jsonl")
    parser.add_argument("--report", default="output/eval_report.release.json")
    parser.add_argument("--summary", default="output/retrain_release_summary.json")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs for train.py (0 keeps config).")
    parser.add_argument("--upload", action="store_true", help="Upload to Firebase if gate passes.")
    parser.add_argument("--credentials", default=None, help="Optional Firebase service account json path.")
    parser.add_argument("--max-cer-mean", type=float, default=0.95)
    parser.add_argument("--max-wer-mean", type=float, default=0.98)
    parser.add_argument("--max-mode-cer", type=float, default=1.05)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    config = str((work_dir / args.config).resolve())
    checkpoint = str((work_dir / args.checkpoint).resolve())
    manifest = str((work_dir / args.manifest).resolve())
    predictions = str((work_dir / args.predictions).resolve())
    report = str((work_dir / args.report).resolve())
    summary_path = (work_dir / args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    train_cmd = [sys.executable, "src/train.py", "--config", config]
    if args.epochs > 0:
        train_cmd += ["--epochs", str(args.epochs)]
    run(train_cmd, cwd=work_dir)

    run(
        [
            sys.executable,
            "src/predict.py",
            "--config",
            config,
            "--checkpoint",
            checkpoint,
            "--manifest",
            manifest,
            "--output",
            predictions,
        ],
        cwd=work_dir,
    )
    run(
        [
            sys.executable,
            "src/evaluate.py",
            "--manifest",
            manifest,
            "--predictions",
            predictions,
            "--report",
            report,
        ],
        cwd=work_dir,
    )

    rep = read_json(Path(report))
    cer_mean = float(rep.get("cer_mean", 99.0))
    wer_mean = float(rep.get("wer_mean", 99.0))
    by_mode = rep.get("cer_by_mode") or {}
    mode_cer = {k: float(v) for k, v in by_mode.items() if isinstance(v, (int, float))}
    samples = int(rep.get("samples", 0))
    gate_ok, gate_reason = passes_export_gate(
        cer_mean=cer_mean,
        val_cer=cer_mean,
        collapse_count=0,
        sample_count=max(1, samples),
        max_cer_mean=args.max_cer_mean,
        max_val_cer=args.max_cer_mean,
        per_mode_val_cer=mode_cer,
        max_per_mode_val_cer=args.max_mode_cer,
    )
    if wer_mean > args.max_wer_mean:
        gate_ok = False
        gate_reason = f"wer_mean {wer_mean:.3f} > {args.max_wer_mean:.3f}"

    exported = False
    uploaded = False
    if gate_ok:
        run(
            [
                sys.executable,
                "src/export_onnx.py",
                "--config",
                config,
                "--checkpoint",
                checkpoint,
                "--trace-time",
                "128",
                "--trace-tokens",
                "128",
                "--smoke-time",
                "38",
            ],
            cwd=work_dir,
        )
        exported = True

        if args.upload:
            upload_cmd = [sys.executable, "src/upload_firebase.py", "--local", "output/model.onnx"]
            if args.credentials:
                upload_cmd += ["--credentials", args.credentials]
            run(upload_cmd, cwd=work_dir)
            uploaded = True

    summary = {
        "status": "ok" if gate_ok else "blocked_by_gate",
        "gate_ok": gate_ok,
        "gate_reason": gate_reason,
        "metrics": {
            "samples": samples,
            "cer_mean": cer_mean,
            "wer_mean": wer_mean,
            "cer_by_mode": mode_cer,
            "math_expression_normalized_exact_mean": rep.get("math_expression_normalized_exact_mean"),
            "math_symbol_f1_mean": rep.get("math_symbol_f1_mean"),
        },
        "artifacts": {
            "checkpoint": checkpoint,
            "predictions": predictions,
            "report": report,
            "exported": exported,
            "uploaded": uploaded,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote pipeline summary: {summary_path}")
    if not gate_ok:
        print(f"Gate blocked release: {gate_reason}")


if __name__ == "__main__":
    main()
