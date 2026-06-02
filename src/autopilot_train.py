from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from decode_quality import passes_export_gate


@dataclass
class PhaseResult:
    phase_name: str
    best_val_cer: float | None
    best_val_core: float | None
    last_epoch: int
    status: str
    note: str = ""


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def _run_cmd_capture(cmd: list[str], cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return completed.returncode, completed.stdout


def _parse_training_entries(log_path: Path, start_line: int = 0) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    for ln in lines[start_line:]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "epoch" in obj:
            entries.append(obj)
    return entries


def _val_core(entry: dict[str, Any]) -> float | None:
    m = entry.get("val_mode_cer")
    if not isinstance(m, dict):
        return None
    vals: list[float] = []
    for k in ("english", "hebrew", "math"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if len(vals) < 2:
        return None
    return sum(vals) / len(vals)


def _best_metrics(entries: list[dict[str, Any]]) -> tuple[float | None, float | None, int]:
    best_val: float | None = None
    best_core: float | None = None
    last_epoch = 0
    for e in entries:
        last_epoch = max(last_epoch, int(e.get("epoch", 0)))
        v = e.get("val_cer")
        if isinstance(v, (int, float)):
            if best_val is None or float(v) < best_val:
                best_val = float(v)
        c = _val_core(e)
        if c is not None and (best_core is None or c < best_core):
            best_core = c
    return best_val, best_core, last_epoch


def _set_phase_cfg(cfg: dict[str, Any], phase_key: str, output_dir: Path) -> dict[str, Any]:
    p = cfg.get(phase_key) or {}
    cfg["train_manifest"] = str(output_dir / {
        "curriculum_phase1": "train_short.jsonl",
        "curriculum_phase2a": "train_medium.jsonl",
        "curriculum_phase2b": "train.jsonl",
        "curriculum_phase2c": "train_iam_long.jsonl",
    }[phase_key])
    cfg["max_tgt_len"] = int(p.get("max_tgt_len", cfg.get("max_tgt_len", 128)))
    cfg["epochs"] = int(p.get("epochs", cfg.get("epochs", 10)))
    cfg["learning_rate"] = float(p.get("learning_rate", cfg.get("learning_rate", 2e-4)))
    cfg["early_stop_patience"] = int(p.get("early_stop_patience", cfg.get("early_stop_patience", 12)))
    return cfg


def _phase_train(
    work_dir: Path,
    cfg_path: Path,
    phase_key: str,
    resume: bool,
) -> PhaseResult:
    cfg = _read_yaml(cfg_path)
    output_dir = Path(cfg.get("output_dir", "output"))
    if not output_dir.is_absolute():
        output_dir = (work_dir / output_dir).resolve()
    cfg = _set_phase_cfg(cfg, phase_key, output_dir)
    cfg["resume_from_checkpoint"] = bool(resume)
    cfg["model_path"] = str(output_dir / "checkpoint_best.pt")
    _write_yaml(cfg_path, cfg)

    log_path = output_dir / "training_log.jsonl"
    start_line = 0
    if log_path.exists():
        start_line = len(log_path.read_text(encoding="utf-8").splitlines())

    _run_cmd(
        [sys.executable, "src/train.py", "--config", str(cfg_path)],
        cwd=work_dir,
    )
    entries = _parse_training_entries(log_path, start_line=start_line)
    best_val, best_core, last_epoch = _best_metrics(entries)
    return PhaseResult(
        phase_name=phase_key,
        best_val_cer=best_val,
        best_val_core=best_core,
        last_epoch=last_epoch,
        status="ok",
    )


def _run_data_prep(work_dir: Path) -> None:
    _run_cmd([sys.executable, "src/prepare_raw.py", "--raw", "data/raw", "--output", "output/all.jsonl"], cwd=work_dir)
    _run_cmd(
        [
            sys.executable,
            "src/split_manifest.py",
            "--source",
            "output/all.jsonl",
            "--train-out",
            "output/train.jsonl",
            "--val-out",
            "output/val.jsonl",
            "--val-ratio",
            "0.1",
            "--seed",
            "1337",
            "--balance-modes",
            "--per-mode-target",
            "6000",
            "--max-oversample",
            "8.0",
        ],
        cwd=work_dir,
    )
    _run_cmd(
        [
            sys.executable,
            "src/build_curriculum_manifest.py",
            "--train",
            "output/train.jsonl",
            "--short-out",
            "output/train_short.jsonl",
            "--medium-out",
            "output/train_medium.jsonl",
            "--iam-out",
            "output/train_iam_long.jsonl",
            "--max-chars-short",
            "32",
            "--medium-min-chars",
            "33",
            "--medium-max-chars",
            "72",
            "--iam-min-chars",
            "24",
            "--rewrite-train",
        ],
        cwd=work_dir,
    )


def _write_report(path: Path, results: list[PhaseResult], note: str = "") -> None:
    obj = {
        "status": "ok",
        "note": note,
        "phases": [
            {
                "phase": r.phase_name,
                "status": r.status,
                "best_val_cer": r.best_val_cer,
                "best_val_core": r.best_val_core,
                "last_epoch": r.last_epoch,
                "note": r.note,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote autopilot report: {path}")


def _tune_decode_weights_from_report(cfg: dict[str, Any], eval_report_path: Path) -> dict[str, Any]:
    if not eval_report_path.exists():
        return cfg
    rep = json.loads(eval_report_path.read_text(encoding="utf-8"))
    by_mode = rep.get("cer_by_mode") or {}
    math_cer = float(by_mode.get("math", 1.0))
    he_cer = float(by_mode.get("hebrew", 1.0))
    en_cer = float(by_mode.get("english", 1.0))
    # Heuristic: if math is weak, increase CTC decode contribution for monotonic alignment;
    # otherwise keep AR-biased decode for language quality.
    if math_cer > 1.0:
        cfg["decode_ctc_weight"] = 0.65
        cfg["decode_ar_weight"] = 0.35
    elif he_cer > en_cer + 0.08:
        cfg["decode_ctc_weight"] = 0.55
        cfg["decode_ar_weight"] = 0.45
    else:
        cfg["decode_ctc_weight"] = 0.35
        cfg["decode_ar_weight"] = 0.65
    cfg["decode_strategy"] = "joint"
    return cfg


def _evaluate_checkpoint(work_dir: Path, cfg_path: Path, checkpoint_path: Path) -> tuple[dict[str, Any], Path]:
    cfg = _read_yaml(cfg_path)
    val_manifest = cfg.get("val_manifest", "output/val.jsonl")
    pred_out = Path(cfg.get("output_dir", "output")) / "predictions_autopilot.jsonl"
    rep_out = Path(cfg.get("output_dir", "output")) / "eval_report_autopilot.json"
    _run_cmd(
        [
            sys.executable,
            "src/predict.py",
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(checkpoint_path),
            "--manifest",
            str(val_manifest),
            "--output",
            str(pred_out),
        ],
        cwd=work_dir,
    )
    _run_cmd(
        [
            sys.executable,
            "src/evaluate.py",
            "--manifest",
            str(val_manifest),
            "--predictions",
            str(pred_out),
            "--report",
            str(rep_out),
        ],
        cwd=work_dir,
    )
    rep = json.loads(rep_out.read_text(encoding="utf-8"))
    return rep, rep_out


def _maybe_export_and_upload(
    work_dir: Path,
    cfg_path: Path,
    checkpoint_path: Path,
    eval_report: dict[str, Any],
    *,
    do_export: bool,
    do_upload: bool,
    credentials: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"export_attempted": False, "uploaded": False}
    if not do_export:
        return result

    cer_mean = float(eval_report.get("cer_mean", 9.9))
    by_mode = eval_report.get("cer_by_mode") or {}
    val_core = None
    core_vals = [float(by_mode[k]) for k in ("english", "hebrew", "math") if k in by_mode]
    if core_vals:
        val_core = sum(core_vals) / len(core_vals)

    ok, reason = passes_export_gate(
        cer_mean=cer_mean,
        val_cer=val_core if val_core is not None else cer_mean,
        collapse_count=0,
        sample_count=max(1, int(eval_report.get("samples", 1))),
        max_cer_mean=0.90,
        max_val_cer=0.95,
        per_mode_val_cer={k: float(v) for k, v in by_mode.items() if isinstance(v, (int, float))},
        max_per_mode_val_cer=1.05,
    )
    result["gate_ok"] = ok
    result["gate_reason"] = reason
    if not ok:
        return result

    result["export_attempted"] = True
    _run_cmd(
        [sys.executable, "src/export_onnx.py", "--config", str(cfg_path), "--checkpoint", str(checkpoint_path)],
        cwd=work_dir,
    )
    export_summary = Path(work_dir / "output/export_summary.json")
    result["export_summary"] = str(export_summary)

    if do_upload:
        onnx_path = Path(work_dir / "output/model.onnx")
        vocab_path = Path(work_dir / "output/vocab.from_checkpoint.txt")
        cmd: list[str] = [
            sys.executable,
            "src/upload_firebase.py",
            "--local",
            str(onnx_path),
        ]
        if vocab_path.exists():
            cmd += ["--vocab", str(vocab_path)]
        if credentials:
            cmd += ["--credentials", credentials]
        _run_cmd(cmd, cwd=work_dir)
        result["uploaded"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic curriculum trainer for handwriting model.")
    parser.add_argument("--config", default="configs/train_kaggle.yaml")
    parser.add_argument("--work-dir", default=".")
    parser.add_argument("--run-data-prep", action="store_true", help="Run prepare/split/curriculum build before training.")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2a", action="store_true")
    parser.add_argument("--skip-phase2b", action="store_true")
    parser.add_argument("--skip-phase2c", action="store_true")
    parser.add_argument("--report", default="output/autopilot_report.json")
    parser.add_argument("--export", action="store_true", help="Export ONNX at the end if gate passes.")
    parser.add_argument("--upload", action="store_true", help="Upload ONNX/vocab to Firebase if gate passes.")
    parser.add_argument("--firebase-credentials", default=None, help="Path to Firebase service account JSON.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    cfg_path = (work_dir / args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    if args.run_data_prep:
        print("[autopilot] Running data preparation...")
        _run_data_prep(work_dir)

    results: list[PhaseResult] = []

    # Phase 1
    if not args.skip_phase1:
        print("[autopilot] Phase 1 (short)")
        r1 = _phase_train(work_dir, cfg_path, "curriculum_phase1", resume=False)
        results.append(r1)

    # Phase 2a (with one automatic fallback tweak if core metric is still very poor)
    if not args.skip_phase2a:
        print("[autopilot] Phase 2a (medium)")
        r2 = _phase_train(work_dir, cfg_path, "curriculum_phase2a", resume=True)
        results.append(r2)
        if r2.best_val_core is None or r2.best_val_core > 0.95:
            print("[autopilot] Phase 2a core metric is weak; applying automatic fallback and retry.")
            cfg = _read_yaml(cfg_path)
            cfg["ctc_loss_weight"] = float(cfg.get("ctc_loss_weight", 0.3))
            cfg["ar_loss_weight"] = float(cfg.get("ar_loss_weight", 0.7))
            cfg["decode_strategy"] = "joint"
            cfg["decode_ctc_weight"] = 0.25
            cfg["decode_ar_weight"] = 0.75
            p2a = cfg.get("curriculum_phase2a") or {}
            p2a["learning_rate"] = min(float(p2a.get("learning_rate", 1.2e-4)), 6e-5)
            p2a["early_stop_patience"] = max(int(p2a.get("early_stop_patience", 20)), 20)
            cfg["curriculum_phase2a"] = p2a
            _write_yaml(cfg_path, cfg)
            r2b = _phase_train(work_dir, cfg_path, "curriculum_phase2a", resume=True)
            r2b.phase_name = "curriculum_phase2a_retry"
            results.append(r2b)
            # After retry, tune decode weights from latest evaluation signal.
            cfg = _read_yaml(cfg_path)
            output_dir = Path(cfg.get("output_dir", "output"))
            if not output_dir.is_absolute():
                output_dir = (work_dir / output_dir).resolve()
            # Soft signal from current training log eval output file.
            eval_report_path = output_dir / "eval_report_autopilot.json"
            if eval_report_path.exists():
                cfg = _tune_decode_weights_from_report(cfg, eval_report_path)
                _write_yaml(cfg_path, cfg)

    if not args.skip_phase2b:
        print("[autopilot] Phase 2b (full)")
        r3 = _phase_train(work_dir, cfg_path, "curriculum_phase2b", resume=True)
        results.append(r3)

    if not args.skip_phase2c:
        print("[autopilot] Phase 2c (iam-long)")
        r4 = _phase_train(work_dir, cfg_path, "curriculum_phase2c", resume=True)
        results.append(r4)

    report_path = (work_dir / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _read_yaml(cfg_path)
    output_dir = Path(cfg.get("output_dir", "output"))
    if not output_dir.is_absolute():
        output_dir = (work_dir / output_dir).resolve()
    best_ckpt = output_dir / "checkpoint_best.pt"

    eval_report: dict[str, Any] | None = None
    eval_report_path: Path | None = None
    export_upload_result: dict[str, Any] = {}
    if best_ckpt.exists():
        eval_report, eval_report_path = _evaluate_checkpoint(work_dir, cfg_path, best_ckpt)
        # Auto-tune decode weights from evaluation and persist.
        cfg = _tune_decode_weights_from_report(_read_yaml(cfg_path), eval_report_path)
        _write_yaml(cfg_path, cfg)
        export_upload_result = _maybe_export_and_upload(
            work_dir,
            cfg_path,
            best_ckpt,
            eval_report,
            do_export=args.export,
            do_upload=args.upload,
            credentials=args.firebase_credentials,
        )

    _write_report(report_path, results, note="autopilot completed")
    if eval_report is not None:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["final_eval_report"] = str(eval_report_path)
        payload["final_eval"] = eval_report
        payload["export_upload"] = export_upload_result
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[autopilot] Done.")


if __name__ == "__main__":
    main()
