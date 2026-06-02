from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic

from model_factory import build_model

torch._dynamo.config.suppress_errors = True


def _disable_attention_fastpaths_for_export() -> None:
    """Disable fused MHA fastpaths that break ONNX export on some torch builds."""
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        print("      Disabled torch MHA fastpath for ONNX export.")
    except Exception:
        pass


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _resolve_vocab(checkpoint: dict[str, Any]) -> Any:
    vocab = checkpoint.get("vocab")
    if vocab is None:
        raise RuntimeError("Checkpoint is missing 'vocab'. Train and save checkpoint again.")
    return vocab


def _resolve_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state", "model_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    raise RuntimeError(
        "Checkpoint is missing model weights. Expected one of: "
        "'model_state', 'model_state_dict', or 'state_dict'."
    )


def _write_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _quantize_onnx(onnx_path: Path, quantized_path: Path) -> str:
    try:
        quantize_dynamic(
            model_input=onnx_path.as_posix(),
            model_output=quantized_path.as_posix(),
            weight_type=QuantType.QInt8,
        )
        return "default"
    except RuntimeError as exc:
        message = str(exc)
        if "Unable to find data type for weight_name" not in message:
            raise
        print("      Quantization type inference failed; retrying with DefaultTensorType=FLOAT.")
        quantize_dynamic(
            model_input=onnx_path.as_posix(),
            model_output=quantized_path.as_posix(),
            weight_type=QuantType.QInt8,
            extra_options={"DefaultTensorType": 1},
        )
        return "default_tensor_type_float"


def _validate_onnx_runtime(onnx_path: Path, *, short_time: int, tgt_tokens: int, bos_id: int, pad_id: int) -> None:
    """Fail export early if attention reshape is still baked to trace-time only."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    src = np.random.randn(1, short_time, 3).astype(np.float32)
    src_lens = np.array([short_time], dtype=np.int64)
    tgt = np.full((1, tgt_tokens), pad_id, dtype=np.int64)
    tgt[0, 0] = bos_id
    session.run(
        None,
        {
            "src": src,
            "src_lens": src_lens,
            "tgt_inp": tgt,
        },
    )
    print(f"      ONNX runtime smoke test OK (src_time={short_time}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AuraScribble Seq2Seq model to ONNX + INT8 ONNX.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", default="output/checkpoint_best.pt", help="Path to model checkpoint.")
    parser.add_argument(
        "--trace-time",
        type=int,
        default=128,
        help="Dummy source time length used for export example (also written to summary).",
    )
    parser.add_argument(
        "--trace-tokens",
        type=int,
        default=128,
        help="Dummy target token length used for export example.",
    )
    parser.add_argument(
        "--summary",
        default="output/export_summary.json",
        help="Path to JSON export summary.",
    )
    parser.add_argument(
        "--smoke-time",
        type=int,
        default=38,
        help="Short src_time used for post-export ONNX Runtime validation.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    summary_path = Path(args.summary)
    trace_time = int(args.trace_time)
    trace_tokens = int(args.trace_tokens)
    smoke_time = int(args.smoke_time)

    _disable_attention_fastpaths_for_export()

    try:
        if trace_time < 4:
            raise ValueError(f"--trace-time must be >= 4, got {trace_time}")
        if trace_tokens < 2:
            raise ValueError(f"--trace-tokens must be >= 2, got {trace_tokens}")

        print(f"[1/8] Loading config: {config_path}")
        cfg = _load_config(config_path)

        export_cfg = cfg.get("export") or {}
        onnx_out = Path(export_cfg.get("onnx_file", "output/model.onnx"))
        quant_out = Path(export_cfg.get("quantized_onnx_file", "output/model.int8.onnx"))
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        quant_out.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        print("[2/8] Output directories are ready.")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"[3/8] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        vocab = _resolve_vocab(checkpoint)
        model_state = _resolve_model_state(checkpoint)

        vocab_size = len(vocab)
        print(f"      Vocab size from checkpoint: {vocab_size}")

        vocab_out = onnx_out.parent / "vocab.from_checkpoint.txt"
        vocab_out.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        print(f"      Wrote vocab for OTA upload: {vocab_out}")

        model = build_model(cfg, vocab_size)
        model.load_state_dict(model_state, strict=True)
        model.eval()
        print("[4/8] Model initialized and checkpoint weights loaded.")

        bos_id = 2
        eos_id = 3
        if isinstance(vocab, list):
            if "<bos>" in vocab:
                bos_id = vocab.index("<bos>")
            if "<eos>" in vocab:
                eos_id = vocab.index("<eos>")
        pad_fill = eos_id if eos_id >= 0 else 0
        dummy_src = torch.randn(1, trace_time, int(cfg["model"]["input_dim"]), dtype=torch.float32)
        dummy_src_lens = torch.tensor([trace_time], dtype=torch.long)
        dummy_tgt = torch.full((1, trace_tokens), fill_value=pad_fill, dtype=torch.long)
        dummy_tgt[0, 0] = bos_id

        dynamic_axes = {
            "src": {0: "batch", 1: "time"},
            "src_lens": {0: "batch"},
            "tgt_inp": {0: "batch", 1: "tokens"},
            "logits": {0: "batch", 1: "tokens"},
        }
        export_kwargs: dict[str, Any] = dict(
            input_names=["src", "src_lens", "tgt_inp"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=True,
            export_params=True,
        )

        print("[5/8] Exporting model directly to ONNX (no torch.jit.trace).")
        print(f"      Example shapes: src_time={trace_time}, tgt_tokens={trace_tokens}")
        with torch.enable_grad():
            print(f"[6/8] Writing ONNX (opset 18): {onnx_out}")
            try:
                torch.onnx.export(
                    model,
                    (dummy_src, dummy_src_lens, dummy_tgt),
                    onnx_out.as_posix(),
                    dynamo=False,
                    **export_kwargs,
                )
            except TypeError:
                torch.onnx.export(
                    model,
                    (dummy_src, dummy_src_lens, dummy_tgt),
                    onnx_out.as_posix(),
                    **export_kwargs,
                )

        print(f"[7/8] Validating ONNX at short src_time={smoke_time} ...")
        _validate_onnx_runtime(
            onnx_out,
            short_time=smoke_time,
            tgt_tokens=trace_tokens,
            bos_id=bos_id,
            pad_id=pad_fill,
        )

        print(f"[8/8] Quantizing ONNX to INT8: {quant_out}")
        quantization_strategy = _quantize_onnx(onnx_out, quant_out)

        summary = {
            "status": "success",
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "onnx_model": str(onnx_out),
            "quantized_onnx_model": str(quant_out),
            "vocab_for_ota": str(vocab_out),
            "summary_path": str(summary_path),
            "opset_version": 18,
            "quantization": "dynamic_qint8",
            "quantization_strategy": quantization_strategy,
            "vocab_size": vocab_size,
            "trace_shapes": {
                "src_time": trace_time,
                "tgt_tokens": trace_tokens,
                "smoke_src_time": smoke_time,
            },
            "dynamic_axes": dynamic_axes,
            "export_path": "direct_onnx_export_no_jit_trace",
        }
        _write_summary(summary_path, summary)

        print("Export complete.")
        print(f"  - ONNX: {onnx_out}")
        print(f"  - INT8 ONNX: {quant_out}")
        print(f"  - Vocab (upload this): {vocab_out}")
        print(f"  - Summary: {summary_path}")

    except Exception as exc:
        error_summary = {
            "status": "failed",
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "summary_path": str(summary_path),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        _write_summary(summary_path, error_summary)
        print(f"Export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Failure summary written to: {summary_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
