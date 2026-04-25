from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic

from model import HandwritingSeq2SeqModel

torch._dynamo.config.suppress_errors = True


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
    """
    Quantize ONNX with a robust fallback for missing tensor type inference.
    Returns a string describing which strategy succeeded.
    """
    try:
        quantize_dynamic(
            model_input=onnx_path.as_posix(),
            model_output=quantized_path.as_posix(),
            weight_type=QuantType.QInt8,
        )
        return "default"
    except RuntimeError as exc:
        message = str(exc)
        retryable = "Unable to find data type for weight_name" in message
        if not retryable:
            raise

        print("      Quantization type inference failed; retrying with DefaultTensorType=FLOAT.")
        # ONNX TensorProto.FLOAT enum value is 1. We use the numeric constant
        # to avoid introducing an additional dependency import requirement here.
        quantize_dynamic(
            model_input=onnx_path.as_posix(),
            model_output=quantized_path.as_posix(),
            weight_type=QuantType.QInt8,
            extra_options={"DefaultTensorType": 1},
        )
        return "default_tensor_type_float"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AuraScribble Seq2Seq model to ONNX + INT8 ONNX.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", default="output/checkpoint_best.pt", help="Path to model checkpoint.")
    parser.add_argument(
        "--trace-time",
        type=int,
        default=128,
        help="Dummy source time length used for tracing/export shape.",
    )
    parser.add_argument(
        "--trace-tokens",
        type=int,
        default=96,
        help="Dummy target token length used for tracing/export decoder window.",
    )
    parser.add_argument(
        "--summary",
        default="output/export_summary.json",
        help="Path to JSON export summary.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    summary_path = Path(args.summary)
    trace_time = int(args.trace_time)
    trace_tokens = int(args.trace_tokens)

    try:
        if trace_time < 4:
            raise ValueError(f"--trace-time must be >= 4, got {trace_time}")
        if trace_tokens < 2:
            raise ValueError(f"--trace-tokens must be >= 2, got {trace_tokens}")

        print(f"[1/7] Loading config: {config_path}")
        cfg = _load_config(config_path)

        onnx_out = Path(cfg["export"]["onnx_file"])
        quant_out = Path(cfg["export"]["quantized_onnx_file"])
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        quant_out.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[2/7] Output directories are ready.")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"[3/7] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        vocab = _resolve_vocab(checkpoint)
        model_state = _resolve_model_state(checkpoint)

        vocab_size = len(vocab)
        print(f"      Vocab size from checkpoint: {vocab_size}")

        model = HandwritingSeq2SeqModel(
            input_dim=int(cfg["model"]["input_dim"]),
            hidden=int(cfg["model"]["encoder_hidden"]),
            layers=int(cfg["model"]["encoder_layers"]),
            dropout=float(cfg["model"]["dropout"]),
            vocab_size=vocab_size,
        )
        model.load_state_dict(model_state, strict=True)
        model.eval()
        print("[4/7] Model initialized and checkpoint weights loaded.")

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

        print("[5/7] Tracing model with torch.jit.trace (legacy/JIT path).")
        print(f"      Trace shapes: src_time={trace_time}, tgt_tokens={trace_tokens}")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model,
                (dummy_src, dummy_src_lens, dummy_tgt),
                strict=False,
            )

            export_kwargs: dict[str, Any] = dict(
                input_names=["src", "src_lens", "tgt_inp"],
                output_names=["logits"],
                dynamic_axes={
                    "src": {1: "time"},
                    "tgt_inp": {1: "tokens"},
                    "logits": {1: "tokens"},
                },
                opset_version=18,
                do_constant_folding=True,
                export_params=True,
            )

            print(f"[6/7] Exporting ONNX (opset 18): {onnx_out}")
            try:
                torch.onnx.export(
                    traced_model,
                    (dummy_src, dummy_src_lens, dummy_tgt),
                    onnx_out.as_posix(),
                    dynamo=False,
                    **export_kwargs,
                )
            except TypeError:
                # Older torch builds may not accept 'dynamo' kwarg.
                torch.onnx.export(
                    traced_model,
                    (dummy_src, dummy_src_lens, dummy_tgt),
                    onnx_out.as_posix(),
                    **export_kwargs,
                )

        print(f"[7/7] Quantizing ONNX to INT8: {quant_out}")
        quantization_strategy = _quantize_onnx(onnx_out, quant_out)

        summary = {
            "status": "success",
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "onnx_model": str(onnx_out),
            "quantized_onnx_model": str(quant_out),
            "summary_path": str(summary_path),
            "opset_version": 18,
            "quantization": "dynamic_qint8",
            "quantization_strategy": quantization_strategy,
            "vocab_size": vocab_size,
            "trace_shapes": {
                "src_time": trace_time,
                "tgt_tokens": trace_tokens,
            },
            "dynamic_axes": {
                "src": {"1": "time"},
                "tgt_inp": {"1": "tokens"},
                "logits": {"1": "tokens"},
            },
        }
        _write_summary(summary_path, summary)

        print("Export complete.")
        print(f"  - ONNX: {onnx_out}")
        print(f"  - INT8 ONNX: {quant_out}")
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
