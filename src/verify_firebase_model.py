"""
Download models/latest_handwriting.onnx (+ vocab) from Firebase and sanity-check on val.jsonl.

Usage (from tools/handwriting-model with venv active):
  python src/verify_firebase_model.py
  python src/verify_firebase_model.py --manifest data/processed/val.jsonl --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from dataset import points_to_relative_features, read_manifest
from decode import greedy_decode
from metrics import cer
from model_factory import build_model
from tokenizer import CharTokenizer
from upload_firebase import DEFAULT_BUCKET, _resolve_credentials

try:
    from google.cloud import storage
except ImportError as exc:
    raise SystemExit("pip install google-cloud-storage onnxruntime") from exc


def download_blob(bucket_name: str, remote_path: str, local_path: Path) -> None:
    creds = _resolve_credentials(None)
    if creds is None:
        raise RuntimeError(
            "Firebase credentials missing.\n"
            "Option A — service account JSON:\n"
            "  1. Firebase Console → Project aurascribblr → ⚙ Settings → Service accounts\n"
            "  2. Generate new private key → save as:\n"
            "     tools/handwriting-model/configs/firebase_service_account.json\n"
            "Option B — env var for this PowerShell session:\n"
            "  $env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\\path\\to\\your-key.json'\n"
            "Option C — manual download (no key):\n"
            "  Download models/latest_handwriting.onnx and latest_vocab.txt from Storage,\n"
            "  then run: python src/verify_firebase_model.py --local-onnx PATH --local-vocab PATH"
        )
    client = storage.Client(credentials=creds, project=creds.project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_path)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket_name}/{remote_path} not found")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    blob.reload()
    print(f"Downloaded {remote_path} -> {local_path} ({local_path.stat().st_size} bytes)")


def onnx_greedy_decode(session: ort.InferenceSession, tokenizer: CharTokenizer, points: list) -> str:
    feats = points_to_relative_features(points)
    if len(feats) == 0:
        return ""
    bos = tokenizer.bos_id
    eos = tokenizer.eos_id
    pad = tokenizer.pad_id
    blank = tokenizer.blank_id
    token_ids = [bos]
    max_steps = 48
    max_tgt = 96
    src = feats.astype(np.float32)[None, :, :]  # [1, T, 3]
    src_lens = np.array([src.shape[1]], dtype=np.int64)  # [1] — matches Android / export
    for _ in range(max_steps):
        tgt = token_ids + [pad] * max(0, max_tgt - len(token_ids))
        tgt = tgt[:max_tgt]
        tgt_inp = np.array([tgt], dtype=np.int64)  # [1, U]
        logits = session.run(
            None,
            {
                "src": src,
                "src_lens": src_lens,
                "tgt_inp": tgt_inp,
            },
        )[0]
        step_idx = len(token_ids) - 1
        if step_idx >= logits.shape[1]:
            break
        row = logits[0, step_idx]
        next_id = int(row.argmax())
        if next_id in (eos, pad):
            break
        token_ids.append(next_id)
    out = []
    prev = None
    for idx in token_ids[1:]:
        if idx in (blank, pad, bos, eos):
            prev = None
            continue
        if idx == prev:
            continue
        prev = idx
        if 0 <= idx < len(tokenizer.vocab):
            tok = tokenizer.vocab[idx]
            if tok not in ("<blank>", "<pad>", "<bos>", "<eos>"):
                out.append(tok)
    text = "".join(out)
    if tokenizer._contains_hebrew(text):
        return text[::-1]
    return text


def load_pytorch_baseline(config: dict, checkpoint: Path, device: torch.device):
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    vocab = checkpoint_data.get("vocab")
    tokenizer = CharTokenizer(config["vocab_path"])
    if vocab:
        tokenizer.vocab = [v.replace("\n", "") for v in vocab if v.replace("\n", "") != "" or v == " "]
        tokenizer.stoi = {t: i for i, t in enumerate(tokenizer.vocab)}
    model = build_model(config, len(tokenizer)).to(device)
    state = (
        checkpoint_data.get("model_state")
        or checkpoint_data.get("model_state_dict")
        or checkpoint_data.get("state_dict")
    )
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Firebase handwriting ONNX on validation samples.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--manifest", default="data/processed/val.jsonl")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--remote-onnx", default="models/latest_handwriting.onnx")
    parser.add_argument("--remote-vocab", default="models/latest_vocab.txt")
    parser.add_argument("--out-dir", default="output/firebase_verify")
    parser.add_argument("--checkpoint", default="output/checkpoint_best.pt")
    parser.add_argument(
        "--local-onnx",
        default=None,
        help="Skip download; use this ONNX file (e.g. downloaded from Firebase Console).",
    )
    parser.add_argument(
        "--local-vocab",
        default=None,
        help="Skip vocab download; use this vocab.txt path.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(args.local_onnx) if args.local_onnx else out_dir / "latest_handwriting.onnx"
    vocab_path = Path(args.local_vocab) if args.local_vocab else out_dir / "latest_vocab.txt"

    if args.local_onnx:
        if not onnx_path.exists():
            print(f"ONNX not found: {onnx_path}")
            sys.exit(1)
        print(f"Using local ONNX: {onnx_path}")
    else:
        download_blob(args.bucket, args.remote_onnx, onnx_path)

    if args.local_vocab:
        if not vocab_path.exists():
            print(f"Vocab not found: {vocab_path}")
            sys.exit(1)
        print(f"Using local vocab: {vocab_path}")
    else:
        try:
            download_blob(args.bucket, args.remote_vocab, vocab_path)
        except FileNotFoundError:
            print("Warning: models/latest_vocab.txt missing on Firebase — using configs/vocab.txt")
            vocab_path = root / "configs" / "vocab.txt"

    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    manifest = root / args.manifest
    if not manifest.exists():
        print(f"Manifest not found: {manifest}")
        sys.exit(1)

    samples = read_manifest(manifest)[: args.limit]
    if len(samples) < 10:
        print(
            f"WARNING: only {len(samples)} validation samples in {manifest}.\n"
            "  Toy/synthetic val.jsonl (e.g. fake diagonal strokes) is NOT a valid model test.\n"
            "  Prepare real data: data/processed/all.jsonl + scripts/split_data.ps1\n"
            "  or download app corrections to data/corrections/ and retrain."
        )
    tokenizer = CharTokenizer(vocab_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    onnx_scores = []
    rows = []
    for idx, sample in enumerate(samples):
        pred = onnx_greedy_decode(session, tokenizer, sample.points)
        score = cer(pred, sample.text)
        onnx_scores.append(score)
        rows.append(
            {
                "id": idx,
                "truth": sample.text,
                "onnx_prediction": pred,
                "cer": score,
                "mode": sample.mode,
            }
        )
        print(f"[{idx}] truth={sample.text!r} pred={pred!r} cer={score:.3f}")

    report = {
        "samples": len(samples),
        "cer_mean": sum(onnx_scores) / max(1, len(onnx_scores)),
        "vocab_size": len(tokenizer),
        "onnx_bytes": onnx_path.stat().st_size,
        "vocab_path": str(vocab_path),
    }

    ckpt = root / args.checkpoint
    if ckpt.exists():
        device = torch.device("cpu")
        model, pt_tok = load_pytorch_baseline(config, ckpt, device)
        pt_scores = []
        for idx, sample in enumerate(samples):
            pred = greedy_decode(model, pt_tok, sample.points, device)
            pt_scores.append(cer(pred, sample.text))
            rows[idx]["pytorch_prediction"] = pred
            rows[idx]["pytorch_cer"] = rows[idx]["cer"]
        report["pytorch_cer_mean"] = sum(pt_scores) / max(1, len(pt_scores))

    asset_vocab = root / "configs" / "vocab.txt"
    if asset_vocab.exists() and asset_vocab.resolve() != Path(vocab_path).resolve():
        asset_tok = CharTokenizer(asset_vocab)
        if len(asset_tok) != len(tokenizer):
            report["vocab_mismatch"] = {
                "firebase_vocab_size": len(tokenizer),
                "local_vocab_size": len(asset_tok),
                "warning": "APK vocab may not match Firebase model — update app assets or OTA vocab",
            }

    report_path = out_dir / "firebase_verify_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "firebase_verify_samples.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    print("\n=== Summary ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    if len(samples) < 10:
        print("INCONCLUSIVE: too few val samples — CER is not meaningful. See WARNING above.")
        sys.exit(0)
    if report.get("vocab_mismatch"):
        print(f"WARN: {report['vocab_mismatch']['warning']}")
    if report["cer_mean"] > 0.5:
        print("FAIL: cer_mean > 0.5 — Firebase model likely broken or vocab mismatch.")
        sys.exit(2)
    if report["cer_mean"] > 0.25:
        print("WARN: cer_mean > 0.25 — model quality is weak.")
    else:
        print("OK: Firebase ONNX looks reasonable on validation samples.")


if __name__ == "__main__":
    main()
