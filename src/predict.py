from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from dataset import read_manifest
from decode import greedy_decode
from model import HandwritingSeq2SeqModel
from tokenizer import CharTokenizer


def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint.get("vocab")
    if vocab is None:
        raise RuntimeError(f"Checkpoint {checkpoint_path} is missing vocab")

    tokenizer = CharTokenizer(config["vocab_path"])
    tokenizer.vocab = [v.replace("\n", "") for v in vocab if v.replace("\n", "") != "" or v == " "]
    tokenizer.stoi = {t: i for i, t in enumerate(tokenizer.vocab)}

    model = HandwritingSeq2SeqModel(
        input_dim=config["input_dim"],
        hidden=config["hidden_dim"],
        layers=config["num_layers"],
        dropout=config["dropout"],
        vocab_size=len(tokenizer),
    ).to(device)
    state_dict = (
        checkpoint.get("model_state")
        or checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
    )
    if state_dict is None:
        raise RuntimeError("Checkpoint has no model weights")
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default="output/checkpoint_best.pt")
    parser.add_argument("--manifest", default="data/processed/val.jsonl")
    parser.add_argument("--output", default="output/predictions.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, tokenizer = load_model_from_checkpoint(checkpoint_path, config, device)
    samples = read_manifest(args.manifest)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            pred = greedy_decode(
                model,
                tokenizer,
                sample.points,
                device,
                max_seq_len=config["max_seq_len"],
                max_steps=int(config.get("decode_max_steps", 96)),
                max_tgt_window=int(config.get("decode_max_tgt_window", 96)),
            )
            f.write(json.dumps({"id": idx, "prediction": pred}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} predictions to {out_path}")


if __name__ == "__main__":
    main()
