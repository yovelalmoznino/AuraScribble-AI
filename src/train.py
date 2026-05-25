from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from dataset import (
    HandwritingSample,
    maybe_augment_relative_features,
    points_to_relative_features,
    read_firebase_corrections,
    read_manifest,
)
from decode import greedy_decode
from metrics import cer
from model import HandwritingSeq2SeqModel
from tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class HandwritingDataset(Dataset):
    def __init__(
        self,
        samples: list[HandwritingSample],
        tokenizer: CharTokenizer,
        max_seq_len: int,
        max_tgt_len: int,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_tgt_len = max_tgt_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        sample = self.samples[idx]
        feats = points_to_relative_features(sample.points)
        if self.augment:
            feats = maybe_augment_relative_features(feats, True)
        if len(feats) > self.max_seq_len:
            feats = feats[: self.max_seq_len]

        input_tensor = torch.tensor(feats, dtype=torch.float32)
        token_ids = self.tokenizer.encode(sample.text, add_special_tokens=True)
        if len(token_ids) > self.max_tgt_len:
            token_ids = token_ids[: self.max_tgt_len]

        decoder_in = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_tensor, decoder_in, targets, sample.text, sample.mode


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]],
    pad_id: int,
) -> dict:
    inputs, decoder_ins, targets, texts, modes = zip(*batch)
    input_lens = torch.tensor([len(x) for x in inputs], dtype=torch.long)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    decoder_padded = pad_sequence(decoder_ins, batch_first=True, padding_value=pad_id)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_id)
    return {
        "inputs": inputs_padded,
        "input_lens": input_lens,
        "decoder_in": decoder_padded,
        "targets": targets_padded,
        "texts": list(texts),
        "modes": list(modes),
    }


def _load_checkpoint_vocab(model_path: str, device: torch.device) -> tuple[dict | None, list[str] | None]:
    path = Path(model_path)
    if not path.exists():
        return None, None
    print(f"Checking checkpoint {model_path} for original vocabulary...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "vocab" in checkpoint:
        vocab = checkpoint["vocab"]
        print(f"Success! Rescued original vocabulary with {len(vocab)} characters.")
        return checkpoint, vocab
    return checkpoint, None


def _build_tokenizer(config: dict, rescued_vocab: list[str] | None) -> CharTokenizer:
    tokenizer = CharTokenizer(config["vocab_path"])
    if rescued_vocab:
        tokenizer.vocab = [
            v.replace("\n", "") for v in rescued_vocab if v.replace("\n", "") != "" or v == " "
        ]
        tokenizer.stoi = {t: i for i, t in enumerate(tokenizer.vocab)}
        try:
            vocab_p = Path(config["vocab_path"])
            vocab_p.parent.mkdir(parents=True, exist_ok=True)
            with vocab_p.open("w", encoding="utf-8") as vf:
                for v in tokenizer.vocab:
                    vf.write(f"{v}\n")
        except OSError as exc:
            print(f"Warning: Could not write vocab file: {exc}")
    return tokenizer


def _evaluate_val_cer(
    model: HandwritingSeq2SeqModel,
    tokenizer: CharTokenizer,
    val_samples: list[HandwritingSample],
    device: torch.device,
    max_seq_len: int,
    max_samples: int | None = None,
) -> tuple[float, dict[str, float]]:
    if not val_samples:
        return float("inf"), {}

    subset = val_samples
    if max_samples is not None and max_samples > 0 and len(val_samples) > max_samples:
        rng = random.Random(1337)
        subset = rng.sample(val_samples, max_samples)

    model.eval()
    scores: list[float] = []
    by_mode: dict[str, list[float]] = {}
    for sample in subset:
        pred = greedy_decode(
            model,
            tokenizer,
            sample.points,
            device,
            max_seq_len=max_seq_len,
        )
        score = cer(pred, sample.text)
        scores.append(score)
        key = sample.mode.lower()
        by_mode.setdefault(key, []).append(score)

    mean_cer = sum(scores) / max(1, len(scores))
    mode_means = {k: sum(v) / len(v) for k, v in by_mode.items()}
    model.train()
    return mean_cer, mode_means


def _save_checkpoint(
    path: Path,
    model: HandwritingSeq2SeqModel,
    tokenizer: CharTokenizer,
    config: dict,
    val_cer: float,
    epoch: int,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": tokenizer.vocab,
            "config": config,
            "val_cer": val_cer,
            "epoch": epoch,
        },
        path,
    )


def train(
    config_path: str,
    corrections_dir: str | None = None,
    data_path: str | None = None,
    epochs: int | None = None,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    out_dir = Path(config.get("output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = config.get("model_path", "models/checkpoint_best.pt")
    checkpoint, rescued_vocab = _load_checkpoint_vocab(model_path, device)
    tokenizer = _build_tokenizer(config, rescued_vocab)

    print(f"Loading training data from {config['train_manifest']}...")
    train_samples = read_manifest(config["train_manifest"])
    if corrections_dir:
        print(f"Integrating corrections from {corrections_dir}...")
        train_samples.extend(read_firebase_corrections(corrections_dir))
    if data_path:
        print(f"Integrating master data from {data_path}...")
        train_samples.extend(read_manifest(data_path))

    val_manifest = config.get("val_manifest")
    val_samples: list[HandwritingSample] = []
    if val_manifest and Path(val_manifest).exists():
        val_samples = read_manifest(val_manifest)
        print(f"Loaded {len(val_samples)} validation samples from {val_manifest}")
    else:
        print(f"Warning: validation manifest missing or empty: {val_manifest}")

    train_ds = HandwritingDataset(
        train_samples,
        tokenizer,
        config["max_seq_len"],
        config["max_tgt_len"],
        augment=True,
    )
    pad_id = tokenizer.pad_id
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=pad_id),
        num_workers=config.get("num_workers", 0),
    )

    model = HandwritingSeq2SeqModel(
        input_dim=config["input_dim"],
        hidden=config["hidden_dim"],
        layers=config["num_layers"],
        dropout=config["dropout"],
        vocab_size=len(tokenizer),
    ).to(device)

    if checkpoint is not None:
        print(f"Loading pre-trained weights from {model_path}...")
        state_dict = (
            checkpoint.get("model_state")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        model.load_state_dict(state_dict)
        print("Weights loaded successfully!")
    else:
        print("Warning: no checkpoint found. Training from scratch.")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="none")
    lr = float(config.get("learning_rate", 2e-5))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    grad_clip = float(config.get("grad_clip_norm", 1.0))
    correction_weight = float(config.get("correction_loss_weight", 3.0))
    val_every = int(config.get("val_every_epochs", 1))
    val_max_samples = config.get("val_max_samples")
    val_max_samples = int(val_max_samples) if val_max_samples is not None else None

    epochs_to_run = epochs if epochs is not None else int(config.get("epochs", 10))
    checkpoint_out_path = out_dir / "checkpoint_best.pt"
    log_path = out_dir / "training_log.jsonl"

    best_val_cer = float("inf")
    if isinstance(checkpoint, dict) and "val_cer" in checkpoint:
        best_val_cer = float(checkpoint["val_cer"])

    print(f"Starting training for {epochs_to_run} epochs (lr={lr}, grad_clip={grad_clip})...")

    for epoch in range(epochs_to_run):
        model.train()
        total_loss = 0.0
        mode_stats: dict[str, list[float]] = {}

        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            input_lens = batch["input_lens"].to(device)
            decoder_in = batch["decoder_in"].to(device)
            targets = batch["targets"].to(device)
            modes = batch["modes"]

            optimizer.zero_grad()
            logits = model(inputs, input_lens, decoder_in)
            raw_loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            per_sample_loss = raw_loss.view(targets.size(0), -1).mean(dim=1)

            weights = torch.ones_like(per_sample_loss)
            for i, mode in enumerate(modes):
                if mode.lower() == "correction":
                    weights[i] = correction_weight

            loss = (per_sample_loss * weights).mean()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()

            for i, mode in enumerate(modes):
                mode_key = mode.lower()
                mode_stats.setdefault(mode_key, []).append(per_sample_loss[i].item())

        avg_loss = total_loss / max(1, len(train_loader))
        log_entry: dict = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
        }
        log_str = f"Epoch {epoch + 1}/{epochs_to_run} - Loss: {avg_loss:.4f}"

        for m, losses in sorted(mode_stats.items()):
            if losses:
                m_avg = sum(losses) / len(losses)
                log_str += f" | {m}: {m_avg:.4f}"

        if val_samples and (epoch + 1) % val_every == 0:
            val_cer_mean, mode_cer = _evaluate_val_cer(
                model,
                tokenizer,
                val_samples,
                device,
                config["max_seq_len"],
                max_samples=val_max_samples,
            )
            log_entry["val_cer"] = round(val_cer_mean, 6)
            log_entry["val_mode_cer"] = {k: round(v, 6) for k, v in mode_cer.items()}
            log_str += f" | val_cer: {val_cer_mean:.4f}"
            for m, v in sorted(mode_cer.items()):
                log_str += f" | val_{m}: {v:.4f}"

            if val_cer_mean < best_val_cer:
                best_val_cer = val_cer_mean
                _save_checkpoint(checkpoint_out_path, model, tokenizer, config, best_val_cer, epoch + 1)
                log_str += " [best]"
                print(f"Saved new best checkpoint (val_cer={best_val_cer:.4f})")

        print(log_str)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    if not val_samples:
        _save_checkpoint(checkpoint_out_path, model, tokenizer, config, best_val_cer, epochs_to_run)
        print(f"Saved checkpoint to {checkpoint_out_path} (no validation set)")

    print(f"Training complete. Best val_cer={best_val_cer:.4f}. Checkpoint: {checkpoint_out_path}")
    print("Run export_onnx.py to export for Android.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, help="Path to merged master data (.jsonl)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--corrections_dir", type=str, help="Path to corrections directory")
    args = parser.parse_args()

    train(
        config_path=args.config,
        data_path=args.data_path,
        epochs=args.epochs,
        corrections_dir=args.corrections_dir,
    )
