from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ייבוא פונקציות מהדאטה-סט
from dataset import (
    HandwritingSample,
    maybe_augment_relative_features,
    points_to_relative_features,
    read_manifest,
    read_firebase_corrections, 
)
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        points = sample.points
        if self.augment:
            points = maybe_augment_relative_features(points)

        feats = points_to_relative_features(points)
        if len(feats) > self.max_seq_len:
            feats = feats[: self.max_seq_len]
        
        input_tensor = torch.tensor(feats, dtype=torch.float32)
        target_indices = self.tokenizer.encode(sample.text)
        if len(target_indices) > self.max_tgt_len:
            target_indices = target_indices[: self.max_tgt_len]
        
        target_tensor = torch.tensor(target_indices, dtype=torch.long)
        return input_tensor, target_tensor, sample.text


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, str]]) -> dict[str, torch.Tensor | list[str]]:
    inputs, targets, texts = zip(*batch)
    input_lens = torch.tensor([len(x) for x in inputs], dtype=torch.long)
    target_lens = torch.tensor([len(y) for y in targets], dtype=torch.long)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return {
        "inputs": inputs_padded,
        "input_lens": input_lens,
        "targets": targets_padded,
        "target_lens": target_lens,
        "texts": list(texts),
    }


def train(config_path: str, corrections_dir: str | None = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    out_dir = Path(config.get("output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CharTokenizer(config["vocab_path"])
    
    print(f"Loading base training data from {config['train_manifest']}...")
    train_samples = read_manifest(config["train_manifest"])
    val_samples = read_manifest(config["val_manifest"])

    if corrections_dir:
        print(f"Integrating new corrections from {corrections_dir}...")
        new_samples = read_firebase_corrections(corrections_dir)
        train_samples.extend(new_samples)

    train_ds = HandwritingDataset(
        train_samples, tokenizer, config["max_seq_len"], config["max_tgt_len"], augment=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get("num_workers", 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- התיקון המרכזי: התאמת השמות בדיוק לפונקציה ב-model.py ---
    model = HandwritingSeq2SeqModel(
        input_dim=config["input_dim"],
        hidden=config["hidden_dim"], 
        layers=config["num_layers"],
        dropout=config["dropout"],
        vocab_size=len(tokenizer)
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0)
    
    # --- מוקש 1 שנוטרל: התאמת שם ה-Learning Rate ל-YAML ---
    lr = config.get("learning_rate", config.get("lr", 0.0001))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- הוספת לולאת האימון (הייתה חסרה בקובץ שהעלית) ---
    print("Starting training loop...")
    epochs = config.get("epochs", 5)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["inputs"].to(device)
            input_lens = batch["input_lens"].to(device)
            targets = batch["targets"].to(device)
            
            optimizer.zero_grad()
            logits = model(inputs, input_lens, targets)
            
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # --- מוקש 2 שנוטרל: התאמת ייצוא ה-ONNX ל-3 הפרמטרים שהמודל דורש ---
    print("Exporting model to ONNX...")
    model.eval()
    dummy_src = torch.zeros(1, 10, config["input_dim"]).to(device)
    dummy_lens = torch.tensor([10], dtype=torch.long).to(device)
    dummy_tgt = torch.zeros(1, 5, dtype=torch.long).to(device)
    
    torch.onnx.export(
        model,
        (dummy_src, dummy_lens, dummy_tgt),
        out_dir / "latest_model.onnx",
        input_names=["src", "src_lens", "tgt_inp"],
        output_names=["logits"],
        dynamic_axes={
            "src": {0: "batch", 1: "seq"}, 
            "tgt_inp": {0: "batch", 1: "seq_tgt"},
            "logits": {0: "batch", 1: "seq_tgt"}
        },
    )
    print("Training and export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--corrections_dir", type=str, help="Path to directory with Firebase corrections")
    args = parser.parse_args()

    train(args.config, args.corrections_dir)
