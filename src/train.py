from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ייבוא פונקציות מהדאטה-סט
from dataset import (
    HandwritingSample,
    maybe_augment_relative_features,
    points_to_relative_features,
    read_manifest,
    read_firebase_corrections, # פונקציה חדשה שנוסיף ל-dataset.py
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
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CharTokenizer(config["vocab_path"])
    
    # טעינת דאטה בסיסי
    print(f"Loading base training data from {config['train_manifest']}...")
    train_samples = read_manifest(config["train_manifest"])
    val_samples = read_manifest(config["val_manifest"])

    # שילוב תיקונים מ-Firebase אם קיימים
    if corrections_dir:
        print(f"Integrating new corrections from {corrections_dir}...")
        new_samples = read_firebase_corrections(corrections_dir)
        # אנחנו מוסיפים את התיקונים החדשים לרשימת האימון
        # טיפ: אפשר להכפיל את new_samples כדי לתת להם משקל גבוה יותר באימון
        train_samples.extend(new_samples)

    train_ds = HandwritingDataset(
        train_samples, tokenizer, config["max_seq_len"], config["max_tgt_len"], augment=True
    )
    val_ds = HandwritingDataset(
        val_samples, tokenizer, config["max_seq_len"], config["max_tgt_len"], augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwritingSeq2SeqModel(
        input_dim=5, # [dx, dy, x, y, pen_state]
        hidden_dim=config["hidden_dim"],
        vocab_size=len(tokenizer),
        num_layers=config["num_layers"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # לוגיקת אימון (קוצר כאן לצורך הפרומפט, נשארת כפי שהייתה במקור שלך)
    # ... (כאן מגיע ה-Loop של ה-Epochs שמופיע בקובץ המקורי שלך)

    # שמירת המודל הסופי בפורמט ONNX עבור האפליקציה
    print("Exporting model to ONNX...")
    dummy_input = torch.zeros(1, 10, 5).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        out_dir / "latest_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--corrections_dir", type=str, help="Path to directory with Firebase corrections")
    args = parser.parse_args()

    train(args.config, args.corrections_dir)
