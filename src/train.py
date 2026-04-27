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
        feats = points_to_relative_features(points)
        
        if self.augment:
            feats = maybe_augment_relative_features(feats, True)

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

# עדכון חתימת הפונקציה לקבלת data_path
def train(config_path: str, corrections_dir: str | None = None, data_path: str | None = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    out_dir = Path(config.get("output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = config.get("model_path", "models/checkpoint_best.pt")
    rescued_vocab = None
    checkpoint = None
    if Path(model_path).exists():
        print(f"Checking checkpoint {model_path} for original vocabulary...")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "vocab" in checkpoint:
            rescued_vocab = checkpoint["vocab"]
            print(f"Success! Rescued original vocabulary with {len(rescued_vocab)} characters.")

    tokenizer = CharTokenizer(config["vocab_path"])
    
    if rescued_vocab:
        tokenizer.vocab = rescued_vocab
        tokenizer.stoi = {t: i for i, t in enumerate(rescued_vocab)}
        tokenizer.blank_id = tokenizer.stoi.get("<blank>", 0)
        tokenizer.pad_id = tokenizer.stoi.get("<pad>", 1)
        tokenizer.bos_id = tokenizer.stoi.get("<bos>", tokenizer.pad_id)
        tokenizer.eos_id = tokenizer.stoi.get("<eos>", tokenizer.pad_id)
        
        with open(config["vocab_path"], "w", encoding="utf-8") as vf:
            for v in rescued_vocab:
                vf.write(f"{v}\n")
    
    print(f"Loading base training data from {config['train_manifest']}...")
    train_samples = read_manifest(config["train_manifest"])
    val_samples = read_manifest(config["val_manifest"])

    # אפשרות 1: טעינה מתיקיית קבצים בודדים (השיטה הישנה)
    if corrections_dir:
        print(f"Integrating new corrections from directory: {corrections_dir}...")
        new_samples = read_firebase_corrections(corrections_dir)
        train_samples.extend(new_samples)

    # אפשרות 2: טעינה מקובץ מאסטר מאוחד (השיטה ההיברידית)
    if data_path:
        print(f"Integrating master data from file: {data_path}...")
        master_samples = read_manifest(data_path)
        train_samples.extend(master_samples)

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
    
    model = HandwritingSeq2SeqModel(
        input_dim=config["input_dim"],
        hidden=config["hidden_dim"], 
        layers=config["num_layers"],
        dropout=config["dropout"],
        vocab_size=len(tokenizer)
    ).to(device)

    if checkpoint is not None:
        print(f"Loading pre-trained weights from {model_path}...")
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print("Weights loaded successfully!")
    else:
        print(f"Warning: Model weights not found. Training from scratch!")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0)
    
    # שימוש ב-Learning Rate מהקונפיג (מומלץ 0.00002 ל-Fine-tuning)
    lr = config.get("learning_rate", config.get("lr", 0.0001))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training loop for {config.get('epochs', 5)} epochs...")
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

    print("Exporting model to ONNX...")
    model.eval()
    max_t = config.get("max_seq_len", 128)
    max_u = config.get("max_tgt_len", 160)
    dummy_src = torch.zeros(1, max_t, config["input_dim"]).to(device)
    dummy_lens = torch.tensor([max_t], dtype=torch.long).to(device)
    dummy_tgt = torch.zeros(1, max_u, dtype=torch.long).to(device)
    
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
    parser.add_argument("--corrections_dir", type=str, help="Path to directory with individual JSON corrections")
    parser.add_argument("--data_path", type=str, help="Path to merged master data (.jsonl)")
    args = parser.parse_args()

    train(args.config, args.corrections_dir, args.data_path)
