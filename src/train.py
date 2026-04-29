from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.export
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
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
        # החזרה של ה-mode (סוג המידע) לצורך לוגים מפורטים
        return input_tensor, target_tensor, sample.text, sample.mode


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, str, str]]) -> dict[str, torch.Tensor | list[str]]:
    inputs, targets, texts, modes = zip(*batch)
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
        "modes": list(modes),
    }


def train(config_path: str, corrections_dir: str | None = None, data_path: str | None = None, epochs: int | None = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    out_dir = Path(config.get("output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- טעינת מילון מה-Checkpoint ---
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
        tokenizer.vocab = [v.replace('\n', '') for v in rescued_vocab if v.replace('\n', '') != '' or v == ' ']
        tokenizer.stoi = {t: i for i, t in enumerate(tokenizer.vocab)}
        
        try:
            vocab_p = Path(config["vocab_path"])
            vocab_p.parent.mkdir(parents=True, exist_ok=True)
            with open(vocab_p, "w", encoding="utf-8") as vf:
                for v in tokenizer.vocab: vf.write(f"{v}\n")
        except Exception as e:
            print(f"Warning: Could not write vocab file to original path: {e}. Keeping in memory.")
    
    # --- טעינת נתונים ---
    print(f"Loading base training data from {config['train_manifest']}...")
    train_samples = read_manifest(config["train_manifest"])

    if corrections_dir:
        print(f"Integrating new corrections from directory: {corrections_dir}...")
        new_samples = read_firebase_corrections(corrections_dir)
        train_samples.extend(new_samples)

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
    
    # --- בניית המודל ---
    model = HandwritingSeq2SeqModel(
        input_dim=config["input_dim"],
        hidden=config["hidden_dim"], 
        layers=config["num_layers"],
        dropout=config["dropout"],
        vocab_size=len(tokenizer)
    ).to(device)

    # --- טעינת משקולות ---
    if checkpoint is not None:
        print(f"Loading pre-trained weights from {model_path}...")
        state_dict = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        model.load_state_dict(state_dict)
        print("Weights loaded successfully!")
    else:
        print(f"Warning: Model weights not found. Training from scratch!")

    # שימוש ב-reduction='none' כדי לאפשר חישוב סטטיסטיקה לפי סוג מידע
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction='none')
    
    # --- הגדרת אימון ---
    lr = config.get("learning_rate", config.get("lr", 0.0001))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs_to_run = epochs if epochs is not None else config.get("epochs", 5)
    print(f"Starting training loop for {epochs_to_run} epochs...")

    model.train()
    for epoch in range(epochs_to_run):
        total_loss = 0.0
        # מילון למעקב אחרי ה-Loss לפי סוג (mode)
        mode_stats = {"english": [], "math": [], "hebrew": [], "synthetic": [], "correction": []}
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["inputs"].to(device)
            input_lens = batch["input_lens"].to(device)
            targets = batch["targets"].to(device)
            modes = batch["modes"]
            
            optimizer.zero_grad()
            logits = model(inputs, input_lens, targets)
            
            # חישוב ה-Loss הגולמי
            raw_loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            # חישוב ממוצע לכל דוגמה בבאץ'
            per_sample_loss = raw_loss.view(targets.size(0), -1).mean(dim=1)
            
            # שיוך ה-Loss לסוג המידע המתאים
            for i, mode in enumerate(modes):
                mode_key = mode.lower()
                if mode_key in mode_stats:
                    mode_stats[mode_key].append(per_sample_loss[i].item())
                else:
                    # טיפול במקרים של שמות מצבים לא צפויים
                    if mode_key not in mode_stats: mode_stats[mode_key] = []
                    mode_stats[mode_key].append(per_sample_loss[i].item())
            
            loss = per_sample_loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(train_loader))
        
        # הדפסת לוג מפורט בסוף האפוק
        log_str = f"Epoch {epoch+1}/{epochs_to_run} - Loss: {avg_loss:.4f}"
        for m, losses in mode_stats.items():
            if losses:
                m_avg = sum(losses) / len(losses)
                log_str += f" | {m}: {m_avg:.4f}"
        print(log_str)

    # --- שמירת Checkpoint מעודכן ---
    checkpoint_out_path = out_dir / "checkpoint_best.pt"
    torch.save({
        "model_state": model.state_dict(),
        "vocab": tokenizer.vocab,
        "config": config,
    }, checkpoint_out_path)
    print(f"Updated checkpoint saved to {checkpoint_out_path}")

    # --- ייצוא ל-ONNX ---
    print("Exporting model to ONNX...")
    model.eval()
    model.to('cpu') 

    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            m.flatten_parameters()

    input_dim = config["input_dim"]
    dummy_src = torch.randn(1, 10, input_dim)
    dummy_lens = torch.tensor([10], dtype=torch.long)
    dummy_tgt = torch.zeros(1, 1, dtype=torch.long)
    dummy_inputs = (dummy_src, dummy_lens, dummy_tgt)
    
    try:
        # עדכון שם הקובץ ל-handwriting.onnx כפי שהוגדר
        onnx_file_path = out_dir / "handwriting.onnx"
        print(f"🔄 מייצא ל-ONNX (handwriting.onnx) באמצעות dynamic_shapes...")

        # הגדרת ממדים דינמיים לפי הפרוטוקול העדכני
        d_batch = torch.export.Dim("batch_size", min=1, max=1024)
        d_seq = torch.export.Dim("seq_len", min=1, max=2048)
        d_tgt_len = torch.export.Dim("tgt_len", min=1, max=512)
        
        torch.onnx.export(
            model,
            dummy_inputs,
            str(onnx_file_path),
            export_params=True,
            opset_version=18, 
            do_constant_folding=True,
            input_names=['inputs', 'input_lens', 'targets'],
            output_names=['output'],
            # שימוש במפתחות התואמים לחתימת המודל: src, src_lens, tgt_inp
            dynamic_shapes={
                'src': {0: d_batch, 1: d_seq},
                'src_lens': {0: d_batch},
                'tgt_inp': {0: d_batch, 1: d_tgt_len}
            }
        )
        print(f"✅ ONNX export successful! Saved to {onnx_file_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        

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
        corrections_dir=args.corrections_dir
    )
