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

from dataset import (
    HandwritingSample,
    maybe_augment_relative_features,
    points_to_relative_features,
    read_manifest,
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
        points = sample.points[: self.max_seq_len]
        if not points:
            points = [[0.0, 0.0, 0.0]]
        features = points_to_relative_features(points)
        features = maybe_augment_relative_features(features, enabled=self.augment)
        src = torch.tensor(features, dtype=torch.float32)

        tgt_ids = self.tokenizer.encode(sample.text, rtl_aware=True, add_special_tokens=True)[: self.max_tgt_len]
        if len(tgt_ids) < 2:
            tgt_ids = [self.tokenizer.bos_id, self.tokenizer.eos_id]
        tgt = torch.tensor(tgt_ids, dtype=torch.long)
        return src, tgt, sample.mode


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    srcs = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    modes = [b[2] for b in batch]
    src_lens = torch.tensor([x.shape[0] for x in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([y.shape[0] for y in tgts], dtype=torch.long)
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=0.0)
    tgt_pad = pad_sequence(tgts, batch_first=True, padding_value=pad_id)
    return src_pad, tgt_pad, src_lens, tgt_lens, modes


def seq_greedy_decode(logits: torch.Tensor, tokenizer: CharTokenizer) -> list[str]:
    pred_ids = logits.argmax(dim=-1)  # [B,U]
    out: list[str] = []
    for i in range(pred_ids.shape[0]):
        ids = pred_ids[i].detach().cpu().tolist()
        out.append(tokenizer.decode(ids, rtl_aware=True))
    return out


def cer(pred: str, truth: str) -> float:
    if truth == "":
        return 0.0 if pred == "" else 1.0
    dp = [[0] * (len(truth) + 1) for _ in range(len(pred) + 1)]
    for i in range(len(pred) + 1):
        dp[i][0] = i
    for j in range(len(truth) + 1):
        dp[0][j] = j
    for i in range(1, len(pred) + 1):
        for j in range(1, len(truth) + 1):
            cost = 0 if pred[i - 1] == truth[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / len(truth)


def wer(pred: str, truth: str) -> float:
    pw = pred.split()
    tw = truth.split()
    if not tw:
        return 0.0 if not pw else 1.0
    dp = [[0] * (len(tw) + 1) for _ in range(len(pw) + 1)]
    for i in range(len(pw) + 1):
        dp[i][0] = i
    for j in range(len(tw) + 1):
        dp[0][j] = j
    for i in range(1, len(pw) + 1):
        for j in range(1, len(tw) + 1):
            cost = 0 if pw[i - 1] == tw[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / len(tw)


def _load_state_dict_flexible(
    model: nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str], list[str]]:
    model_state = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    loaded_layers: list[str] = []
    skipped_layers: list[str] = []

    for name, tensor in checkpoint_state.items():
        target = model_state.get(name)
        if target is None:
            skipped_layers.append(f"{name} (missing_in_model)")
            continue
        if target.shape != tensor.shape:
            skipped_layers.append(f"{name} (shape {tuple(tensor.shape)} -> {tuple(target.shape)})")
            continue
        matched[name] = tensor
        loaded_layers.append(name)

    result = model.load_state_dict(matched, strict=False)
    missing_layers = list(result.missing_keys)
    unexpected_layers = list(result.unexpected_keys)
    return loaded_layers, skipped_layers, missing_layers, unexpected_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["training"]["seed"]))
    tokenizer = CharTokenizer.from_file(cfg["vocab"]["file"])
    max_seq_len = int(cfg["training"]["max_seq_len"])
    max_tgt_len = int(cfg["training"].get("max_tgt_len", 160))

    train_manifest = Path(cfg["data"]["train_manifest"])
    val_manifest = Path(cfg["data"]["val_manifest"])
    if not train_manifest.exists() or not val_manifest.exists():
        raise RuntimeError(
            "Processed manifests were not found. Run `python src/preprocess.py` first to generate "
            "`data/processed/train.jsonl` and `data/processed/val.jsonl`."
        )

    train_samples = read_manifest(train_manifest)
    val_samples = read_manifest(val_manifest)

    if not train_samples:
        raise RuntimeError("No training samples found in processed train manifest.")
    if not val_samples:
        raise RuntimeError("No validation samples found in processed validation manifest.")

    train_ds = HandwritingDataset(train_samples, tokenizer, max_seq_len, max_tgt_len=max_tgt_len, augment=True)
    val_ds = HandwritingDataset(val_samples, tokenizer, max_seq_len, max_tgt_len=max_tgt_len, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )

    requested_device = str(cfg["training"]["device"]).lower()
    device = torch.device("cuda" if (requested_device == "auto" and torch.cuda.is_available()) else requested_device if requested_device != "auto" else "cpu")
    model = HandwritingSeq2SeqModel(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden=int(cfg["model"]["encoder_hidden"]),
        layers=int(cfg["model"]["encoder_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        vocab_size=len(tokenizer.vocab),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    seq_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint_latest.pt"
    best_ckpt = out_dir / "checkpoint_best.pt"
    best_cer = float("inf")
    best_wer = float("inf")
    resume_enabled = bool(cfg["training"].get("resume", True))
    early_stopping_enabled = bool(cfg["training"].get("early_stopping", True))
    early_stopping_patience = int(cfg["training"].get("early_stopping_patience", 10))
    early_stopping_min_delta = float(cfg["training"].get("early_stopping_min_delta", 1e-4))
    epochs_without_improvement = 0
    start_epoch = 1

    epochs = int(cfg["training"]["epochs"])
    if resume_enabled and (best_ckpt.exists() or ckpt.exists()):
        resume_ckpt = best_ckpt if best_ckpt.exists() else ckpt
        try:
            state = torch.load(resume_ckpt, map_location=device)
            checkpoint_state = state.get("model_state")
            if checkpoint_state is None:
                raise RuntimeError("Checkpoint is missing 'model_state'.")

            loaded_layers, skipped_layers, missing_layers, unexpected_layers = _load_state_dict_flexible(
                model=model,
                checkpoint_state=checkpoint_state,
            )
            can_resume_optimizer = not skipped_layers and not missing_layers and not unexpected_layers
            if "optimizer_state" in state and can_resume_optimizer:
                opt.load_state_dict(state["optimizer_state"])
            elif "optimizer_state" in state and not can_resume_optimizer:
                print(
                    "Skipped optimizer state resume because model was partially loaded "
                    "(layer mismatches or missing keys detected). Optimizer will start fresh."
                )
            start_epoch = int(state.get("epoch", 0)) + 1
            best_cer = float(state.get("best_val_cer", state.get("val_cer", best_cer)))
            best_wer = float(state.get("best_val_wer", state.get("val_wer", best_wer)))
            if not can_resume_optimizer:
                best_cer = float("inf")
                print(
                    "Partial resume detected: reset best_cer to inf so early stopping "
                    "tracks only the current vocab run."
                )
            print(f"Resumed training from {resume_ckpt} at epoch {start_epoch}")
            print(
                f"Flexible resume: loaded {len(loaded_layers)} layer(s), "
                f"skipped {len(skipped_layers)} layer(s) due to mismatch/missing."
            )
            if skipped_layers:
                print("Skipped layers:")
                for layer in skipped_layers:
                    print(f"  - {layer}")
            if missing_layers:
                print("Model layers left randomly initialized:")
                for layer in missing_layers:
                    print(f"  - {layer}")
            if unexpected_layers:
                print("Unexpected checkpoint layers ignored:")
                for layer in unexpected_layers:
                    print(f"  - {layer}")
        except Exception as exc:
            print(f"Could not resume from checkpoint ({exc}); starting fresh.")

    if start_epoch > epochs:
        print(
            f"Checkpoint already at epoch {start_epoch - 1}, "
            f"which is >= configured epochs ({epochs}). Increase training.epochs to continue."
        )
        return

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        for src_pad, tgt_pad, src_lens, _, _ in tqdm(train_loader, desc=f"train {epoch}/{epochs}", leave=False):
            src_pad = src_pad.to(device)
            tgt_pad = tgt_pad.to(device)
            src_lens = src_lens.to(device)

            tgt_inp = tgt_pad[:, :-1]
            tgt_out = tgt_pad[:, 1:]
            logits = model(src_pad, src_lens, tgt_inp)
            loss = seq_loss(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += float(loss.item())

        model.eval()
        cer_scores: list[float] = []
        wer_scores: list[float] = []
        cer_by_mode: dict[str, list[float]] = {}
        wer_by_mode: dict[str, list[float]] = {}
        val_examples: list[dict[str, str]] = []
        with torch.no_grad():
            for src_pad, tgt_pad, src_lens, tgt_lens, modes in val_loader:
                src_pad = src_pad.to(device)
                tgt_pad = tgt_pad.to(device)
                src_lens = src_lens.to(device)
                tgt_inp = tgt_pad[:, :-1]
                logits = model(src_pad, src_lens, tgt_inp)
                preds = seq_greedy_decode(logits, tokenizer)
                truths = []
                for i in range(tgt_pad.shape[0]):
                    ids = tgt_pad[i, : tgt_lens[i]].tolist()
                    truths.append(tokenizer.decode(ids, rtl_aware=True))
                for p, t, m in zip(preds, truths, modes):
                    c = cer(p, t)
                    w = wer(p, t)
                    cer_scores.append(c)
                    wer_scores.append(w)
                    key = (m or "auto").lower()
                    cer_by_mode.setdefault(key, []).append(c)
                    wer_by_mode.setdefault(key, []).append(w)
                for p, t in zip(preds[:3], truths[:3]):
                    val_examples.append({"prediction": p, "truth": t})
        val_cer = sum(cer_scores) / max(1, len(cer_scores))
        val_wer = sum(wer_scores) / max(1, len(wer_scores))
        val_cer_by_mode = {
            mode: (sum(values) / len(values))
            for mode, values in sorted(cer_by_mode.items())
            if values
        }
        val_wer_by_mode = {
            mode: (sum(values) / len(values))
            for mode, values in sorted(wer_by_mode.items())
            if values
        }
        print(
            f"epoch={epoch} "
            f"train_loss={running/max(1,len(train_loader)):.4f} "
            f"val_cer={val_cer:.4f} val_wer={val_wer:.4f}"
        )
        if val_cer_by_mode:
            print(f"val_cer_by_mode={json.dumps(val_cer_by_mode, ensure_ascii=False)}")

        payload = {
            "epoch": epoch,
            "val_cer": val_cer,
            "val_wer": val_wer,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "config": cfg,
            "vocab": tokenizer.vocab,
            "best_val_cer": min(best_cer, val_cer),
            "best_val_wer": min(best_wer, val_wer),
            "val_cer_by_mode": val_cer_by_mode,
            "val_wer_by_mode": val_wer_by_mode,
        }
        torch.save(payload, ckpt)
        if val_cer < (best_cer - early_stopping_min_delta):
            best_cer = val_cer
            best_wer = val_wer
            epochs_without_improvement = 0
            torch.save(payload, best_ckpt)
            (out_dir / "best_val_examples.json").write_text(
                json.dumps(val_examples[:20], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            epochs_without_improvement += 1

        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: no val_cer improvement greater than "
                f"{early_stopping_min_delta} for {early_stopping_patience} epoch(s)."
            )
            break

    summary = {
        "best_val_cer": best_cer,
        "best_val_wer": best_wer,
        "checkpoint_latest": str(ckpt),
        "checkpoint_best": str(best_ckpt),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_mode_counts": _count_modes(train_samples),
        "val_mode_counts": _count_modes(val_samples),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Training complete. Best CER={best_cer:.4f}")


def _count_modes(samples: list[HandwritingSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = (sample.mode or "auto").lower()
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
