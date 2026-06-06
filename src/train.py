from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
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
from decode_quality import is_template_collapse
from metrics import cer
from model_factory import build_model
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        sample = self.samples[idx]
        feats = points_to_relative_features(sample.points)
        if self.augment:
            feats = maybe_augment_relative_features(feats, True)
        if len(feats) > self.max_seq_len:
            feats = feats[: self.max_seq_len]

        input_tensor = torch.tensor(feats, dtype=torch.float32)
        token_ids = self.tokenizer.encode(
            sample.text,
            add_special_tokens=True,
            mode=sample.mode,
            add_mode_prefix=bool(self.tokenizer.mode_prefix_id(sample.mode) is not None),
        )
        if len(token_ids) > self.max_tgt_len:
            token_ids = token_ids[: self.max_tgt_len]

        decoder_in = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets = torch.tensor(token_ids[1:], dtype=torch.long)
        # CTC targets intentionally exclude BOS/EOS and mode-prefix tokens.
        ctc_ids = self.tokenizer.encode(
            sample.text,
            add_special_tokens=False,
            mode=sample.mode,
            add_mode_prefix=False,
        )
        ctc_targets = torch.tensor(ctc_ids, dtype=torch.long)
        return input_tensor, decoder_in, targets, ctc_targets, sample.text, sample.mode


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]],
    pad_id: int,
) -> dict:
    inputs, decoder_ins, targets, ctc_targets, texts, modes = zip(*batch)
    input_lens = torch.tensor([len(x) for x in inputs], dtype=torch.long)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    decoder_padded = pad_sequence(decoder_ins, batch_first=True, padding_value=pad_id)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_id)
    ctc_target_lens = torch.tensor([len(x) for x in ctc_targets], dtype=torch.long)
    ctc_targets_padded = pad_sequence(ctc_targets, batch_first=True, padding_value=pad_id)
    return {
        "inputs": inputs_padded,
        "input_lens": input_lens,
        "decoder_in": decoder_padded,
        "targets": targets_padded,
        "ctc_targets": ctc_targets_padded,
        "ctc_target_lens": ctc_target_lens,
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


def _filter_samples(
    samples: list[HandwritingSample],
    *,
    min_points: int = 3,
    min_text_len: int = 1,
) -> list[HandwritingSample]:
    kept: list[HandwritingSample] = []
    dropped = 0
    for sample in samples:
        text = (sample.text or "").strip()
        if len(text) < min_text_len or len(sample.points) < min_points:
            dropped += 1
            continue
        feats = points_to_relative_features(sample.points)
        if len(feats) < 2:
            dropped += 1
            continue
        kept.append(sample)
    if dropped:
        print(f"Filtered out {dropped} invalid/short samples ({len(kept)} kept).", flush=True)
    return kept


def _mode_loss_weight(mode: str, config: dict) -> float:
    weights = config.get("mode_loss_weights") or {}
    if not isinstance(weights, dict):
        weights = {}
    key = (mode or "auto").lower()
    if key in weights:
        return float(weights[key])
    if key == "correction":
        return float(config.get("correction_loss_weight", 3.0))
    return 1.0


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
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    val_samples: list[HandwritingSample],
    device: torch.device,
    max_seq_len: int,
    max_samples: int | None = None,
    config: dict | None = None,
) -> tuple[float, dict[str, float]]:
    if not val_samples:
        return float("inf"), {}

    subset = val_samples
    if max_samples is not None and max_samples > 0 and len(val_samples) > max_samples:
        rng = random.Random(1337)
        subset = rng.sample(val_samples, max_samples)

    cfg = config or {}
    decode_steps = int(cfg.get("decode_max_steps", 128))
    decode_window = int(cfg.get("decode_max_tgt_window", 128))
    repetition_penalty = float(cfg.get("decode_repetition_penalty", 2.0))

    model.eval()
    scores: list[float] = []
    by_mode: dict[str, list[float]] = {}
    template_collapses = 0
    for sample in subset:
        pred = greedy_decode(
            model,
            tokenizer,
            sample.points,
            device,
            max_seq_len=max_seq_len,
            max_steps=decode_steps,
            max_tgt_window=decode_window,
            repetition_penalty=repetition_penalty,
            mode=sample.mode,
        )
        if is_template_collapse(pred, sample.text, sample.mode):
            template_collapses += 1
        score = cer(pred, sample.text)
        scores.append(score)
        key = sample.mode.lower()
        by_mode.setdefault(key, []).append(score)

    mean_cer = sum(scores) / max(1, len(scores))
    mode_means = {k: sum(v) / len(v) for k, v in by_mode.items()}
    collapse_rate = template_collapses / max(1, len(scores))
    if subset:
        print(
            f"  val template_collapse: {template_collapses}/{len(scores)} ({collapse_rate:.1%})",
            flush=True,
        )
    model.train()
    return mean_cer, mode_means


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
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

    resume = bool(config.get("resume_from_checkpoint", True))
    model_path = config.get("model_path", "models/checkpoint_best.pt")
    checkpoint, rescued_vocab = (None, None)
    if resume and model_path and Path(model_path).exists():
        checkpoint, rescued_vocab = _load_checkpoint_vocab(model_path, device)
    elif resume and model_path:
        print(f"No checkpoint at {model_path} — training from scratch.", flush=True)
    else:
        print("resume_from_checkpoint=false — training from scratch (fresh weights).", flush=True)

    tokenizer = _build_tokenizer(config, rescued_vocab if resume else None)

    print(f"Loading training data from {config['train_manifest']}...", flush=True)
    train_samples = read_manifest(config["train_manifest"])
    if corrections_dir:
        print(f"Integrating corrections from {corrections_dir}...", flush=True)
        train_samples.extend(read_firebase_corrections(corrections_dir))
    if data_path:
        print(f"Integrating master data from {data_path}...", flush=True)
        train_samples.extend(read_manifest(data_path))

    min_pts = int(config.get("min_points", 3))
    min_txt = int(config.get("min_text_len", 1))
    train_samples = _filter_samples(train_samples, min_points=min_pts, min_text_len=min_txt)

    val_manifest = config.get("val_manifest")
    val_samples: list[HandwritingSample] = []
    if val_manifest and Path(val_manifest).exists():
        val_samples = _filter_samples(
            read_manifest(val_manifest),
            min_points=min_pts,
            min_text_len=min_txt,
        )
        print(f"Loaded {len(val_samples)} validation samples from {val_manifest}", flush=True)
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

    model = build_model(config, len(tokenizer)).to(device)

    if checkpoint is not None:
        print(f"Loading pre-trained weights from {model_path}...", flush=True)
        state_dict = (
            checkpoint.get("model_state")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
        )
        if not isinstance(state_dict, dict):
            print("Checkpoint has no model_state — training from scratch.", flush=True)
        else:
            try:
                model.load_state_dict(state_dict, strict=True)
                print("Weights loaded successfully!", flush=True)
            except RuntimeError as exc:
                print(f"Checkpoint incompatible ({exc}) — training from scratch.", flush=True)

    label_smoothing = float(config.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    criterion_ctc = nn.CTCLoss(blank=tokenizer.blank_id, reduction="none", zero_infinity=True)
    ctc_loss_weight = float(config.get("ctc_loss_weight", 0.3))
    ar_loss_weight = float(config.get("ar_loss_weight", 0.7))
    use_hybrid_loss = bool(config.get("model_type", "transformer").lower() == "hybrid")
    lr = float(config.get("learning_rate", 2e-5))
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    grad_clip = float(config.get("grad_clip_norm", 1.0))
    use_amp = bool(config.get("use_amp", device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    val_every = int(config.get("val_every_epochs", 1))
    val_max_samples = config.get("val_max_samples")
    val_max_samples = int(val_max_samples) if val_max_samples is not None else None

    epochs_to_run = epochs if epochs is not None else int(config.get("epochs", 10))
    checkpoint_out_path = out_dir / "checkpoint_best.pt"
    log_path = out_dir / "training_log.jsonl"
    early_stop_patience = int(config.get("early_stop_patience", 0))
    early_stop_min_delta = float(config.get("early_stop_min_delta", 0.0))

    best_val_cer = float("inf")
    if isinstance(checkpoint, dict) and "val_cer" in checkpoint:
        # When the val set composition changes between runs (e.g. we added a lot
        # more math samples), carrying over the old best_val_cer makes the new
        # run unable to ever "beat" it, so no checkpoint gets written. Allow the
        # caller to opt into resetting it.
        if config.get("reset_best_val_cer_on_resume", False):
            print(
                f"Resume: ignoring previous best_val_cer={float(checkpoint['val_cer']):.4f} "
                "(reset_best_val_cer_on_resume=true); first new epoch will set baseline.",
                flush=True,
            )
        else:
            best_val_cer = float(checkpoint["val_cer"])

    n_batches = len(train_loader)
    log_every = int(config.get("log_every_batches", 25))
    epochs_without_improve = 0

    # Linear warmup -> cosine decay. Critical for stable from-scratch
    # transformer training; the warmup resets at the start of each
    # curriculum phase (each phase calls train() fresh), which is fine.
    warmup_frac = float(config.get("warmup_frac", 0.05))
    total_steps = max(1, n_batches * epochs_to_run)
    warmup_steps = max(1, int(total_steps * warmup_frac))

    # Cosine decays to a floor (lr_min_frac of peak) rather than to 0, so
    # the model keeps learning through the late epochs instead of stalling
    # when the LR collapses — important for an underfitting model.
    lr_min_frac = float(config.get("lr_min_frac", 0.1))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return lr_min_frac + (1.0 - lr_min_frac) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    print(
        f"Starting training for {epochs_to_run} epochs "
        f"({n_batches} batches/epoch, log_every={log_every}, lr={lr}, "
        f"warmup_steps={warmup_steps}/{total_steps}, "
        f"weight_decay={weight_decay}, early_stop_patience={early_stop_patience}, "
        f"hybrid_loss={'on' if use_hybrid_loss else 'off'})...",
        flush=True,
    )

    for epoch in range(epochs_to_run):
        model.train()
        total_loss = 0.0
        total_ar_loss = 0.0
        total_ctc_loss = 0.0
        mode_stats: dict[str, list[float]] = {}
        epoch_t0 = time.perf_counter()
        print(f"Epoch {epoch + 1}/{epochs_to_run} started...", flush=True)

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["inputs"].to(device)
            input_lens = batch["input_lens"].to(device)
            decoder_in = batch["decoder_in"].to(device)
            targets = batch["targets"].to(device)
            ctc_targets = batch["ctc_targets"].to(device)
            ctc_target_lens = batch["ctc_target_lens"].to(device)
            modes = batch["modes"]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(inputs, input_lens, decoder_in)
                raw_loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                per_sample_ar = raw_loss.view(targets.size(0), -1).mean(dim=1)
                per_sample_loss = per_sample_ar
                per_sample_ctc = None
                if use_hybrid_loss and hasattr(model, "ctc_logits"):
                    ctc_logits, ctc_input_lens = model.ctc_logits(inputs, input_lens)
                    ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
                    per_sample_ctc = criterion_ctc(
                        ctc_log_probs,
                        ctc_targets,
                        ctc_input_lens,
                        ctc_target_lens,
                    )
                    # CTCLoss(reduction="none") can scale with target length.
                    # Normalize to keep CTC and AR on comparable scales.
                    per_sample_ctc = per_sample_ctc / torch.clamp(
                        ctc_target_lens.to(per_sample_ctc.dtype),
                        min=1.0,
                    )
                    per_sample_loss = ar_loss_weight * per_sample_ar + ctc_loss_weight * per_sample_ctc
                weights = torch.tensor(
                    [_mode_loss_weight(m, config) for m in modes],
                    device=per_sample_loss.device,
                    dtype=per_sample_loss.dtype,
                )
                loss = (per_sample_loss * weights).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            total_ar_loss += float(per_sample_ar.mean().item())
            if per_sample_ctc is not None:
                total_ctc_loss += float(per_sample_ctc.mean().item())

            for i, mode in enumerate(modes):
                mode_key = mode.lower()
                mode_stats.setdefault(mode_key, []).append(per_sample_loss[i].item())

            if log_every > 0 and (batch_idx + 1) % log_every == 0:
                print(
                    f"  epoch {epoch + 1} batch {batch_idx + 1}/{n_batches} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        avg_loss = total_loss / max(1, n_batches)
        epoch_sec = time.perf_counter() - epoch_t0
        log_entry: dict = {
            "epoch": epoch + 1,
            "train_joint_loss": round(avg_loss, 6),
            "train_ar_loss": round(total_ar_loss / max(1, n_batches), 6),
        }
        if use_hybrid_loss:
            log_entry["train_ctc_loss"] = round(total_ctc_loss / max(1, n_batches), 6)
        log_str = (
            f"Epoch {epoch + 1}/{epochs_to_run} - Loss: {avg_loss:.4f} "
            f"({epoch_sec / 60:.1f} min)"
        )
        log_str += f" | ar_loss: {log_entry['train_ar_loss']:.4f}"
        if use_hybrid_loss:
            log_str += f" | ctc_loss: {log_entry.get('train_ctc_loss', 0.0):.4f}"

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
                config=config,
            )
            log_entry["val_cer"] = round(val_cer_mean, 6)
            log_entry["val_mode_cer"] = {k: round(v, 6) for k, v in mode_cer.items()}
            log_str += f" | val_cer: {val_cer_mean:.4f}"
            for m, v in sorted(mode_cer.items()):
                log_str += f" | val_{m}: {v:.4f}"

            improved = val_cer_mean < (best_val_cer - early_stop_min_delta)
            if improved:
                best_val_cer = val_cer_mean
                epochs_without_improve = 0
                _save_checkpoint(checkpoint_out_path, model, tokenizer, config, best_val_cer, epoch + 1)
                log_str += " [best]"
                print(f"Saved new best checkpoint (val_cer={best_val_cer:.4f})")
            else:
                epochs_without_improve += 1
                log_str += f" (no_improve {epochs_without_improve}/{early_stop_patience})"

        print(log_str, flush=True)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if (
            early_stop_patience > 0
            and val_samples
            and (epoch + 1) % val_every == 0
            and epochs_without_improve >= early_stop_patience
        ):
            print(
                f"Early stopping: val_cer did not improve for {early_stop_patience} epochs "
                f"(best={best_val_cer:.4f}).",
                flush=True,
            )
            break

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
