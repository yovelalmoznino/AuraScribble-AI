from __future__ import annotations

import numpy as np
import torch

from dataset import points_to_relative_features
from model import HandwritingSeq2SeqModel
from tokenizer import CharTokenizer


def greedy_decode(
    model: HandwritingSeq2SeqModel,
    tokenizer: CharTokenizer,
    points: list[list[float]],
    device: torch.device,
    *,
    max_seq_len: int = 256,
    max_steps: int = 48,
    max_tgt_window: int = 96,
) -> str:
    """Greedy autoregressive decode aligned with OnnxHandwritingRecognizer."""
    feats = points_to_relative_features(points)
    if len(feats) == 0:
        return ""
    if len(feats) > max_seq_len:
        feats = feats[:max_seq_len]

    model.eval()
    src = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    src_lens = torch.tensor([src.shape[1]], dtype=torch.long, device=device)
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id
    bos_id = tokenizer.bos_id

    token_ids = [bos_id]
    for _ in range(max_steps):
        window = token_ids + [pad_id] * max(0, max_tgt_window - len(token_ids))
        window = window[:max_tgt_window]
        tgt_tensor = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(src, src_lens, tgt_tensor)
        step_idx = len(token_ids) - 1
        if step_idx >= logits.shape[1]:
            break
        row = logits[0, step_idx]
        next_id = int(row.argmax().item())
        if next_id in (eos_id, pad_id):
            break
        token_ids.append(next_id)

    return tokenizer.decode(token_ids[1:], rtl_aware=True)
