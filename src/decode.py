from __future__ import annotations

import torch

from dataset import points_to_relative_features
from decode_quality import is_template_collapse
from tokenizer import CharTokenizer

# Penalize logits for chars that extend known bad partial decodes (greedy only).
_TEMPLATE_CHAR_PENALTY = 5.0


def greedy_decode(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    points: list[list[float]],
    device: torch.device,
    *,
    max_seq_len: int = 256,
    max_steps: int = 128,
    max_tgt_window: int = 128,
    repetition_penalty: float = 2.0,
    no_repeat_window: int = 12,
    mode: str | None = None,
) -> str:
    """Greedy autoregressive decode (align max_steps with ONNX tgt window)."""
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
    prefix_id = tokenizer.mode_prefix_id(mode)
    if prefix_id is not None:
        token_ids.append(prefix_id)

    start = 1 + (1 if prefix_id is not None else 0)

    def _apply_template_penalty(row: torch.Tensor) -> None:
        partial = tokenizer.decode(token_ids[start:], rtl_aware=False, mode=mode)
        if not is_template_collapse(partial, mode=mode):
            return
        for ch in set(partial.lower() + "theandwanther.\\frac{12}x-{)"):
            tid = tokenizer.stoi.get(ch)
            if tid is not None and 0 <= tid < row.shape[0]:
                row[tid] = row[tid] / _TEMPLATE_CHAR_PENALTY

    for _ in range(max_steps):
        window = token_ids + [pad_id] * max(0, max_tgt_window - len(token_ids))
        window = window[:max_tgt_window]
        tgt_tensor = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(src, src_lens, tgt_tensor)
        step_idx = len(token_ids) - 1
        if step_idx >= logits.shape[1]:
            break
        row = logits[0, step_idx].clone()
        if repetition_penalty > 1.0 and len(token_ids) >= 2:
            recent = token_ids[-no_repeat_window:]
            for tid in set(recent):
                if 0 <= tid < row.shape[0]:
                    count = recent.count(tid)
                    row[tid] = row[tid] / (repetition_penalty**count)
        if len(token_ids) >= 4 and len(set(token_ids[-4:])) == 1:
            stuck = token_ids[-1]
            if 0 <= stuck < row.shape[0]:
                row[stuck] = row[stuck] / max(repetition_penalty**3, 8.0)
        _apply_template_penalty(row)
        next_id = int(row.argmax().item())
        if next_id in (eos_id, pad_id):
            break
        token_ids.append(next_id)

    return tokenizer.decode(token_ids[start:], rtl_aware=True, mode=mode)
