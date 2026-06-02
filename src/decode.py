from __future__ import annotations

import torch
import torch.nn.functional as F

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


def _ctc_collapse_ids(ids: list[int], blank_id: int) -> list[int]:
    out: list[int] = []
    prev = None
    for i in ids:
        if i == blank_id:
            prev = None
            continue
        if prev == i:
            continue
        out.append(i)
        prev = i
    return out


def ctc_greedy_decode(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    points: list[list[float]],
    device: torch.device,
    *,
    max_seq_len: int = 256,
    mode: str | None = None,
) -> str:
    feats = points_to_relative_features(points)
    if len(feats) == 0:
        return ""
    if len(feats) > max_seq_len:
        feats = feats[:max_seq_len]
    if not hasattr(model, "ctc_logits"):
        return greedy_decode(
            model,
            tokenizer,
            points,
            device,
            max_seq_len=max_seq_len,
            mode=mode,
        )

    model.eval()
    src = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    src_lens = torch.tensor([src.shape[1]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, out_lens = model.ctc_logits(src, src_lens)
    t = int(out_lens[0].item())
    frame_ids = logits[0, :t].argmax(dim=-1).tolist()
    seq_ids = _ctc_collapse_ids(frame_ids, tokenizer.blank_id)
    return tokenizer.decode(seq_ids, rtl_aware=True, mode=mode)


def _ar_sequence_logprob(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    points: list[list[float]],
    text: str,
    device: torch.device,
    *,
    max_seq_len: int = 256,
    max_tgt_window: int = 128,
    mode: str | None = None,
) -> float:
    feats = points_to_relative_features(points)
    if len(feats) == 0:
        return float("-inf")
    if len(feats) > max_seq_len:
        feats = feats[:max_seq_len]
    ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        mode=mode,
        add_mode_prefix=bool(tokenizer.mode_prefix_id(mode) is not None),
    )
    if len(ids) < 2:
        return float("-inf")
    ids = ids[:max_tgt_window]
    dec_in = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    src = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    src_lens = torch.tensor([src.shape[1]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(src, src_lens, dec_in)
        logp = F.log_softmax(logits, dim=-1)
    gathered = logp.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
    return float(gathered.mean().item())


def _ctc_sequence_logprob(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    points: list[list[float]],
    text: str,
    device: torch.device,
    *,
    max_seq_len: int = 256,
    mode: str | None = None,
) -> float:
    if not hasattr(model, "ctc_logits"):
        return float("-inf")
    feats = points_to_relative_features(points)
    if len(feats) == 0:
        return float("-inf")
    if len(feats) > max_seq_len:
        feats = feats[:max_seq_len]
    ids = tokenizer.encode(text, add_special_tokens=False, mode=mode, add_mode_prefix=False)
    if len(ids) == 0:
        return float("-inf")
    src = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    src_lens = torch.tensor([src.shape[1]], dtype=torch.long, device=device)
    tgt = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    tgt_lens = torch.tensor([len(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, out_lens = model.ctc_logits(src, src_lens)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        ctc = torch.nn.CTCLoss(blank=tokenizer.blank_id, reduction="mean", zero_infinity=True)
        loss = ctc(log_probs, tgt, out_lens, tgt_lens)
    return float(-loss.item())


def joint_decode(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    points: list[list[float]],
    device: torch.device,
    *,
    max_seq_len: int = 256,
    max_steps: int = 128,
    max_tgt_window: int = 128,
    repetition_penalty: float = 2.0,
    mode: str | None = None,
    ctc_weight: float = 0.5,
    ar_weight: float = 0.5,
) -> str:
    # Candidate 1: AR greedy
    ar_pred = greedy_decode(
        model,
        tokenizer,
        points,
        device,
        max_seq_len=max_seq_len,
        max_steps=max_steps,
        max_tgt_window=max_tgt_window,
        repetition_penalty=repetition_penalty,
        mode=mode,
    )
    if not hasattr(model, "ctc_logits"):
        return ar_pred
    # Candidate 2: CTC greedy
    ctc_pred = ctc_greedy_decode(
        model,
        tokenizer,
        points,
        device,
        max_seq_len=max_seq_len,
        mode=mode,
    )
    candidates = []
    for pred in {ar_pred, ctc_pred}:
        ar_lp = _ar_sequence_logprob(
            model,
            tokenizer,
            points,
            pred,
            device,
            max_seq_len=max_seq_len,
            max_tgt_window=max_tgt_window,
            mode=mode,
        )
        ctc_lp = _ctc_sequence_logprob(
            model,
            tokenizer,
            points,
            pred,
            device,
            max_seq_len=max_seq_len,
            mode=mode,
        )
        score = ar_weight * ar_lp + ctc_weight * ctc_lp
        candidates.append((score, pred))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
