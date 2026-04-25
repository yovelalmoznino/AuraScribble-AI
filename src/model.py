from __future__ import annotations

import torch
from torch import nn


class HandwritingSeq2SeqModel(nn.Module):
    """
    BiLSTM encoder + attention GRU decoder for online handwriting recognition.

    Inputs:
      - src: [B, T, 3] where channels = (dx, dy, pen_state)
      - tgt_inp: [B, U] teacher-forcing token sequence (typically <bos> + tokens)
    Output:
      - logits: [B, U, V]
    """

    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float, vocab_size: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.token_embed = nn.Embedding(vocab_size, hidden)
        self.decoder = nn.GRU(
            input_size=hidden + hidden * 2,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden + hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, vocab_size),
        )
        self.hidden_bridge = nn.Linear(hidden * 2, hidden)

    def _attend(self, decoder_state: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # decoder_state: [B, H], enc_out: [B, T, 2H], src_mask: [B, T]
        scores = torch.einsum("bh,bth->bt", decoder_state, self.hidden_bridge(enc_out))
        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # [B, T]
        context = torch.einsum("bt,bth->bh", attn, enc_out)  # [B, 2H]
        return context

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        # src: [B,T,3]
        enc_out, _ = self.encoder(src)  # [B,T,2H]
        B, T, _ = enc_out.shape
        U = tgt_inp.shape[1]
        src_mask = torch.arange(T, device=src.device).unsqueeze(0) < src_lens.unsqueeze(1)

        # Initialize decoder state from first encoder frame (or zeros fallback).
        init_ctx = enc_out[:, 0]
        dec_h = torch.tanh(self.hidden_bridge(init_ctx)).unsqueeze(0)  # [1,B,H]

        logits_steps = []
        for u in range(U):
            tok_emb = self.token_embed(tgt_inp[:, u])  # [B,H]
            ctx = self._attend(dec_h.squeeze(0), enc_out, src_mask)  # [B,2H]
            dec_in = torch.cat([tok_emb, ctx], dim=-1).unsqueeze(1)  # [B,1,H+2H]
            dec_out, dec_h = self.decoder(dec_in, dec_h)  # dec_out [B,1,H]
            step = torch.cat([dec_out.squeeze(1), ctx], dim=-1)  # [B,H+2H]
            logits_steps.append(self.out_proj(step))  # [B,V]
        return torch.stack(logits_steps, dim=1)  # [B,U,V]
