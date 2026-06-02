from __future__ import annotations

import math

import torch
from torch import nn

from model_transformer import PositionalEncoding


class ConvStem(nn.Module):
    """Light temporal convolution stem to denoise/downsample stroke sequence."""

    def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        hidden = max(64, d_model // 2)
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # src: [B, T, C] -> [B, C, T]
        x = src.transpose(1, 2)
        x = self.net(x)
        # Back to [B, T', C]
        x = x.transpose(1, 2)
        # Only one stride-2 layer in stem.
        out_lens = torch.clamp((src_lens + 1) // 2, min=1)
        return x, out_lens


class HybridHandwritingModel(nn.Module):
    """
    Hybrid model:
      - Conv+Transformer encoder
      - CTC head (alignment-friendly)
      - Autoregressive decoder head (language modeling)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        vocab_size: int,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.stem = ConvStem(input_dim=input_dim, d_model=d_model, dropout=dropout)
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.token_embed = nn.Embedding(vocab_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )
        self.ctc_proj = nn.Linear(d_model, vocab_size)

    def encode(self, src: torch.Tensor, src_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, enc_lens = self.stem(src, src_lens)
        x = self.pos_enc(x)
        t = x.size(1)
        src_key_padding = torch.arange(t, device=src.device).unsqueeze(0) >= enc_lens.unsqueeze(1)
        memory = self.encoder(x, src_key_padding_mask=src_key_padding)
        return memory, enc_lens

    def ctc_logits(self, src: torch.Tensor, src_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory, enc_lens = self.encode(src, src_lens)
        return self.ctc_proj(memory), enc_lens

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        # Keep same forward contract as existing models for decode/export compatibility.
        memory, enc_lens = self.encode(src, src_lens)
        u = tgt_inp.size(1)
        dec = self.token_embed(tgt_inp) * math.sqrt(self.d_model)
        dec = self.pos_enc(dec)
        causal = torch.triu(torch.ones(u, u, device=src.device, dtype=torch.bool), diagonal=1)
        mem_t = memory.size(1)
        mem_padding = torch.arange(mem_t, device=src.device).unsqueeze(0) >= enc_lens.unsqueeze(1)
        out = self.decoder(
            dec,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=mem_padding,
        )
        return self.out_proj(out)
