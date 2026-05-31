from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class StrokeTransformerSeq2Seq(nn.Module):
    """
    Transformer encoder + decoder for online handwriting recognition.

    Forward signature matches HandwritingSeq2SeqModel for ONNX export:
      src [B,T,3], src_lens [B], tgt_inp [B,U] -> logits [B,U,V]
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
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

    def _encode(self, src: torch.Tensor, src_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        t = src.size(1)
        src_key_padding = torch.arange(t, device=src.device).unsqueeze(0) >= src_lens.unsqueeze(1)
        memory = self.encoder(x, src_key_padding_mask=src_key_padding)
        return memory, src_key_padding

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        memory, src_key_padding = self._encode(src, src_lens)
        u = tgt_inp.size(1)
        dec = self.token_embed(tgt_inp) * math.sqrt(self.d_model)
        dec = self.pos_enc(dec)
        causal = torch.triu(torch.ones(u, u, device=src.device, dtype=torch.bool), diagonal=1)
        out = self.decoder(
            dec,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=src_key_padding,
        )
        return self.out_proj(out)
