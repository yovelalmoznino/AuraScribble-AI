from __future__ import annotations

from typing import Any

import torch.nn as nn

from model import HandwritingSeq2SeqModel
from model_transformer import StrokeTransformerSeq2Seq


def build_model(config: dict[str, Any], vocab_size: int) -> nn.Module:
    model_type = (config.get("model_type") or "transformer").lower()
    input_dim = int(config.get("input_dim", 3))
    dropout = float(config.get("dropout", 0.15))

    if model_type == "lstm":
        return HandwritingSeq2SeqModel(
            input_dim=input_dim,
            hidden=int(config.get("hidden_dim", 256)),
            layers=int(config.get("num_layers", 3)),
            dropout=dropout,
            vocab_size=vocab_size,
        )

    return StrokeTransformerSeq2Seq(
        input_dim=input_dim,
        d_model=int(config.get("d_model", config.get("hidden_dim", 256))),
        nhead=int(config.get("nhead", 8)),
        num_encoder_layers=int(config.get("encoder_layers", config.get("num_layers", 4))),
        num_decoder_layers=int(config.get("decoder_layers", 2)),
        dropout=dropout,
        vocab_size=vocab_size,
        max_len=int(config.get("max_seq_len", 512)),
    )
