# -*- coding: utf-8 -*-
"""
Transformer encoder for observed activity sequences.
Encodes A_obs [B, T, N_obs] into token embeddings for conditioning diffusion.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_prior.unet_prior import SinusoidalPositionEmbeddings


class ActivityEncoder(nn.Module):
    """
    Encode observed activity sequences with a Transformer encoder.
    Returns per-time-step tokens plus a pooled summary vector.
    """

    def __init__(
        self,
        n_obs: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(n_obs, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = SinusoidalPositionEmbeddings(d_model)
        self.pooling = pooling

    def forward(self, A: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            A: [B, T, N_obs] activity matrix.
            padding_mask: Optional bool mask [B, T] where True indicates padding.
        Returns:
            tokens: [B, T, d_model] encoded tokens.
            pooled: [B, d_model] pooled summary.
        """
        B, T, _ = A.shape
        tokens = self.input_proj(A)  # [B, T, d_model]

        # Positional/time encoding
        time_ids = torch.arange(T, device=A.device)
        pos = self.pos_emb(time_ids).unsqueeze(0)  # [1, T, d_model]
        tokens = tokens + pos

        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)

        if self.pooling == "cls":
            pooled = tokens[:, 0]
        else:
            if padding_mask is not None:
                valid = (~padding_mask).float().unsqueeze(-1)
                pooled = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            else:
                pooled = tokens.mean(dim=1)

        return tokens, pooled


def build_activity_encoder_from_args(args) -> ActivityEncoder:
    return ActivityEncoder(
        n_obs=args.n_obs,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling="mean",
    )



