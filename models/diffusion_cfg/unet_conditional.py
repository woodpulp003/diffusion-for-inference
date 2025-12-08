# -*- coding: utf-8 -*-
"""
Conditional UNet with cross-attention for diffusion over weight matrices.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_prior.unet_prior import SinusoidalPositionEmbeddings, TimeMLP


class FiLMBlock(nn.Module):
    """Conv block with FiLM conditioning from time and pooled context."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, cond_dim: int):
        super().__init__()
        self.residual = in_ch == out_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim + cond_dim, out_ch * 2),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)

        params = self.film(torch.cat([time_emb, cond_vec], dim=-1))
        scale, shift = params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + scale) + shift
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        if self.residual:
            h = h + x
        return h


class CrossAttention2d(nn.Module):
    """Cross-attention from spatial features to conditioning tokens."""

    def __init__(self, channels: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        self.query_proj = nn.Linear(channels, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=cond_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor, cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            cond_tokens: [B, L, cond_dim]
            cond_mask: optional [B, L] padding mask (True = pad)
        """
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]
        q = self.query_proj(seq)  # [B, HW, cond_dim]
        attn_out, _ = self.attn(q, cond_tokens, cond_tokens, key_padding_mask=cond_mask)
        out = self.out_proj(attn_out)  # [B, HW, C]
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class UNetConditional(nn.Module):
    """
    UNet conditioned on activity encoder outputs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4),
        time_emb_dim: int = 128,
        cond_dim: int = 128,
        num_attn_heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.cond_dim = cond_dim

        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = TimeMLP(time_emb_dim, base_channels * 4)
        self.cond_proj = nn.Linear(cond_dim, base_channels * 4)

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for mult in channel_multipliers:
            out_channels = base_channels * mult
            self.down_blocks.append(FiLMBlock(channels, out_channels, base_channels * 4, cond_dim))
            channels = out_channels

        self.mid_block = FiLMBlock(channels, channels, base_channels * 4, cond_dim)
        self.mid_attn = CrossAttention2d(channels, cond_dim, num_heads=num_attn_heads)

        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        prev_channels = channels
        for mult in reversed(channel_multipliers):
            out_ch = base_channels * mult
            self.up_convs.append(nn.Conv2d(prev_channels + out_ch, out_ch, 3, padding=1))
            self.up_blocks.append(FiLMBlock(out_ch, out_ch, base_channels * 4, cond_dim))
            prev_channels = out_ch

        self.conv_out = nn.Conv2d(base_channels, in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: Optional[torch.Tensor],
        cond_pooled: Optional[torch.Tensor],
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, 1, N, N] noisy weights.
            t: [B] integer timesteps.
            cond_tokens: [B, L, cond_dim] conditioning tokens (can be None or zeros for null).
            cond_pooled: [B, cond_dim] pooled conditioning vector.
            cond_mask: optional [B, L] bool padding mask.
        """
        B = x.shape[0]
        if cond_tokens is None:
            cond_tokens = torch.zeros(B, 1, self.cond_dim, device=x.device, dtype=x.dtype)
        if cond_pooled is None:
            cond_pooled = torch.zeros(B, self.cond_dim, device=x.device, dtype=x.dtype)

        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)
        cond_vec = self.cond_proj(cond_pooled)
        time_cond = t_emb + cond_vec

        h = self.conv_in(x)
        skip_connections = []
        for down in self.down_blocks:
            h = down(h, time_cond, cond_pooled)
            skip_connections.append(h)
            h = F.avg_pool2d(h, 2)

        h = self.mid_block(h, time_cond, cond_pooled)
        h = h + self.mid_attn(h, cond_tokens, cond_mask)

        for up_conv, up_block in zip(self.up_convs, self.up_blocks):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            if skip_connections:
                skip = skip_connections.pop()
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
            h = up_conv(h)
            h = up_block(h, time_cond, cond_pooled)

        return self.conv_out(h)



