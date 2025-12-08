# -*- coding: utf-8 -*-
"""
UNet architecture for unconditional diffusion prior over weight matrices.

Treats weight matrices W [N, N] as single-channel images [1, N, N].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time-step embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [B] integer timesteps
        
        Returns:
            embeddings: [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class TimeMLP(nn.Module):
    """Small MLP to process time embeddings."""
    
    def __init__(self, time_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(time_emb)


class ConvBlock(nn.Module):
    """Convolutional block with time conditioning via FiLM."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, residual: bool = True):
        super().__init__()
        self.residual = residual and (in_channels == out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # FiLM: Feature-wise Linear Modulation (scale and shift)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),  # [scale, shift]
        )
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_emb_dim]
        
        Returns:
            [B, C_out, H, W]
        """
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        
        # FiLM conditioning
        time_params = self.time_mlp(time_emb)  # [B, 2*C]
        scale, shift = time_params.chunk(2, dim=1)  # Each [B, C]
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        h = h * (1 + scale) + shift
        
        h = F.silu(h)
        
        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Residual connection
        if self.residual:
            h = h + x
        
        return h


class UNetPrior(nn.Module):
    """
    UNet architecture for unconditional diffusion prior over weight matrices.
    
    Input: [B, 1, N, N] (weight matrix as single-channel image)
    Output: [B, 1, N, N] (predicted noise)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4),
        time_emb_dim: int = 128,
        num_timesteps: int = 1000,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        
        # Time embeddings
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = TimeMLP(time_emb_dim, base_channels * 4)
        
        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for mult in channel_multipliers:
            out_channels = base_channels * mult
            self.down_blocks.append(
                ConvBlock(channels, out_channels, base_channels * 4)
            )
            channels = out_channels
        
        # Mid block
        self.mid_block = ConvBlock(channels, channels, base_channels * 4)
        
        # Up blocks (reverse order with skip connections)
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        prev_channels = channels
        for mult in reversed(channel_multipliers):
            out_ch = base_channels * mult
            # Conv to handle skip connection concatenation
            self.up_convs.append(nn.Conv2d(prev_channels + out_ch, out_ch, 3, padding=1))
            self.up_blocks.append(ConvBlock(out_ch, out_ch, base_channels * 4))
            prev_channels = out_ch
        
        # Final output
        self.conv_out = nn.Conv2d(base_channels, in_channels, 1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, N, N] noisy weight matrix
            t: [B] integer timesteps
        
        Returns:
            noise_pred: [B, 1, N, N] predicted noise
        """
        # Time embedding
        t_emb = self.time_emb(t)  # [B, time_emb_dim]
        t_emb = self.time_mlp(t_emb)  # [B, hidden_dim]
        
        # Initial conv
        h = self.conv_in(x)
        
        # Down path with skip connections
        skip_connections = []
        for down_block in self.down_blocks:
            h = down_block(h, t_emb)
            skip_connections.append(h)
            h = F.avg_pool2d(h, 2)  # Downsample
        
        # Mid block
        h = self.mid_block(h, t_emb)
        
        # Up path with skip connections
        for i, (up_conv, up_block) in enumerate(zip(self.up_convs, self.up_blocks)):
            # Upsample
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            
            # Concatenate skip connection
            if skip_connections:
                skip = skip_connections.pop()
                # Handle size mismatch (due to odd dimensions)
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
            
            # Conv and block
            h = up_conv(h)
            h = up_block(h, t_emb)
        
        # Final output
        noise_pred = self.conv_out(h)
        
        return noise_pred


if __name__ == "__main__":
    # Quick test
    model = UNetPrior(in_channels=1, base_channels=32, channel_multipliers=(1, 2))
    
    B, N = 4, 50
    x = torch.randn(B, 1, N, N)
    t = torch.randint(0, 1000, (B,))
    
    noise_pred = model(x, t)
    print(f"✓ UNet test passed")
    print(f"  Input: {x.shape}, Timesteps: {t.shape}, Output: {noise_pred.shape}")

