# -*- coding: utf-8 -*-
"""
Noise schedule utilities for DDPM.

Implements beta schedules, alpha computations, and forward diffusion process.
"""

import torch
import numpy as np
from typing import Optional


def make_beta_schedule(T: int, schedule: str = "cosine", beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Create beta schedule for diffusion timesteps.
    
    Args:
        T: Number of diffusion timesteps.
        schedule: "linear" or "cosine".
        beta_start: Starting beta value.
        beta_end: Ending beta value.
    
    Returns:
        betas: [T] tensor of beta values.
    """
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T)
    elif schedule == "cosine":
        # Cosine schedule (better for images)
        s = 0.008  # Small offset
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return betas


def compute_alpha_cumprod(betas: torch.Tensor):
    """
    Compute alphas and cumulative product of alphas.
    
    Args:
        betas: [T] beta values.
    
    Returns:
        alphas: [T] alpha values (1 - beta).
        alpha_cumprod: [T] cumulative product of alphas.
    """
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alpha_cumprod


def q_sample(
    W0: torch.Tensor,
    t: torch.Tensor,
    alpha_cumprod: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward diffusion process: add noise to clean weight matrix.
    
    q(W_t | W_0) = N(W_t; sqrt(alpha_bar_t) * W_0, (1 - alpha_bar_t) * I)
    
    Args:
        W0: [B, 1, N, N] clean weight matrices.
        t: [B] integer timesteps (0 to T-1).
        alpha_cumprod: [T] cumulative product of alphas.
        noise: Optional [B, 1, N, N] noise tensor. If None, sampled.
    
    Returns:
        Wt: [B, 1, N, N] noisy weight matrices at timestep t.
    """
    if noise is None:
        noise = torch.randn_like(W0)
    
    # Extract alpha_bar_t for each sample in batch
    # t is [B], alpha_cumprod is [T]
    # We need to index alpha_cumprod with t
    sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod[t])  # [B]
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod[t])  # [B]
    
    # Reshape for broadcasting: [B, 1, 1, 1]
    sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1)
    
    # Forward diffusion
    Wt = sqrt_alpha_cumprod_t * W0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    return Wt


if __name__ == "__main__":
    # Quick test
    T = 1000
    betas = make_beta_schedule(T, schedule="cosine")
    alphas, alpha_cumprod = compute_alpha_cumprod(betas)
    
    print(f"✓ Noise schedule test passed")
    print(f"  T: {T}, betas shape: {betas.shape}")
    print(f"  beta range: [{betas.min():.6f}, {betas.max():.6f}]")
    print(f"  alpha_cumprod range: [{alpha_cumprod.min():.6f}, {alpha_cumprod.max():.6f}]")
    
    # Test q_sample
    B, N = 2, 50
    W0 = torch.randn(B, 1, N, N)
    t = torch.randint(0, T, (B,))
    Wt = q_sample(W0, t, alpha_cumprod)
    
    print(f"  q_sample: W0 {W0.shape} -> Wt {Wt.shape}")

