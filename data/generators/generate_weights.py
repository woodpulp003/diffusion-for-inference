# -*- coding: utf-8 -*-
"""
Weight Matrix Generator for Sompolinsky-style Rate Networks.
Generates W with W_ij ~ Normal(0, g^2/N), giving spectral radius ≈ g.
"""

import torch
import math
from typing import Optional


def sample_weight_matrix(
    N: int,
    g: float,
    ei_ratio: Optional[float] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample a random synaptic weight matrix.
    
    Args:
        N: Number of neurons.
        g: Gain parameter (spectral radius ≈ g, edge of chaos at g ≈ 1).
        ei_ratio: E/I ratio for Dale's law (None for unconstrained).
        seed: Random seed.
    
    Returns:
        W: Weight matrix [N, N], float32.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    scale = g / math.sqrt(N)
    
    if ei_ratio is None:
        W = torch.randn(N, N, dtype=torch.float32) * scale
    else:
        if not (0.0 < ei_ratio < 1.0):
            raise ValueError(f"ei_ratio must be in (0, 1), got {ei_ratio}")
        
        n_exc = math.ceil(N * ei_ratio)
        W_magnitudes = torch.abs(torch.randn(N, N, dtype=torch.float32)) * scale
        sign_mask = torch.ones(N, N, dtype=torch.float32)
        sign_mask[:, n_exc:] = -1.0
        W = W_magnitudes * sign_mask
    
    return W


def get_spectral_radius(W: torch.Tensor) -> float:
    """Compute spectral radius (largest eigenvalue magnitude)."""
    eigenvalues = torch.linalg.eigvals(W)
    return torch.abs(eigenvalues).max().item()


def sample_weight_matrix_batch(
    batch_size: int,
    N: int,
    g: float,
    ei_ratio: Optional[float] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample a batch of weight matrices. Returns [batch_size, N, N]."""
    if seed is not None:
        torch.manual_seed(seed)
    
    matrices = [sample_weight_matrix(N, g, ei_ratio) for _ in range(batch_size)]
    return torch.stack(matrices, dim=0)


if __name__ == "__main__":
    # Quick verification
    W = sample_weight_matrix(N=100, g=1.5, seed=42)
    print(f"✓ W shape: {W.shape}, std: {W.std():.4f}, spectral radius: {get_spectral_radius(W):.4f}")
    
    W_ei = sample_weight_matrix(N=100, g=1.2, ei_ratio=0.8, seed=42)
    n_exc = math.ceil(100 * 0.8)
    assert (W_ei[:, :n_exc] >= 0).all() and (W_ei[:, n_exc:] <= 0).all()
    print(f"✓ E/I Dale's law verified")
