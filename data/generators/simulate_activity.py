# -*- coding: utf-8 -*-
"""
Activity Simulator Wrapper for Dataset Generation.
Generates multiple activity trials from a weight matrix.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from typing import Optional, List

from simulator.rate_model import simulate_rate_network


def generate_activity_trials(
    W: torch.Tensor,
    num_trials: int,
    T: int,
    dt: float,
    tau: float,
    I: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    h0_std: float = 1.0,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Generate multiple activity trials from a weight matrix.
    
    Args:
        W: Weight matrix [N, N].
        num_trials: Number of trials.
        T: Timesteps per trial.
        dt: Euler step size.
        tau: Membrane time constant.
        I: External input [N].
        noise_std: Process noise std.
        h0_std: Initial condition std (h0 ~ Normal(0, h0_std)).
        seed: Random seed.
    
    Returns:
        List of activity tensors, each [T, N].
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    N = W.shape[0]
    device = W.device
    dtype = W.dtype
    
    activities = []
    for _ in range(num_trials):
        h0 = torch.randn(N, device=device, dtype=dtype) * h0_std
        activity = simulate_rate_network(W, h0, T, dt, tau, I, noise_std)
        activities.append(activity)
    
    return activities


def generate_activity_trials_batched(
    W: torch.Tensor,
    num_trials: int,
    T: int,
    dt: float,
    tau: float,
    I: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    h0_std: float = 1.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Same as generate_activity_trials but returns [num_trials, T, N] tensor."""
    activities = generate_activity_trials(W, num_trials, T, dt, tau, I, noise_std, h0_std, seed)
    return torch.stack(activities, dim=0)


if __name__ == "__main__":
    # Quick verification
    torch.manual_seed(42)
    N, T, num_trials = 30, 50, 5
    
    W = torch.randn(N, N) / (N ** 0.5)
    activities = generate_activity_trials(W, num_trials, T, dt=0.1, tau=1.0, seed=42)
    
    print(f"✓ Generated {len(activities)} trials, each shape: {activities[0].shape}")
    
    # Verify trials differ (different h0)
    diff = (activities[0] - activities[1]).abs().mean()
    print(f"✓ Trial variability confirmed: mean diff = {diff:.4f}")
