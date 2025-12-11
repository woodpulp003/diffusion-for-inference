# -*- coding: utf-8 -*-
"""
Differentiable rate-model simulator (Euler integration).

This mirrors simulator.rate_model.simulate_rate_network but keeps autograd
enabled for DPS refinement.
"""

import torch
from typing import Optional


def simulate_rate_network_diff(
    W: torch.Tensor,
    h0: torch.Tensor,
    T: int,
    dt: float,
    tau: float,
    I: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Simulate Sompolinsky-style rate dynamics with gradients w.r.t. W.

    Args:
        W: [N, N] weight matrix (requires_grad may be True).
        h0: [N] initial hidden state.
        T: number of timesteps.
        dt: Euler step size.
        tau: membrane time constant.
        I: optional external input [N].

    Returns:
        x_trajectory: [T, N] activities (tanh of hidden state).
    """
    N = W.shape[0]
    device = W.device
    dtype = W.dtype

    h = h0.to(device=device, dtype=dtype)
    if I is None:
        I = torch.zeros(N, device=device, dtype=dtype)
    else:
        I = I.to(device=device, dtype=dtype)

    xs = []
    for _ in range(T):
        x = torch.tanh(h)  # [N]
        xs.append(x)
        recurrent = torch.matmul(W, x)  # [N]
        dhdt = (-h + recurrent + I) / tau
        h = h + dt * dhdt

    return torch.stack(xs, dim=0)  # [T, N]


__all__ = ["simulate_rate_network_diff"]

