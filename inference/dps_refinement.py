# -*- coding: utf-8 -*-
"""
DPS posterior refinement over weight matrices.

Given a warm-start W_cfg and observed activity A_obs, refine W via gradient
descent on simulator loss + prior regularization.
"""

from typing import Optional, Dict
import torch
from torch.optim import Adam

from models.simulator_diff import simulate_rate_network_diff


def refine_weight_matrix(
    W_cfg: torch.Tensor,
    A_obs: torch.Tensor,
    metadata: Dict,
    num_steps: int = 300,
    lr: float = 1e-3,
    lambda_prior: float = 0.01,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Refine weight matrix with DPS.

    Args:
        W_cfg: [N, N] warm-start weights (tensor).
        A_obs: [T, N_obs] observed activity (tensor).
        metadata: dict with keys 'T', 'dt', 'tau', optionally 'N_obs'.
        num_steps: refinement steps.
        lr: Adam learning rate.
        lambda_prior: prior penalty weight.
        device: device string.

    Returns:
        W_refined: [N, N] tensor on CPU.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    W = W_cfg.to(device).clone().detach().requires_grad_(True)
    A_obs = A_obs.to(device)

    T = int(metadata.get("T", A_obs.shape[0]))
    dt = float(metadata.get("dt", 0.1))
    tau = float(metadata.get("tau", 1.0))
    N_obs = int(metadata.get("N_obs", A_obs.shape[1]))

    optimizer = Adam([W], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        h0 = torch.randn(W.shape[0], device=device, dtype=W.dtype)
        A_sim = simulate_rate_network_diff(W, h0, T, dt, tau)  # [T, N]
        A_sim_obs = A_sim[:, :N_obs]

        loss_fit = ((A_sim_obs - A_obs) ** 2).mean()
        loss_prior = ((W - W_cfg.to(device)) ** 2).mean()
        loss = loss_fit + lambda_prior * loss_prior

        loss.backward()
        optimizer.step()

    return W.detach().cpu()


__all__ = ["refine_weight_matrix"]

