# -*- coding: utf-8 -*-
"""
Differentiable ODE-based neural simulator implementing Sompolinsky-style rate dynamics.

Dynamics: τ dh/dt = -h + W·φ(h) + I + η(t), where x(t) = φ(h(t)) = tanh(h(t))
"""

import torch
from typing import Optional, Tuple, Union


def simulate_rate_network(
    W: torch.Tensor,
    h0: torch.Tensor,
    T: int,
    dt: float,
    tau: float,
    I: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    return_h: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Simulates Sompolinsky-style rate dynamics using Euler integration.
    
    Args:
        W: Synaptic weight matrix [N, N], differentiable.
        h0: Initial condition [N].
        T: Number of timesteps.
        dt: Euler step size.
        tau: Membrane time constant.
        I: External input [N], defaults to zero.
        noise_std: Gaussian noise std (0.0 for deterministic).
        return_h: If True, also return internal state trajectory.
    
    Returns:
        x: Activity trace [T, N] where x[t] = tanh(h[t]).
        h: (Optional) Internal state trajectory [T, N].
    """
    N = W.shape[0]
    device = W.device
    dtype = W.dtype
    
    h = h0.to(device=device, dtype=dtype)
    
    if I is None:
        I = torch.zeros(N, device=device, dtype=dtype)
    else:
        I = I.to(device=device, dtype=dtype)
    
    h_trajectory = []
    
    for t in range(T):
        h_trajectory.append(h)
        x = torch.tanh(h)
        recurrent_input = W @ x
        dhdt = (-h + recurrent_input + I) / tau
        h = h + dt * dhdt
        
        if noise_std > 0.0:
            noise = torch.randn(N, device=device, dtype=dtype)
            h = h + (dt ** 0.5) * noise_std * noise
    
    h_trajectory = torch.stack(h_trajectory, dim=0)
    x_trajectory = torch.tanh(h_trajectory)
    
    if return_h:
        return x_trajectory, h_trajectory
    return x_trajectory


if __name__ == "__main__":
    # Quick verification that gradients flow correctly
    torch.manual_seed(42)
    N, T = 50, 100
    
    W = torch.randn(N, N) / (N ** 0.5)
    W.requires_grad = True
    h0 = torch.randn(N)
    
    x = simulate_rate_network(W, h0, T=T, dt=0.1, tau=1.0)
    loss = x.mean()
    loss.backward()
    
    assert W.grad is not None and W.grad.norm() > 0, "Gradient flow failed!"
    print(f"✓ Gradient flow verified: W.grad.norm() = {W.grad.norm():.4f}")
