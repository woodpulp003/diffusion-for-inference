# -*- coding: utf-8 -*-
"""
Unconditional DDPM for weight matrices.

Implements the full diffusion process: forward diffusion, reverse sampling,
and training loss computation.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Handle imports for both package and direct execution
try:
    from models.diffusion_prior.unet_prior import UNetPrior
    from models.diffusion_prior.noise_schedule import (
        make_beta_schedule,
        compute_alpha_cumprod,
        q_sample,
    )
except ImportError:
    # Direct execution
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from models.diffusion_prior.unet_prior import UNetPrior
    from models.diffusion_prior.noise_schedule import (
        make_beta_schedule,
        compute_alpha_cumprod,
        q_sample,
    )


class DiffusionPrior(nn.Module):
    """
    Unconditional DDPM for weight matrices W [N, N].
    
    Models p_θ(W) via diffusion process.
    """
    
    def __init__(
        self,
        unet: UNetPrior,
        T: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        
        self.unet = unet
        self.T = T
        
        # Noise schedule
        betas = make_beta_schedule(T, schedule, beta_start, beta_end)
        alphas, alpha_cumprod = compute_alpha_cumprod(betas)
        
        # Register as buffers (not parameters, but part of model state)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        
        # Precompute for sampling
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - alpha_cumprod))
        
        # For reverse process: posterior variance
        # q(W_{t-1} | W_t, W_0) variance
        posterior_variance = betas * (1.0 - torch.roll(alpha_cumprod, 1)) / (1.0 - alpha_cumprod)
        posterior_variance[0] = betas[0]  # First timestep
        self.register_buffer('posterior_variance', posterior_variance)
    
    def p_losses(self, W0: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute training loss: MSE between predicted and actual noise.
        
        Args:
            W0: [B, 1, N, N] clean weight matrices.
            t: [B] integer timesteps. If None, sampled uniformly.
        
        Returns:
            loss: Scalar MSE loss.
        """
        B = W0.shape[0]
        
        if t is None:
            t = torch.randint(0, self.T, (B,), device=W0.device)
        
        # Sample noise
        noise = torch.randn_like(W0)
        
        # Forward diffusion: add noise
        Wt = q_sample(W0, t, self.alpha_cumprod, noise)
        
        # Predict noise
        noise_pred = self.unet(Wt, t)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def p_sample(self, Wt: torch.Tensor, t: int) -> torch.Tensor:
        """
        Single reverse diffusion step.
        
        Args:
            Wt: [B, 1, N, N] noisy weight matrix at timestep t.
            t: Integer timestep.
        
        Returns:
            W_prev: [B, 1, N, N] denoised weight matrix at timestep t-1.
        """
        # Predict noise
        t_tensor = torch.full((Wt.shape[0],), t, device=Wt.device, dtype=torch.long)
        noise_pred = self.unet(Wt, t_tensor)
        
        # Compute coefficients
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        alpha_cumprod_prev = self.alpha_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=Wt.device)
        
        # Compute predicted W0 (x_0)
        pred_W0 = (Wt - self.sqrt_one_minus_alpha_cumprod[t] * noise_pred) / self.sqrt_alpha_cumprod[t]
        
        # Compute mean of q(W_{t-1} | W_t, W_0)
        pred_coeff1 = torch.sqrt(alpha_cumprod_prev) * self.betas[t] / (1.0 - alpha_cumprod_t)
        pred_coeff2 = torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
        mean = pred_coeff1 * pred_W0 + pred_coeff2 * Wt
        
        # Compute variance
        if t == 0:
            variance = 0.0
            W_prev = mean
        else:
            variance = self.posterior_variance[t]
            # Sample
            noise = torch.randn_like(Wt)
            W_prev = mean + torch.sqrt(variance) * noise
        
        return W_prev
    
    def p_sample_loop(self, shape: tuple, device: torch.device, verbose: bool = False) -> torch.Tensor:
        """
        Full reverse diffusion sampling loop.
        
        Args:
            shape: (B, 1, N, N) shape of output.
            device: Device to run on.
            verbose: Print progress if True.
        
        Returns:
            W0: [B, 1, N, N] sampled weight matrices.
        """
        B = shape[0]
        
        # Start from pure noise
        Wt = torch.randn(shape, device=device)
        
        # Reverse diffusion: T-1 -> 0
        if verbose:
            print(f"    Steps {self.T}->0:", end=' ', flush=True)
        
        for t in reversed(range(self.T)):
            Wt = self.p_sample(Wt, t)
            if verbose and (t % max(1, self.T // 10) == 0 or t == 0):
                print(f"{t}", end=' ', flush=True)
        
        if verbose:
            print("✓")
        
        return Wt
    
    def sample(self, num_samples: int, N: int, device: torch.device, verbose: bool = False) -> torch.Tensor:
        """
        Sample weight matrices from the prior.
        
        Args:
            num_samples: Number of samples to generate.
            N: Size of weight matrix (N x N).
            device: Device to run on.
            verbose: Print progress if True.
        
        Returns:
            W_samples: [num_samples, 1, N, N] sampled weight matrices.
        """
        self.eval()
        with torch.no_grad():
            shape = (num_samples, 1, N, N)
            W_samples = self.p_sample_loop(shape, device, verbose=verbose)
        return W_samples


if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unet = UNetPrior(in_channels=1, base_channels=32, channel_multipliers=(1, 2))
    diffusion = DiffusionPrior(unet, T=1000, schedule="cosine")
    
    B, N = 2, 50
    W0 = torch.randn(B, 1, N, N, device=device)
    
    # Test loss
    loss = diffusion.p_losses(W0)
    print(f"✓ Loss computation: {loss.item():.4f}")
    
    # Test sampling (small T for speed)
    diffusion.T = 10  # Quick test
    W_samples = diffusion.sample(2, N, device)
    print(f"✓ Sampling: {W_samples.shape}")
    print(f"  Mean: {W_samples.mean():.4f}, Std: {W_samples.std():.4f}")

