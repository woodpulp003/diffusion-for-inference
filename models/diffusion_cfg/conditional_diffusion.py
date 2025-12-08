# -*- coding: utf-8 -*-
"""
Conditional diffusion with classifier-free guidance over weight matrices.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_prior.noise_schedule import make_beta_schedule, compute_alpha_cumprod, q_sample


def extract(tensor: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Extract values for batch of indices and reshape for broadcasting."""
    out = tensor.gather(-1, timesteps)
    return out.view(-1, *([1] * (len(shape) - 1)))


class ConditionalDiffusion(nn.Module):
    """
    Conditional DDPM with classifier-free guidance.
    """

    def __init__(
        self,
        unet: nn.Module,
        activity_encoder: nn.Module,
        T: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cond_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.unet = unet
        self.activity_encoder = activity_encoder
        self.T = T
        self.cond_drop_prob = cond_drop_prob

        betas = make_beta_schedule(T, schedule, beta_start, beta_end)
        alphas, alpha_cumprod = compute_alpha_cumprod(betas)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - torch.roll(alpha_cumprod, 1)) / (1.0 - alpha_cumprod)
        posterior_variance[0] = betas[0]
        self.register_buffer("posterior_variance", posterior_variance)

    def forward(self, W0: torch.Tensor, A: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.p_losses(W0, A, t)

    def p_losses(self, W0: torch.Tensor, A: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = W0.shape[0]
        device = W0.device
        if t is None:
            t = torch.randint(0, self.T, (B,), device=device)

        noise = torch.randn_like(W0)
        Wt = q_sample(W0, t, self.alpha_cumprod, noise)

        cond_tokens, cond_pooled = self.activity_encoder(A)

        drop_mask = (torch.rand(B, device=device) < self.cond_drop_prob).float().view(B, 1, 1)
        cond_tokens_dropped = cond_tokens * (1.0 - drop_mask)
        cond_pooled_dropped = cond_pooled * (1.0 - drop_mask.view(B, 1))

        noise_pred = self.unet(Wt, t, cond_tokens_dropped, cond_pooled_dropped)
        return F.mse_loss(noise_pred, noise)

    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: torch.Tensor,
        cond_pooled: torch.Tensor,
        cond_mask: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alpha_cumprod, t, x.shape)
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x.shape)

        eps_cond = self.unet(x, t, cond_tokens, cond_pooled, cond_mask)
        if guidance_scale > 0.0:
            zeros_tokens = torch.zeros_like(cond_tokens)
            zeros_pooled = torch.zeros_like(cond_pooled)
            eps_uncond = self.unet(x, t, zeros_tokens, zeros_pooled, cond_mask)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        model_mean = sqrt_recip_alpha_t * (x - betas_t / sqrt_one_minus_alpha_cumprod_t * eps)

        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(x.shape[0], *([1] * (len(x.shape) - 1)))
        posterior_var_t = extract(self.posterior_variance, t, x.shape)
        return model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: Tuple[int, int, int, int],
        cond_tokens: torch.Tensor,
        cond_pooled: torch.Tensor,
        cond_mask: Optional[torch.Tensor],
        guidance_scale: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = cond_tokens.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, cond_tokens, cond_pooled, cond_mask, guidance_scale)
        return img

    @torch.no_grad()
    def sample(
        self,
        cond_tokens: torch.Tensor,
        cond_pooled: torch.Tensor,
        N: int,
        guidance_scale: float = 0.0,
        cond_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or cond_tokens.device
        b = cond_tokens.shape[0]
        shape = (b, 1, N, N)
        return self.p_sample_loop(shape, cond_tokens, cond_pooled, cond_mask, guidance_scale, device)



