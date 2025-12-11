# -*- coding: utf-8 -*-
"""
Evaluation script for CFG conditional diffusion: Recover W from A_obs (v2 with guidance_scale=3.0).

Tests how well the conditional diffusion model can recover the ground-truth
weight matrix W when given the corresponding observed activity A_obs.

This version uses guidance_scale=3.0 and saves to reconstruction_results_v2/.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
from typing import Optional, List, Dict
import random
import json

from models.diffusion_cfg.activity_encoder import ActivityEncoder
from models.diffusion_cfg.unet_conditional import UNetConditional
from models.diffusion_cfg.conditional_diffusion import ConditionalDiffusion
from models.diffusion_prior.unet_prior import UNetPrior
from models.diffusion_prior.prior_diffusion import DiffusionPrior
from data.generators.build_dataset import load_network, list_networks, load_metadata
from simulator.rate_model import simulate_rate_network


def load_models(
    cond_checkpoint: str,
    prior_checkpoint: Optional[str],
    device: torch.device,
):
    """
    Load conditional CFG model and optional unconditional prior.
    
    Returns:
        diffusion_cond: ConditionalDiffusion model
        diffusion_prior: Optional DiffusionPrior model
        metadata: Dict with model config
    """
    print(f"Loading conditional checkpoint: {cond_checkpoint}")
    ckpt_cond = torch.load(cond_checkpoint, map_location=device)
    
    T = ckpt_cond.get('T', 1000)
    schedule = ckpt_cond.get('schedule', 'cosine')
    d_model = ckpt_cond.get('d_model', 128)
    num_heads = ckpt_cond.get('num_heads', 4)
    num_layers = ckpt_cond.get('num_layers', 2)
    N = ckpt_cond.get('N', 50)
    n_obs = ckpt_cond.get('N_obs', N)
    
    print(f"  T={T}, schedule={schedule}, d_model={d_model}")
    print(f"  N={N}, N_obs={n_obs}")
    
    # Create conditional model
    activity_encoder = ActivityEncoder(
        n_obs=n_obs,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)
    
    unet_cond = UNetConditional(
        in_channels=1,
        base_channels=64,
        channel_multipliers=(1, 2, 4),
        cond_dim=d_model,
    ).to(device)
    
    diffusion_cond = ConditionalDiffusion(
        unet=unet_cond,
        activity_encoder=activity_encoder,
        T=T,
        schedule=schedule,
    ).to(device)
    
    diffusion_cond.load_state_dict(ckpt_cond['model_state_dict'])
    diffusion_cond.eval()
    
    diffusion_prior = None
    if prior_checkpoint:
        print(f"\nLoading unconditional prior checkpoint: {prior_checkpoint}")
        ckpt_prior = torch.load(prior_checkpoint, map_location=device)
        T_prior = ckpt_prior.get('T', 1000)
        schedule_prior = ckpt_prior.get('schedule', 'cosine')
        
        unet_prior = UNetPrior(
            in_channels=1,
            base_channels=64,
            channel_multipliers=(1, 2, 4),
        ).to(device)
        
        diffusion_prior = DiffusionPrior(
            unet=unet_prior,
            T=T_prior,
            schedule=schedule_prior,
        ).to(device)
        
        diffusion_prior.load_state_dict(ckpt_prior['model_state_dict'])
        diffusion_prior.eval()
        print(f"  T={T_prior}, schedule={schedule_prior}")
    
    metadata = {
        'N': N,
        'n_obs': n_obs,
        'T': T,
        'd_model': d_model,
    }
    
    return diffusion_cond, diffusion_prior, metadata


def load_dataset(dataset_dir: str, num_eval: int, activity_mode: str = "first_trial"):
    """
    Load (W, A) pairs from dataset.
    
    Args:
        dataset_dir: Directory with network_*.pt files
        num_eval: Number of samples to load
        activity_mode: "first_trial", "mean_trials", or "random_trial"
    
    Returns:
        W_list: List of [N, N] weight matrices
        A_list: List of [T, N_obs] activity matrices
        metadata: Dataset metadata
    """
    network_files = list_networks(dataset_dir)
    metadata = load_metadata(dataset_dir)
    
    num_eval = min(num_eval, len(network_files))
    
    W_list = []
    A_list = []
    
    for i in range(num_eval):
        data = load_network(network_files[i])
        W = data["W"].float()  # [N, N]
        activities = data["activities"]  # List of [T, N_obs]
        
        # Select activity based on mode
        if activity_mode == "first_trial":
            A = activities[0].float()  # [T, N_obs]
        elif activity_mode == "mean_trials":
            A = torch.stack(activities, dim=0).float().mean(dim=0)  # [T, N_obs]
        elif activity_mode == "random_trial":
            A = random.choice(activities).float()  # [T, N_obs]
        else:
            raise ValueError(f"Unknown activity_mode: {activity_mode}")
        
        W_list.append(W)
        A_list.append(A)
    
    print(f"Loaded {num_eval} (W, A) pairs")
    print(f"  Activity mode: {activity_mode}")
    print(f"  W shape: {W_list[0].shape}, A shape: {A_list[0].shape}")
    
    return W_list, A_list, metadata


def compute_entrywise_mse(W_pred: torch.Tensor, W_true: torch.Tensor) -> float:
    """Compute entrywise MSE."""
    return ((W_pred - W_true) ** 2).mean().item()


def compute_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = np.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_entrywise_correlation(W_pred: torch.Tensor, W_true: torch.Tensor) -> float:
    """Compute entrywise Pearson correlation."""
    pred_flat = W_pred.flatten().cpu().numpy()
    true_flat = W_true.flatten().cpu().numpy()
    corr = compute_pearsonr(pred_flat, true_flat)
    return corr


def compute_spectral_radius(W: torch.Tensor) -> float:
    """Compute spectral radius (max absolute eigenvalue)."""
    W_cpu = W.cpu()
    eigenvalues = torch.linalg.eigvals(W_cpu)
    return torch.abs(eigenvalues).max().item()


def compute_row_col_norm_mse(W_pred: torch.Tensor, W_true: torch.Tensor) -> tuple:
    """Compute MSE of row and column L2 norms."""
    row_norms_pred = torch.norm(W_pred, dim=1)  # [N]
    row_norms_true = torch.norm(W_true, dim=1)  # [N]
    row_mse = ((row_norms_pred - row_norms_true) ** 2).mean().item()
    
    col_norms_pred = torch.norm(W_pred, dim=0)  # [N]
    col_norms_true = torch.norm(W_true, dim=0)  # [N]
    col_mse = ((col_norms_pred - col_norms_true) ** 2).mean().item()
    
    return row_mse, col_mse


def simulate_activity_for_comparison(
    W: torch.Tensor,
    T: int,
    dt: float,
    tau: float,
    num_trials: int = 5,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Simulate activity from weight matrix for functional comparison.
    
    Returns:
        A: [T, N] average activity across trials
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    N = W.shape[0]
    device = W.device
    dtype = W.dtype
    
    activities = []
    for _ in range(num_trials):
        h0 = torch.randn(N, device=device, dtype=dtype)
        A_trial = simulate_rate_network(W, h0, T, dt, tau, noise_std=0.0)
        activities.append(A_trial)
    
    # Average across trials: [T, N]
    A_avg = torch.stack(activities, dim=0).mean(dim=0)
    return A_avg


def autocorr_1d(x: torch.Tensor) -> torch.Tensor:
    """Compute normalized autocorrelation for 1D signal."""
    x = x - x.mean()
    if x.numel() < 2:
        return torch.tensor([1.0], device=x.device, dtype=x.dtype)
    n = x.numel()
    # full autocorr using convolution with reversed signal
    corr_full = torch.nn.functional.conv1d(
        x.view(1, 1, -1), x.flip(0).view(1, 1, -1), padding=n - 1
    ).flatten()  # length 2*n - 1
    # keep non-negative lags: length n, starting at lag 0
    corr = corr_full[n - 1:]
    # normalize by zero-lag value to get correlation-like scale
    corr = corr / corr[0].clamp(min=1e-8)
    return corr


def compute_activity_similarity(
    W_pred: torch.Tensor,
    W_true: torch.Tensor,
    T: int,
    dt: float,
    tau: float,
) -> Dict[str, float]:
    """
    Compute functional similarity by simulating activity.
    
    Returns:
        Dict with mse/corr over rates, variance, full activity, and autocorr correlation.
    """
    # Simulate activities
    A_pred = simulate_activity_for_comparison(W_pred, T, dt, tau, seed=42)  # [T, N]
    A_true = simulate_activity_for_comparison(W_true, T, dt, tau, seed=42)  # [T, N]
    
    # Time-averaged firing rates
    rate_pred = A_pred.mean(dim=0)  # [N]
    rate_true = A_true.mean(dim=0)  # [N]
    
    mse_rate = ((rate_pred - rate_true) ** 2).mean().item()
    corr_rate = compute_pearsonr(rate_pred.cpu().numpy(), rate_true.cpu().numpy())
    
    # Variance per neuron
    var_pred = A_pred.var(dim=0)  # [N]
    var_true = A_true.var(dim=0)  # [N]
    var_mse_per_neuron = ((var_pred - var_true) ** 2).mean().item()
    
    # Full activity MSE
    mse_activity = ((A_pred - A_true) ** 2).mean().item()
    
    # Autocorrelation similarity (mean over neurons)
    autocorr_corr_list = []
    for n in range(A_pred.shape[1]):
        ac_pred = autocorr_1d(A_pred[:, n])
        ac_true = autocorr_1d(A_true[:, n])
        min_len = min(ac_pred.numel(), ac_true.numel())
        c = compute_pearsonr(
            ac_pred[:min_len].cpu().numpy(), ac_true[:min_len].cpu().numpy()
        )
        autocorr_corr_list.append(c)
    autocorr_corr = float(np.mean(autocorr_corr_list))
    
    return {
        'mse_rate': float(mse_rate),
        'corr_rate': float(corr_rate),
        'mse_activity': float(mse_activity),
        'var_mse_per_neuron': float(var_mse_per_neuron),
        'autocorr_corr': float(autocorr_corr),
    }


def evaluate_reconstruction(
    diffusion_cond: ConditionalDiffusion,
    W_true: torch.Tensor,
    A: torch.Tensor,
    N: int,
    n_obs: int,
    num_samples: int,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate K samples of W_pred conditioned on A.
    
    Returns:
        W_pred_samples: [num_samples, N, N] averaged prediction
    """
    A_batch = A.unsqueeze(0).to(device)  # [1, T, N_obs]
    
    # Encode activity
    cond_tokens, cond_pooled = diffusion_cond.activity_encoder(A_batch)
    
    # Generate multiple samples and average
    W_samples = []
    for _ in range(num_samples):
        W_sample = diffusion_cond.sample(
            cond_tokens=cond_tokens,
            cond_pooled=cond_pooled,
            N=N,
            guidance_scale=guidance_scale,
            device=device,
        )  # [1, 1, N, N]
        W_samples.append(W_sample.squeeze(0).squeeze(0))  # [N, N]
    
    # Average across samples
    W_pred = torch.stack(W_samples, dim=0).mean(dim=0)  # [N, N]
    return W_pred.cpu()


def evaluate_unconditional_baseline(
    diffusion_prior: DiffusionPrior,
    N: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate unconditional prior sample."""
    W_samples = []
    for _ in range(num_samples):
        W_sample = diffusion_prior.sample(1, N, device)  # [1, 1, N, N]
        W_samples.append(W_sample.squeeze(0).squeeze(0))  # [N, N]
    
    W_pred = torch.stack(W_samples, dim=0).mean(dim=0)  # [N, N]
    return W_pred.cpu()


def aggregate_metrics(metrics_list: List[Dict], aggregate: str = "mean") -> Dict:
    """Aggregate metrics across samples."""
    if aggregate == "mean":
        return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    elif aggregate == "median":
        return {k: np.median([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")


def print_metrics_summary(
    metrics_cond: Dict,
    metrics_prior: Optional[Dict],
    num_samples: int,
):
    """Print formatted metrics summary."""
    print("\n" + "=" * 80)
    print(f"Reconstruction Metrics (K={num_samples} samples averaged)")
    print("=" * 80)
    
    print("\nConditional CFG Model:")
    print(f"  Entrywise MSE:            {metrics_cond['mse_entry']:.6f}")
    print(f"  Entrywise Correlation:    {metrics_cond['corr_entry']:.6f}")
    print(f"  Spectral radius error:    {metrics_cond['spec_radius_error']:.6f}")
    print(f"  Row norm MSE:             {metrics_cond['mse_row_norm']:.6f}")
    print(f"  Col norm MSE:             {metrics_cond['mse_col_norm']:.6f}")
    
    if 'mse_rate' in metrics_cond:
        print(f"  Activity rate MSE:        {metrics_cond['mse_rate']:.6f}")
        print(f"  Activity rate correlation: {metrics_cond['corr_rate']:.6f}")
    if 'mse_activity' in metrics_cond:
        print(f"  Activity MSE:             {metrics_cond['mse_activity']:.6f}")
    if 'var_mse_per_neuron' in metrics_cond:
        print(f"  Activity variance MSE:    {metrics_cond['var_mse_per_neuron']:.6f}")
    if 'autocorr_corr' in metrics_cond:
        print(f"  Autocorr correlation:     {metrics_cond['autocorr_corr']:.6f}")
    
    if metrics_prior:
        print("\nBaseline (Unconditional Prior):")
        print(f"  Entrywise MSE:            {metrics_prior['mse_entry']:.6f}")
        print(f"  Entrywise Correlation:    {metrics_prior['corr_entry']:.6f}")
        print(f"  Spectral radius error:    {metrics_prior['spec_radius_error']:.6f}")
        print(f"  Row norm MSE:             {metrics_prior['mse_row_norm']:.6f}")
        print(f"  Col norm MSE:             {metrics_prior['mse_col_norm']:.6f}")
    
    print("=" * 80)


def main(
    cond_checkpoint: str,
    prior_checkpoint: Optional[str],
    dataset_dir: str,
    num_eval: int = 100,
    num_samples: int = 4,
    guidance_scale: float = 3.0,  # Default changed to 3.0
    activity_mode: str = "first_trial",
    aggregate: str = "mean",
    simulate_activity: bool = False,
    output_dir: Optional[str] = None,
    device: str = "cuda",
):
    """Main evaluation function."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup output directory - default changed to reconstruction_results_v2
    if output_dir is None:
        output_dir = "evaluation/reconstruction_results_v2"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    print(f"Guidance scale: {guidance_scale}")
    
    # Load models
    print("\n" + "=" * 80)
    print("Loading Models")
    print("=" * 80)
    diffusion_cond, diffusion_prior, model_metadata = load_models(
        cond_checkpoint, prior_checkpoint, device
    )
    
    N = model_metadata['N']
    n_obs = model_metadata['n_obs']
    
    # Load dataset
    print("\n" + "=" * 80)
    print("Loading Dataset")
    print("=" * 80)
    W_list, A_list, dataset_metadata = load_dataset(dataset_dir, num_eval, activity_mode)
    
    # Get simulation parameters from dataset metadata
    T_sim = dataset_metadata.get('T', 100)
    dt = dataset_metadata.get('dt', 0.1)
    tau = dataset_metadata.get('tau', 1.0)
    
    # Evaluate each sample
    print("\n" + "=" * 80)
    print(f"Evaluating {num_eval} samples")
    print("=" * 80)
    
    metrics_cond_list = []
    metrics_prior_list = []
    
    # Storage for saving predictions
    W_true_list = []
    W_pred_cond_list = []
    W_pred_prior_list = []
    A_conditioning_list = []
    A_pred_sim_list = []
    A_true_sim_list = []
    
    with torch.no_grad():
        for i in range(num_eval):
            if (i + 1) % 10 == 0:
                print(f"  Processing sample {i+1}/{num_eval}...", flush=True)
            
            W_true = W_list[i]  # [N, N]
            A = A_list[i]  # [T, N_obs]
            
            # Conditional prediction
            W_pred_cond = evaluate_reconstruction(
                diffusion_cond,
                W_true,
                A,
                N,
                n_obs,
                num_samples,
                guidance_scale,
                device,
            )
            
            # Compute metrics for conditional
            mse_entry = compute_entrywise_mse(W_pred_cond, W_true)
            corr_entry = compute_entrywise_correlation(W_pred_cond, W_true)
            spec_radius_true = compute_spectral_radius(W_true)
            spec_radius_pred = compute_spectral_radius(W_pred_cond)
            spec_radius_error = abs(spec_radius_pred - spec_radius_true)
            mse_row_norm, mse_col_norm = compute_row_col_norm_mse(W_pred_cond, W_true)
            
            metrics_cond = {
                'mse_entry': mse_entry,
                'corr_entry': corr_entry,
                'spec_radius_error': spec_radius_error,
                'mse_row_norm': mse_row_norm,
                'mse_col_norm': mse_col_norm,
            }
            
            # Optional: functional similarity
            if simulate_activity:
                activity_metrics = compute_activity_similarity(
                    W_pred_cond, W_true, T_sim, dt, tau
                )
                metrics_cond.update(activity_metrics)
                
                # Store simulated activities
                A_pred_sim = simulate_activity_for_comparison(W_pred_cond, T_sim, dt, tau, seed=42)
                A_true_sim = simulate_activity_for_comparison(W_true, T_sim, dt, tau, seed=42)
                A_pred_sim_list.append(A_pred_sim.cpu())
                A_true_sim_list.append(A_true_sim.cpu())
            
            metrics_cond_list.append(metrics_cond)
            
            # Store for saving
            W_true_list.append(W_true.cpu())
            W_pred_cond_list.append(W_pred_cond.cpu())
            A_conditioning_list.append(A.cpu())
            
            # Unconditional baseline (if available)
            if diffusion_prior:
                W_pred_prior = evaluate_unconditional_baseline(
                    diffusion_prior, N, num_samples, device
                )
                
                mse_entry_prior = compute_entrywise_mse(W_pred_prior, W_true)
                corr_entry_prior = compute_entrywise_correlation(W_pred_prior, W_true)
                spec_radius_pred_prior = compute_spectral_radius(W_pred_prior)
                spec_radius_error_prior = abs(spec_radius_pred_prior - spec_radius_true)
                mse_row_norm_prior, mse_col_norm_prior = compute_row_col_norm_mse(
                    W_pred_prior, W_true
                )
                
                metrics_prior = {
                    'mse_entry': mse_entry_prior,
                    'corr_entry': corr_entry_prior,
                    'spec_radius_error': spec_radius_error_prior,
                    'mse_row_norm': mse_row_norm_prior,
                    'mse_col_norm': mse_col_norm_prior,
                }
                
                metrics_prior_list.append(metrics_prior)
                W_pred_prior_list.append(W_pred_prior.cpu())
            else:
                W_pred_prior_list.append(None)
    
    # Aggregate metrics
    print("\n" + "=" * 80)
    print("Aggregating Metrics")
    print("=" * 80)
    
    metrics_cond_agg = aggregate_metrics(metrics_cond_list, aggregate)
    metrics_prior_agg = aggregate_metrics(metrics_prior_list, aggregate) if metrics_prior_list else None
    
    # Print summary
    print_metrics_summary(metrics_cond_agg, metrics_prior_agg, num_samples)
    
    # Save predictions and data
    print("\n" + "=" * 80)
    print("Saving Predictions and Data")
    print("=" * 80)
    
    # Stack all tensors
    W_true_stack = torch.stack(W_true_list, dim=0)  # [num_eval, N, N]
    W_pred_cond_stack = torch.stack(W_pred_cond_list, dim=0)  # [num_eval, N, N]
    A_conditioning_stack = torch.stack(A_conditioning_list, dim=0)  # [num_eval, T, N_obs]
    
    # Save conditional predictions (and simulated activities if available)
    torch.save({
        'W_true': W_true_stack,
        'W_pred_cond': W_pred_cond_stack,
        'A_conditioning': A_conditioning_stack,
        'num_samples': num_samples,
        'guidance_scale': guidance_scale,
        'activity_mode': activity_mode,
        'N': N,
        'n_obs': n_obs,
    }, output_path / 'conditional_predictions.pt')
    print(f"  Saved conditional predictions to {output_path / 'conditional_predictions.pt'}")
    
    # Save unconditional baseline if available
    if diffusion_prior and all(w is not None for w in W_pred_prior_list):
        W_pred_prior_stack = torch.stack(W_pred_prior_list, dim=0)
        torch.save({
            'W_true': W_true_stack,
            'W_pred_prior': W_pred_prior_stack,
            'num_samples': num_samples,
        }, output_path / 'unconditional_predictions.pt')
        print(f"  Saved unconditional predictions to {output_path / 'unconditional_predictions.pt'}")
    
    # Save simulated activities if computed
    if simulate_activity and len(A_pred_sim_list) > 0 and len(A_true_sim_list) > 0:
        torch.save({
            'A_true_sim': torch.stack(A_true_sim_list, dim=0),
            'A_pred_sim': torch.stack(A_pred_sim_list, dim=0),
        }, output_path / 'simulated_activities.pt')
        print(f"  Saved simulated activities to {output_path / 'simulated_activities.pt'}")
    
    # Save metrics
    torch.save({
        'metrics_cond': metrics_cond_list,
        'metrics_cond_aggregated': metrics_cond_agg,
        'metrics_prior': metrics_prior_list if metrics_prior_list else None,
        'metrics_prior_aggregated': metrics_prior_agg,
        'aggregate_method': aggregate,
        'num_eval': num_eval,
        'num_samples': num_samples,
        'guidance_scale': guidance_scale,
        'activity_mode': activity_mode,
        'simulate_activity': simulate_activity,
    }, output_path / 'metrics.pt')
    print(f"  Saved metrics to {output_path / 'metrics.pt'}")
    
    # Save metadata
    metadata = {
        'cond_checkpoint': cond_checkpoint,
        'prior_checkpoint': prior_checkpoint,
        'dataset_dir': dataset_dir,
        'num_eval': num_eval,
        'num_samples': num_samples,
        'guidance_scale': guidance_scale,
        'activity_mode': activity_mode,
        'aggregate': aggregate,
        'simulate_activity': simulate_activity,
        'N': N,
        'n_obs': n_obs,
        'T': model_metadata['T'],
        'd_model': model_metadata['d_model'],
    }
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {output_path / 'metadata.json'}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print(f"All results saved to: {output_path}")
    print("=" * 80)


def create_arg_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate CFG reconstruction: Recover W from A_obs (v2 with guidance_scale=3.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--cond_checkpoint",
        type=str,
        required=True,
        help="Path to conditional CFG checkpoint",
    )
    parser.add_argument(
        "--prior_checkpoint",
        type=str,
        default=None,
        help="Optional path to unconditional prior checkpoint for baseline",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory with network_*.pt files",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=100,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of W samples to generate per A (averaged)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,  # Default changed to 3.0
        help="CFG guidance scale",
    )
    parser.add_argument(
        "--activity_mode",
        type=str,
        default="first_trial",
        choices=["first_trial", "mean_trials", "random_trial"],
        help="How to select activity from multiple trials",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation method for metrics",
    )
    parser.add_argument(
        "--simulate_activity",
        action="store_true",
        help="Enable functional similarity via activity simulation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saving predictions (default: evaluation/reconstruction_results_v2)",
    )
    
    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    main(
        cond_checkpoint=args.cond_checkpoint,
        prior_checkpoint=args.prior_checkpoint,
        dataset_dir=args.dataset_dir,
        num_eval=args.num_eval,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        activity_mode=args.activity_mode,
        aggregate=args.aggregate,
        simulate_activity=args.simulate_activity,
        output_dir=args.output_dir,
        device=args.device,
    )

