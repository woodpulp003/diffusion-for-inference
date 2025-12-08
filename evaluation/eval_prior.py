# -*- coding: utf-8 -*-
"""
Evaluation script for unconditional diffusion prior over weight matrices.

Compares real weight matrices from the dataset with fake weight matrices
sampled from the trained diffusion prior.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np

from models.diffusion_prior import UNetPrior, DiffusionPrior
from training.train_prior import WeightsDataset


def load_real_weights(dataset_dir, num_eval, device):
    """
    Load real weight matrices from dataset.
    
    Args:
        dataset_dir: Directory containing network_*.pt files.
        num_eval: Number of weight matrices to load.
        device: Device to load on.
    
    Returns:
        real_W: [num_eval, N, N] tensor of real weight matrices.
        N: Network size.
    """
    dataset = WeightsDataset(dataset_dir)
    
    # Limit to available samples
    num_eval = min(num_eval, len(dataset))
    
    real_W_list = []
    for i in range(num_eval):
        W = dataset[i]  # [N, N]
        real_W_list.append(W)
    
    real_W = torch.stack(real_W_list, dim=0)  # [num_eval, N, N]
    N = real_W.shape[1]
    
    print(f"Loaded {num_eval} real weight matrices from {dataset_dir}")
    print(f"  Shape: {real_W.shape}, Network size: N = {N}")
    
    return real_W, N


def sample_fake_weights(
    checkpoint_file,
    num_eval,
    N,
    device,
):
    """
    Sample fake weight matrices from trained diffusion prior.
    
    Args:
        checkpoint_file: Path to trained checkpoint.
        num_eval: Number of samples to generate.
        N: Network size.
        device: Device to sample on.
    
    Returns:
        fake_W: [num_eval, N, N] tensor of sampled weight matrices.
    """
    print(f"Loading checkpoint: {checkpoint_file}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device)
    T = checkpoint.get('T', 1000)
    schedule = checkpoint.get('schedule', 'cosine')
    
    print(f"  T = {T}, schedule = {schedule}")
    
    # Create model
    unet = UNetPrior(in_channels=1, base_channels=64, channel_multipliers=(1, 2, 4))
    diffusion = DiffusionPrior(unet=unet, T=T, schedule=schedule)
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    diffusion.to(device)
    diffusion.eval()
    
    print(f"Sampling {num_eval} fake weight matrices (full quality, {T} steps each)...")
    
    # Sample in batches to avoid memory issues
    batch_size = 32
    fake_W_list = []
    num_batches = (num_eval + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(0, num_eval, batch_size):
            batch_size_actual = min(batch_size, num_eval - batch_idx)
            batch_num = batch_idx // batch_size + 1
            
            print(f"  Batch {batch_num}/{num_batches} ({batch_size_actual} samples)...", end=' ', flush=True)
            
            # Sample: returns [batch_size, 1, N, N]
            W_samples = diffusion.sample(batch_size_actual, N, device, verbose=(batch_num == 1))
            
            # Squeeze channel dimension: [batch_size, N, N]
            W_samples = W_samples.squeeze(1)
            fake_W_list.append(W_samples.cpu())
            
            print("✓")
    
    fake_W = torch.cat(fake_W_list, dim=0)  # [num_eval, N, N]
    
    print(f"  Shape: {fake_W.shape}")
    
    return fake_W


def compute_entry_statistics(W):
    """
    Compute entry-wise statistics for weight matrices.
    
    Args:
        W: [B, N, N] tensor of weight matrices.
    
    Returns:
        Dictionary with mean, std, min, max.
    """
    # Flatten all entries
    entries = W.flatten()  # [B * N * N]
    
    return {
        'mean': entries.mean().item(),
        'std': entries.std().item(),
        'min': entries.min().item(),
        'max': entries.max().item(),
    }


def compute_spectral_radii(W):
    """
    Compute spectral radius for each weight matrix.
    
    Args:
        W: [B, N, N] tensor of weight matrices.
    
    Returns:
        spec_radii: [B] tensor of spectral radii.
    """
    B, N, _ = W.shape
    spec_radii = []
    
    # Move to CPU for eigen decomposition (more stable)
    W_cpu = W.cpu()
    
    for i in range(B):
        W_i = W_cpu[i]  # [N, N]
        eigenvalues = torch.linalg.eigvals(W_i)
        spec_radius = torch.abs(eigenvalues).max().item()
        spec_radii.append(spec_radius)
    
    return torch.tensor(spec_radii)


def compute_row_col_norms(W):
    """
    Compute row and column L2 norms for each weight matrix.
    
    Args:
        W: [B, N, N] tensor of weight matrices.
    
    Returns:
        row_norms: [B * N] flattened row norms.
        col_norms: [B * N] flattened column norms.
    """
    B, N, _ = W.shape
    
    # Row norms: L2 norm along last dimension
    row_norms = torch.norm(W, dim=2)  # [B, N]
    row_norms = row_norms.flatten()  # [B * N]
    
    # Column norms: L2 norm along second-to-last dimension
    col_norms = torch.norm(W, dim=1)  # [B, N]
    col_norms = col_norms.flatten()  # [B * N]
    
    return row_norms, col_norms


def print_entry_statistics(real_stats, fake_stats):
    """Print entry-wise statistics comparison."""
    print("\n" + "=" * 60)
    print("Entry-Wise Statistics Comparison")
    print("=" * 60)
    print(f"{'':<12} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
    print("-" * 60)
    print(f"{'Real W:':<12} {real_stats['mean']:>12.6f} {real_stats['std']:>12.6f} "
          f"{real_stats['min']:>12.6f} {real_stats['max']:>12.6f}")
    print(f"{'Fake W:':<12} {fake_stats['mean']:>12.6f} {fake_stats['std']:>12.6f} "
          f"{fake_stats['min']:>12.6f} {fake_stats['max']:>12.6f}")
    print("=" * 60)


def print_spectral_radius_comparison(real_spec_radii, fake_spec_radii):
    """Print spectral radius comparison."""
    real_mean = real_spec_radii.mean().item()
    real_std = real_spec_radii.std().item()
    real_min = real_spec_radii.min().item()
    real_max = real_spec_radii.max().item()
    
    fake_mean = fake_spec_radii.mean().item()
    fake_std = fake_spec_radii.std().item()
    fake_min = fake_spec_radii.min().item()
    fake_max = fake_spec_radii.max().item()
    
    print("\n" + "=" * 60)
    print("Spectral Radius Comparison")
    print("=" * 60)
    print(f"Real: mean {real_mean:.4f}, std {real_std:.4f}, min {real_min:.4f}, max {real_max:.4f}")
    print(f"Fake: mean {fake_mean:.4f}, std {fake_std:.4f}, min {fake_min:.4f}, max {fake_max:.4f}")
    print("=" * 60)


def print_norm_comparison(
    real_row_norms,
    fake_row_norms,
    real_col_norms,
    fake_col_norms,
):
    """Print row/column norm comparison."""
    print("\n" + "=" * 60)
    print("Row/Column Norm Comparison")
    print("=" * 60)
    
    print("Row norm mean/std:")
    print(f"  Real: mean {real_row_norms.mean().item():.6f}, std {real_row_norms.std().item():.6f}")
    print(f"  Fake: mean {fake_row_norms.mean().item():.6f}, std {fake_row_norms.std().item():.6f}")
    
    print("\nColumn norm mean/std:")
    print(f"  Real: mean {real_col_norms.mean().item():.6f}, std {real_col_norms.std().item():.6f}")
    print(f"  Fake: mean {fake_col_norms.mean().item():.6f}, std {fake_col_norms.std().item():.6f}")
    
    print("=" * 60)


def evaluate_prior(
    checkpoint_file,
    dataset_dir,
    num_eval=256,
    device="cuda",
):
    """
    Main evaluation function.
    
    Args:
        checkpoint_file: Path to trained checkpoint.
        dataset_dir: Directory with network_*.pt files.
        num_eval: Number of weight matrices to compare.
        device: Device to use (cuda or cpu).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load real weight matrices
    print("\n" + "=" * 60)
    print("Loading Real Weight Matrices")
    print("=" * 60)
    real_W, N = load_real_weights(dataset_dir, num_eval, device)
    
    # Sample fake weight matrices
    print("\n" + "=" * 60)
    print("Sampling Fake Weight Matrices")
    print("=" * 60)
    fake_W = sample_fake_weights(checkpoint_file, num_eval, N, device)
    
    # Ensure same number of samples
    num_eval = min(real_W.shape[0], fake_W.shape[0])
    real_W = real_W[:num_eval]
    fake_W = fake_W[:num_eval]
    
    print(f"\nComparing {num_eval} real vs {num_eval} fake weight matrices...")
    
    # Compute entry-wise statistics
    print("\n" + "=" * 60)
    print("Computing Entry-Wise Statistics")
    print("=" * 60)
    print("  Processing real weights...", end=' ', flush=True)
    real_entry_stats = compute_entry_statistics(real_W)
    print("✓")
    print("  Processing fake weights...", end=' ', flush=True)
    fake_entry_stats = compute_entry_statistics(fake_W)
    print("✓")
    print_entry_statistics(real_entry_stats, fake_entry_stats)
    
    # Compute spectral radii
    print("\n" + "=" * 60)
    print("Computing Spectral Radii")
    print("=" * 60)
    print(f"  Processing {num_eval} real matrices...", end=' ', flush=True)
    real_spec_radii = compute_spectral_radii(real_W)
    print("✓")
    print(f"  Processing {num_eval} fake matrices...", end=' ', flush=True)
    fake_spec_radii = compute_spectral_radii(fake_W)
    print("✓")
    print_spectral_radius_comparison(real_spec_radii, fake_spec_radii)
    
    # Compute row/column norms
    print("\n" + "=" * 60)
    print("Computing Row/Column Norms")
    print("=" * 60)
    print("  Processing real weights...", end=' ', flush=True)
    real_row_norms, real_col_norms = compute_row_col_norms(real_W)
    print("✓")
    print("  Processing fake weights...", end=' ', flush=True)
    fake_row_norms, fake_col_norms = compute_row_col_norms(fake_W)
    print("✓")
    print_norm_comparison(real_row_norms, fake_row_norms, real_col_norms, fake_col_norms)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


def create_arg_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate unconditional diffusion prior over weight matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help="Path to trained diffusion model checkpoint.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing network_*.pt files.",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=256,
        help="Number of weight matrices to compare.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for evaluation.",
    )
    
    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    evaluate_prior(
        checkpoint_file=args.checkpoint_file,
        dataset_dir=args.dataset_dir,
        num_eval=args.num_eval,
        device=args.device,
    )

