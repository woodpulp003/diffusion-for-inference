# -*- coding: utf-8 -*-
"""
Compare CFG v1 and v2 checkpoints side-by-side using the same evaluation metrics.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import random

from evaluation.eval_cfg import (
    load_real_weights_activities,
    sample_fake_weights_cfg,
    compute_entry_statistics,
    compute_spectral_radii,
    compute_row_col_norms,
    print_entry_statistics,
    print_spectral_radius_comparison,
    print_norm_comparison,
)


def compare_cfg_models(
    checkpoint_v1,
    checkpoint_v2,
    dataset_dir,
    num_eval=256,
    guidance_scale=1.5,
    device="cuda",
):
    """
    Compare CFG v1 and v2 checkpoints.
    
    Args:
        checkpoint_v1: Path to v1 checkpoint.
        checkpoint_v2: Path to v2 checkpoint.
        dataset_dir: Directory with network_*.pt files.
        num_eval: Number of weight matrices to compare.
        guidance_scale: CFG guidance scale.
        device: Device to use.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("CFG Model Comparison: v1 vs v2")
    print("=" * 80)
    print(f"Using device: {device}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Number of samples: {num_eval}")
    
    # Load real (W, A) pairs (same for both models)
    print("\n" + "=" * 80)
    print("Loading Real Weight Matrices and Activity")
    print("=" * 80)
    real_W, real_A, N, n_obs = load_real_weights_activities(dataset_dir, num_eval, device)
    
    # Sample from v1
    print("\n" + "=" * 80)
    print("Sampling from CFG v1")
    print("=" * 80)
    fake_W_v1 = sample_fake_weights_cfg(
        checkpoint_v1,
        real_A,
        N,
        n_obs,
        device,
        guidance_scale=guidance_scale,
    )
    
    # Sample from v2
    print("\n" + "=" * 80)
    print("Sampling from CFG v2")
    print("=" * 80)
    fake_W_v2 = sample_fake_weights_cfg(
        checkpoint_v2,
        real_A,
        N,
        n_obs,
        device,
        guidance_scale=guidance_scale,
    )
    
    # Ensure same number of samples
    num_eval = min(real_W.shape[0], fake_W_v1.shape[0], fake_W_v2.shape[0])
    real_W = real_W[:num_eval]
    fake_W_v1 = fake_W_v1[:num_eval]
    fake_W_v2 = fake_W_v2[:num_eval]
    
    print(f"\nComparing {num_eval} real vs {num_eval} fake (v1) vs {num_eval} fake (v2) weight matrices...")
    
    # Compute statistics for all three
    print("\n" + "=" * 80)
    print("Computing Entry-Wise Statistics")
    print("=" * 80)
    print("  Processing real weights...", end=' ', flush=True)
    real_entry_stats = compute_entry_statistics(real_W)
    print("✓")
    print("  Processing fake weights (v1)...", end=' ', flush=True)
    fake_entry_stats_v1 = compute_entry_statistics(fake_W_v1)
    print("✓")
    print("  Processing fake weights (v2)...", end=' ', flush=True)
    fake_entry_stats_v2 = compute_entry_statistics(fake_W_v2)
    print("✓")
    
    print("\n" + "=" * 80)
    print("Entry-Wise Statistics Comparison")
    print("=" * 80)
    print(f"{'':<12} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
    print("-" * 80)
    print(f"{'Real W:':<12} {real_entry_stats['mean']:>12.6f} {real_entry_stats['std']:>12.6f} "
          f"{real_entry_stats['min']:>12.6f} {real_entry_stats['max']:>12.6f}")
    print(f"{'Fake W (v1):':<12} {fake_entry_stats_v1['mean']:>12.6f} {fake_entry_stats_v1['std']:>12.6f} "
          f"{fake_entry_stats_v1['min']:>12.6f} {fake_entry_stats_v1['max']:>12.6f}")
    print(f"{'Fake W (v2):':<12} {fake_entry_stats_v2['mean']:>12.6f} {fake_entry_stats_v2['std']:>12.6f} "
          f"{fake_entry_stats_v2['min']:>12.6f} {fake_entry_stats_v2['max']:>12.6f}")
    print("=" * 80)
    
    # Compute spectral radii
    print("\n" + "=" * 80)
    print("Computing Spectral Radii")
    print("=" * 80)
    print(f"  Processing {num_eval} real matrices...", end=' ', flush=True)
    real_spec_radii = compute_spectral_radii(real_W)
    print("✓")
    print(f"  Processing {num_eval} fake matrices (v1)...", end=' ', flush=True)
    fake_spec_radii_v1 = compute_spectral_radii(fake_W_v1)
    print("✓")
    print(f"  Processing {num_eval} fake matrices (v2)...", end=' ', flush=True)
    fake_spec_radii_v2 = compute_spectral_radii(fake_W_v2)
    print("✓")
    
    print("\n" + "=" * 80)
    print("Spectral Radius Comparison")
    print("=" * 80)
    real_mean = real_spec_radii.mean().item()
    real_std = real_spec_radii.std().item()
    print(f"Real: mean {real_mean:.4f}, std {real_std:.4f}")
    
    v1_mean = fake_spec_radii_v1.mean().item()
    v1_std = fake_spec_radii_v1.std().item()
    print(f"Fake (v1): mean {v1_mean:.4f}, std {v1_std:.4f}")
    
    v2_mean = fake_spec_radii_v2.mean().item()
    v2_std = fake_spec_radii_v2.std().item()
    print(f"Fake (v2): mean {v2_mean:.4f}, std {v2_std:.4f}")
    print("=" * 80)
    
    # Compute row/column norms
    print("\n" + "=" * 80)
    print("Computing Row/Column Norms")
    print("=" * 80)
    print("  Processing real weights...", end=' ', flush=True)
    real_row_norms, real_col_norms = compute_row_col_norms(real_W)
    print("✓")
    print("  Processing fake weights (v1)...", end=' ', flush=True)
    fake_row_norms_v1, fake_col_norms_v1 = compute_row_col_norms(fake_W_v1)
    print("✓")
    print("  Processing fake weights (v2)...", end=' ', flush=True)
    fake_row_norms_v2, fake_col_norms_v2 = compute_row_col_norms(fake_W_v2)
    print("✓")
    
    print("\n" + "=" * 80)
    print("Row/Column Norm Comparison")
    print("=" * 80)
    
    print("Row norm mean/std:")
    print(f"  Real: mean {real_row_norms.mean().item():.6f}, std {real_row_norms.std().item():.6f}")
    print(f"  Fake (v1): mean {fake_row_norms_v1.mean().item():.6f}, std {fake_row_norms_v1.std().item():.6f}")
    print(f"  Fake (v2): mean {fake_row_norms_v2.mean().item():.6f}, std {fake_row_norms_v2.std().item():.6f}")
    
    print("\nColumn norm mean/std:")
    print(f"  Real: mean {real_col_norms.mean().item():.6f}, std {real_col_norms.std().item():.6f}")
    print(f"  Fake (v1): mean {fake_col_norms_v1.mean().item():.6f}, std {fake_col_norms_v1.std().item():.6f}")
    print(f"  Fake (v2): mean {fake_col_norms_v2.mean().item():.6f}, std {fake_col_norms_v2.std().item():.6f}")
    
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)


def create_arg_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Compare CFG v1 and v2 checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint_v1",
        type=str,
        required=True,
        help="Path to CFG v1 checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_v2",
        type=str,
        required=True,
        help="Path to CFG v2 checkpoint.",
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
        "--guidance_scale",
        type=float,
        default=1.5,
        help="CFG guidance scale.",
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
    
    compare_cfg_models(
        checkpoint_v1=args.checkpoint_v1,
        checkpoint_v2=args.checkpoint_v2,
        dataset_dir=args.dataset_dir,
        num_eval=args.num_eval,
        guidance_scale=args.guidance_scale,
        device=args.device,
    )

