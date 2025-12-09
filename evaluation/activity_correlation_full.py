# -*- coding: utf-8 -*-
"""
Analyze activity similarity without collapsing axes.

Loads simulated activities from reconstruction results and computes
per-sample Pearson correlation between the full activity tensors
A_pred and A_true (flattened over time and neurons, no averaging).

Usage:
  python evaluation/activity_correlation_full.py \
      --results_dir evaluation/reconstruction_results \
      --topk 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compute_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = np.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())
    if den == 0:
        return 0.0
    return float(num / den)


def main():
    parser = argparse.ArgumentParser(
        description="Full activity correlation analysis (no axis collapsing)."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation/reconstruction_results",
        help="Directory containing reconstruction outputs.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top/bottom-k samples to list by correlation.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    sims_path = results_dir / "simulated_activities.pt"
    meta_path = results_dir / "metadata.json"

    if not sims_path.exists():
        raise FileNotFoundError(
            f"Missing simulated_activities.pt at {sims_path}. "
            "Re-run eval with --simulate_activity to generate it."
        )

    sims = torch.load(sims_path, map_location="cpu", weights_only=False)
    A_true = sims["A_true_sim"]  # [num_eval, T, N]
    A_pred = sims["A_pred_sim"]  # [num_eval, T, N]

    num_eval, T, N = A_true.shape
    print(f"Loaded simulated activities: {num_eval} samples, T={T}, N={N}")

    # Compute per-sample full correlation (flattened, no averaging over axes)
    corrs = []
    for i in range(num_eval):
        corr = compute_pearsonr(
            A_pred[i].flatten().numpy(),
            A_true[i].flatten().numpy(),
        )
        corrs.append(corr)

    corrs = np.array(corrs)
    print("\nFull activity correlation (flattened A_pred vs A_true):")
    print(f"  Mean:   {corrs.mean():.6f}")
    print(f"  Median: {np.median(corrs):.6f}")
    print(f"  Std:    {corrs.std():.6f}")
    print(f"  Min:    {corrs.min():.6f}")
    print(f"  Max:    {corrs.max():.6f}")

    # Top / bottom samples
    k = min(args.topk, num_eval)
    top_idx = np.argsort(-corrs)[:k]
    bot_idx = np.argsort(corrs)[:k]

    print(f"\nTop {k} samples by correlation:")
    for j, idx in enumerate(top_idx, 1):
        print(f"  {j}. idx={idx} corr={corrs[idx]:.6f}")

    print(f"\nBottom {k} samples by correlation:")
    for j, idx in enumerate(bot_idx, 1):
        print(f"  {j}. idx={idx} corr={corrs[idx]:.6f}")

    # Optional: show metadata for context
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print("\nMetadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

