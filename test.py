# -*- coding: utf-8 -*-
"""
Quick CLI analysis of CFG reconstruction results.

Loads outputs from evaluation/reconstruction_results/ and prints summary
statistics, best/worst samples, and per-sample metrics.

Usage:
  python test.py \
      --results_dir evaluation/reconstruction_results \
      --sample 0 \
      --topk 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_results(results_dir: Path):
    cond_path = results_dir / "conditional_predictions.pt"
    metrics_path = results_dir / "metrics.pt"
    meta_path = results_dir / "metadata.json"

    if not cond_path.exists():
        raise FileNotFoundError(f"Missing {cond_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    cond = torch.load(cond_path, map_location="cpu", weights_only=False)
    metrics = torch.load(metrics_path, map_location="cpu", weights_only=False)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    prior_path = results_dir / "unconditional_predictions.pt"
    prior = torch.load(prior_path, map_location="cpu", weights_only=False) if prior_path.exists() else None

    return cond, prior, metrics, metadata


def summarize(metrics_cond, metrics_prior):
    mse_entries_cond = np.array([m["mse_entry"] for m in metrics_cond])
    corr_entries_cond = np.array([m["corr_entry"] for m in metrics_cond])
    spec_err_cond = np.array([m["spec_radius_error"] for m in metrics_cond])

    summary = {
        "cond": {
            "mse": {
                "mean": mse_entries_cond.mean(),
                "median": np.median(mse_entries_cond),
                "std": mse_entries_cond.std(),
                "min": mse_entries_cond.min(),
                "max": mse_entries_cond.max(),
            },
            "corr": {
                "mean": corr_entries_cond.mean(),
                "median": np.median(corr_entries_cond),
                "std": corr_entries_cond.std(),
                "min": corr_entries_cond.min(),
                "max": corr_entries_cond.max(),
            },
            "spec_err": {
                "mean": spec_err_cond.mean(),
                "median": np.median(spec_err_cond),
                "std": spec_err_cond.std(),
                "min": spec_err_cond.min(),
                "max": spec_err_cond.max(),
            },
        }
    }

    if metrics_prior:
        mse_entries_prior = np.array([m["mse_entry"] for m in metrics_prior])
        corr_entries_prior = np.array([m["corr_entry"] for m in metrics_prior])
        spec_err_prior = np.array([m["spec_radius_error"] for m in metrics_prior])

        summary["prior"] = {
            "mse": {
                "mean": mse_entries_prior.mean(),
                "median": np.median(mse_entries_prior),
                "std": mse_entries_prior.std(),
                "min": mse_entries_prior.min(),
                "max": mse_entries_prior.max(),
            },
            "corr": {
                "mean": corr_entries_prior.mean(),
                "median": np.median(corr_entries_prior),
                "std": corr_entries_prior.std(),
                "min": corr_entries_prior.min(),
                "max": corr_entries_prior.max(),
            },
            "spec_err": {
                "mean": spec_err_prior.mean(),
                "median": np.median(spec_err_prior),
                "std": spec_err_prior.std(),
                "min": spec_err_prior.min(),
                "max": spec_err_prior.max(),
            },
        }

        summary["improvement"] = {
            "mse_mean": mse_entries_prior.mean() - mse_entries_cond.mean(),
            "corr_mean": corr_entries_cond.mean() - corr_entries_prior.mean(),
            "spec_err_mean": spec_err_prior.mean() - spec_err_cond.mean(),
        }

    return summary, mse_entries_cond, corr_entries_cond


def print_summary(summary):
    print("=" * 80)
    print("Aggregated Metrics Summary")
    print("=" * 80)

    cond = summary["cond"]
    print("\nConditional CFG:")
    print(f"  MSE mean/med/std/min/max: "
          f"{cond['mse']['mean']:.6f} / {cond['mse']['median']:.6f} / "
          f"{cond['mse']['std']:.6f} / {cond['mse']['min']:.6f} / {cond['mse']['max']:.6f}")
    print(f"  Corr mean/med/std/min/max: "
          f"{cond['corr']['mean']:.6f} / {cond['corr']['median']:.6f} / "
          f"{cond['corr']['std']:.6f} / {cond['corr']['min']:.6f} / {cond['corr']['max']:.6f}")
    print(f"  Spectral error mean/med/std/min/max: "
          f"{cond['spec_err']['mean']:.6f} / {cond['spec_err']['median']:.6f} / "
          f"{cond['spec_err']['std']:.6f} / {cond['spec_err']['min']:.6f} / {cond['spec_err']['max']:.6f}")

    if "prior" in summary:
        prior = summary["prior"]
        print("\nUnconditional Prior:")
        print(f"  MSE mean/med/std/min/max: "
              f"{prior['mse']['mean']:.6f} / {prior['mse']['median']:.6f} / "
              f"{prior['mse']['std']:.6f} / {prior['mse']['min']:.6f} / {prior['mse']['max']:.6f}")
        print(f"  Corr mean/med/std/min/max: "
              f"{prior['corr']['mean']:.6f} / {prior['corr']['median']:.6f} / "
              f"{prior['corr']['std']:.6f} / {prior['corr']['min']:.6f} / {prior['corr']['max']:.6f}")
        print(f"  Spectral error mean/med/std/min/max: "
              f"{prior['spec_err']['mean']:.6f} / {prior['spec_err']['median']:.6f} / "
              f"{prior['spec_err']['std']:.6f} / {prior['spec_err']['min']:.6f} / {prior['spec_err']['max']:.6f}")

        imp = summary["improvement"]
        print("\nImprovement (CFG vs Prior):")
        print(f"  Δ MSE mean: {imp['mse_mean']:.6f}")
        print(f"  Δ Corr mean: {imp['corr_mean']:.6f}")
        print(f"  Δ Spectral error mean: {imp['spec_err_mean']:.6f}")


def print_best_worst(metrics_cond, topk=5):
    correlations = np.array([m["corr_entry"] for m in metrics_cond])
    mse_entries = np.array([m["mse_entry"] for m in metrics_cond])

    best_corr_idx = np.argsort(-correlations)[:topk]
    worst_corr_idx = np.argsort(correlations)[:topk]

    print("\nTop samples by correlation:")
    for i, idx in enumerate(best_corr_idx):
        print(f"  {i+1}. idx={idx} corr={correlations[idx]:.6f} mse={mse_entries[idx]:.6f}")

    print("\nWorst samples by correlation:")
    for i, idx in enumerate(worst_corr_idx):
        print(f"  {i+1}. idx={idx} corr={correlations[idx]:.6f} mse={mse_entries[idx]:.6f}")


def print_sample(metrics_cond, idx):
    m = metrics_cond[idx]
    print(f"\nSample {idx} metrics (CFG):")
    for k, v in m.items():
        print(f"  {k}: {v:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze CFG reconstruction results.")
    parser.add_argument("--results_dir", type=str, default="evaluation/reconstruction_results",
                        help="Directory containing saved reconstruction outputs.")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to inspect.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k best/worst samples to list.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cond, prior, metrics, metadata = load_results(results_dir)

    W_true = cond["W_true"]
    W_pred_cond = cond["W_pred_cond"]
    A_cond = cond["A_conditioning"]
    metrics_cond = metrics["metrics_cond"]
    metrics_prior = metrics["metrics_prior"] if prior else None

    print("=" * 80)
    print("Loaded data")
    print("=" * 80)
    print(f"  W_true shape:        {tuple(W_true.shape)}")
    print(f"  W_pred_cond shape:   {tuple(W_pred_cond.shape)}")
    print(f"  A_conditioning shape:{tuple(A_cond.shape)}")
    if prior:
        print(f"  W_pred_prior shape:  {tuple(prior['W_pred_prior'].shape)}")
    else:
        print("  No unconditional prior predictions found.")

    summary, mse_entries_cond, corr_entries_cond = summarize(metrics_cond, metrics_prior)
    print_summary(summary)
    print_best_worst(metrics_cond, topk=args.topk)
    idx = max(0, min(args.sample, len(metrics_cond) - 1))
    print_sample(metrics_cond, idx)

    # Optional: print metadata
    print("\nMetadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

