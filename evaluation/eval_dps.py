# -*- coding: utf-8 -*-
"""
Evaluate DPS refinement: compare W_true, W_cfg, and W_dps structurally and functionally.
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.simulator_diff import simulate_rate_network_diff


def compute_entrywise_mse(a, b):
    return ((a - b) ** 2).mean().item()


def compute_entrywise_corr(a, b):
    a_flat = a.flatten().cpu().numpy()
    b_flat = b.flatten().cpu().numpy()
    if a_flat.std() == 0 or b_flat.std() == 0:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def compute_spectral_radius(W):
    ev = torch.linalg.eigvals(W.cpu())
    return torch.abs(ev).max().item()


def compute_row_col_norm_mse(a, b):
    row_mse = ((torch.norm(a, dim=1) - torch.norm(b, dim=1)) ** 2).mean().item()
    col_mse = ((torch.norm(a, dim=0) - torch.norm(b, dim=0)) ** 2).mean().item()
    return row_mse, col_mse


def simulate_activity(W, T, dt, tau):
    h0 = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
    return simulate_rate_network_diff(W, h0, T, dt, tau)


def activity_metrics(W_pred, W_true, T, dt, tau, n_obs):
    A_pred = simulate_activity(W_pred, T, dt, tau)[:, :n_obs]
    A_true = simulate_activity(W_true, T, dt, tau)[:, :n_obs]
    mse = ((A_pred - A_true) ** 2).mean().item()
    rate_pred = A_pred.mean(dim=0)
    rate_true = A_true.mean(dim=0)
    corr = compute_entrywise_corr(rate_pred, rate_true)
    var_mse = ((A_pred.var(dim=0) - A_true.var(dim=0)) ** 2).mean().item()
    return {"mse_activity": mse, "corr_rate": corr, "var_mse": var_mse}


def summarize(metrics_list):
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        out[k] = float(np.mean(vals))
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPS refinement results.")
    parser.add_argument("--results", type=str, default="evaluation/dps_results/dps_results.pt",
                        help="Path to dps_results.pt saved by dps_trainer.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = torch.load(args.results, map_location=device, weights_only=False)

    W_true = data["W_true"]
    W_cfg = data["W_cfg"]
    W_dps = data["W_dps"]
    meta = data.get("metadata", {})
    T = int(meta.get("T", 100))
    dt = float(meta.get("dt", 0.1))
    tau = float(meta.get("tau", 1.0))
    n_obs = int(meta.get("N_obs", W_true.shape[-1]))

    metrics_cfg = []
    metrics_dps = []
    func_cfg = []
    func_dps = []

    for i in range(W_true.shape[0]):
        WT = W_true[i].to(device)
        WC = W_cfg[i].to(device)
        WD = W_dps[i].to(device)

        # Structural
        m_cfg = {
            "mse_entry": compute_entrywise_mse(WC, WT),
            "corr_entry": compute_entrywise_corr(WC, WT),
            "spec_err": abs(compute_spectral_radius(WC) - compute_spectral_radius(WT)),
        }
        row_mse, col_mse = compute_row_col_norm_mse(WC, WT)
        m_cfg["row_mse"] = row_mse
        m_cfg["col_mse"] = col_mse
        metrics_cfg.append(m_cfg)

        m_dps = {
            "mse_entry": compute_entrywise_mse(WD, WT),
            "corr_entry": compute_entrywise_corr(WD, WT),
            "spec_err": abs(compute_spectral_radius(WD) - compute_spectral_radius(WT)),
        }
        row_mse_d, col_mse_d = compute_row_col_norm_mse(WD, WT)
        m_dps["row_mse"] = row_mse_d
        m_dps["col_mse"] = col_mse_d
        metrics_dps.append(m_dps)

        # Functional
        func_cfg.append(activity_metrics(WC, WT, T, dt, tau, n_obs))
        func_dps.append(activity_metrics(WD, WT, T, dt, tau, n_obs))

    # Aggregate
    agg_cfg = summarize(metrics_cfg)
    agg_dps = summarize(metrics_dps)
    agg_func_cfg = summarize(func_cfg)
    agg_func_dps = summarize(func_dps)

    print("=" * 80)
    print("Structural metrics (mean over samples)")
    print("=" * 80)
    print("CFG:")
    for k, v in agg_cfg.items():
        print(f"  {k}: {v:.6f}")
    print("DPS:")
    for k, v in agg_dps.items():
        print(f"  {k}: {v:.6f}")

    print("\n" + "=" * 80)
    print("Functional metrics (mean over samples)")
    print("=" * 80)
    print("CFG:")
    for k, v in agg_func_cfg.items():
        print(f"  {k}: {v:.6f}")
    print("DPS:")
    for k, v in agg_func_dps.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()


