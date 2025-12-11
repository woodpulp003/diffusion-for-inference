# -*- coding: utf-8 -*-
"""
DPS trainer: loads CFG, produces W_cfg, and refines via DPS.
"""

import sys
from pathlib import Path
import argparse
import torch
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.generators.build_dataset import load_network, list_networks, load_metadata
from models.diffusion_cfg.activity_encoder import ActivityEncoder
from models.diffusion_cfg.unet_conditional import UNetConditional
from models.diffusion_cfg.conditional_diffusion import ConditionalDiffusion
from inference.dps_refinement import refine_weight_matrix


def sample_cfg_prediction(cond_ckpt: str, A_obs: torch.Tensor, N: int, device: torch.device):
    ckpt = torch.load(cond_ckpt, map_location=device)
    T = ckpt.get("T", 1000)
    schedule = ckpt.get("schedule", "cosine")
    d_model = ckpt.get("d_model", 128)
    num_heads = ckpt.get("num_heads", 4)
    num_layers = ckpt.get("num_layers", 2)

    n_obs = A_obs.shape[1]

    activity_encoder = ActivityEncoder(n_obs=n_obs, d_model=d_model, num_heads=num_heads, num_layers=num_layers).to(device)
    unet = UNetConditional(
        in_channels=1,
        base_channels=64,
        channel_multipliers=(1, 2, 4),
        cond_dim=d_model,
    ).to(device)
    diffusion = ConditionalDiffusion(unet=unet, activity_encoder=activity_encoder, T=T, schedule=schedule).to(device)
    diffusion.load_state_dict(ckpt["model_state_dict"])
    diffusion.eval()

    with torch.no_grad():
        A_batch = A_obs.unsqueeze(0).to(device)  # [1, T, n_obs]
        cond_tokens, cond_pooled = activity_encoder(A_batch)
        W_sample = diffusion.sample(cond_tokens, cond_pooled, N=N, guidance_scale=1.5, device=device)  # [1,1,N,N]
        W_sample = W_sample.squeeze(0).squeeze(0)
    return W_sample.cpu()


def main():
    parser = argparse.ArgumentParser(description="DPS refinement pipeline")
    parser.add_argument("--cond_checkpoint", type=str, required=True, help="CFG checkpoint")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset dir with network_*.pt")
    parser.add_argument("--out_dir", type=str, default="evaluation/dps_results", help="Where to save results")
    parser.add_argument("--num_eval", type=int, default=10, help="How many samples to process")
    parser.add_argument("--num_steps", type=int, default=300, help="Refinement steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Refinement LR")
    parser.add_argument("--lambda_prior", type=float, default=0.01, help="Prior penalty weight")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="diffusion-inference", help="W&B project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    WANDB_AVAILABLE = False
    if args.wandb:
        try:
            import wandb  # type: ignore
            WANDB_AVAILABLE = True
        except ImportError:
            print("wandb not installed; continuing without logging.")

    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "cond_checkpoint": args.cond_checkpoint,
                "dataset_dir": args.dataset_dir,
                "num_eval": args.num_eval,
                "num_steps": args.num_steps,
                "lr": args.lr,
                "lambda_prior": args.lambda_prior,
                "device": str(device),
            },
        )

    files = list_networks(args.dataset_dir)
    meta = load_metadata(args.dataset_dir)
    T = meta.get("T", 100)
    dt = meta.get("dt", 0.1)
    tau = meta.get("tau", 1.0)
    N = meta.get("N", 50)

    results = {
        "W_true": [],
        "A_obs": [],
        "W_cfg": [],
        "W_dps": [],
        "metadata": meta,
        "refine_params": {
            "num_steps": args.num_steps,
            "lr": args.lr,
            "lambda_prior": args.lambda_prior,
        },
    }

    for idx in range(min(args.num_eval, len(files))):
        data = load_network(files[idx])
        W_true = data["W"].float()
        activities = data["activities"]
        A_obs = random.choice(activities).float()  # [T, N_obs]
        n_obs = A_obs.shape[1]

        # CFG prediction
        W_cfg = sample_cfg_prediction(args.cond_checkpoint, A_obs, N, device)

        # DPS refinement
        metadata = {"T": T, "dt": dt, "tau": tau, "N_obs": n_obs}
        W_dps = refine_weight_matrix(
            W_cfg=W_cfg,
            A_obs=A_obs,
            metadata=metadata,
            num_steps=args.num_steps,
            lr=args.lr,
            lambda_prior=args.lambda_prior,
            device=args.device,
        )

        results["W_true"].append(W_true)
        results["A_obs"].append(A_obs)
        results["W_cfg"].append(W_cfg)
        results["W_dps"].append(W_dps)

        if args.wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "idx": idx,
                    "refine/loss_dummy": 0.0,  # placeholder if needed later
                }
            )

        print(f"[{idx+1}/{args.num_eval}] done")

    # Save results
    results["W_true"] = torch.stack(results["W_true"], dim=0)
    results["A_obs"] = torch.stack(results["A_obs"], dim=0)
    results["W_cfg"] = torch.stack(results["W_cfg"], dim=0)
    results["W_dps"] = torch.stack(results["W_dps"], dim=0)

    torch.save(results, out_path / "dps_results.pt")
    print(f"Saved DPS results to {out_path / 'dps_results.pt'}")

    if args.wandb and WANDB_AVAILABLE:
        wandb.save(str(out_path / "dps_results.pt"))
        wandb.finish()


if __name__ == "__main__":
    main()

