# -*- coding: utf-8 -*-
"""
Training script for conditional diffusion (classifier-free guidance) that maps
observed activity A_obs -> weight matrices W.
"""

import sys
from pathlib import Path
import argparse
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from data.generators.build_dataset import load_network, list_networks, load_metadata
from models.diffusion_cfg.activity_encoder import ActivityEncoder
from models.diffusion_cfg.unet_conditional import UNetConditional
from models.diffusion_cfg.conditional_diffusion import ConditionalDiffusion
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class WeightsActivityDataset(Dataset):
    """Dataset of (W, activity) pairs."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.network_files = list_networks(dataset_dir)
        if len(self.network_files) == 0:
            raise ValueError(f"No network files found in {dataset_dir}")
        self.metadata = load_metadata(dataset_dir)

    def __len__(self):
        return len(self.network_files)

    def __getitem__(self, idx: int):
        data = load_network(self.network_files[idx])
        W = data["W"].float()  # [N, N]
        activities = data["activities"]
        act = random.choice(activities).float()  # [T, N_obs]
        return W, act


def create_dataloaders(dataset_dir: str, batch_size: int, device: torch.device):
    dataset = WeightsActivityDataset(dataset_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return dataset, dataloader


def train(
    dataset_dir: str,
    num_epochs: int = 1000,
    batch_size: int = 8,
    lr: float = 1e-5,
    T: int = 1000,
    schedule: str = "cosine",
    base_channels: int = 64,
    channel_multipliers: tuple = (1, 2, 4),
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    cond_drop_prob: float = 0.1,
    checkpoint_dir: str = "checkpoints_cfg_v2",
    save_every: int = 10,
    log_interval: int = 100,
    device: str = "cuda",
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    wandb_enabled: bool = False,
    wandb_project: str = "diffusion-inference",
    wandb_run_name: str = None,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, dataloader = create_dataloaders(dataset_dir, batch_size, device)
    meta = dataset.metadata
    N = meta["N"]
    n_obs = meta.get("N_obs", meta["N"]) if "N_obs" in meta else meta["N"]
    print(f"Loaded dataset: {len(dataset)} samples, N={N}, N_obs={n_obs}")

    activity_encoder = ActivityEncoder(n_obs=n_obs, d_model=d_model, num_heads=num_heads, num_layers=num_layers).to(device)
    unet = UNetConditional(
        in_channels=1,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        cond_dim=d_model,
    ).to(device)
    diffusion = ConditionalDiffusion(
        unet=unet,
        activity_encoder=activity_encoder,
        T=T,
        schedule=schedule,
        cond_drop_prob=cond_drop_prob,
    ).to(device)

    optimizer = AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = num_epochs * len(dataloader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if wandb_enabled and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "dataset_dir": dataset_dir,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "T": T,
                "schedule": schedule,
                "base_channels": base_channels,
                "channel_multipliers": channel_multipliers,
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "cond_drop_prob": cond_drop_prob,
                "device": str(device),
            },
        )
        print(f"W&B initialized: project={wandb_project}, run={wandb_run_name}")
    elif wandb_enabled and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without logging.")

    global_step = 0
    for epoch in range(num_epochs):
        diffusion.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for W, A in pbar:
            W = W.to(device).unsqueeze(1)  # [B, 1, N, N]
            A = A.to(device)  # [B, T, N_obs]

            optimizer.zero_grad()
            loss = diffusion.p_losses(W, A)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if wandb_enabled and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step,
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - avg loss: {avg_loss:.6f}")
        if wandb_enabled and WANDB_AVAILABLE:
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch": epoch + 1})

        if (epoch + 1) % save_every == 0:
            ckpt_file = checkpoint_path / f"cfg_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "model_state_dict": diffusion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "N": N,
                    "N_obs": n_obs,
                    "T": T,
                    "schedule": schedule,
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                },
                ckpt_file,
            )
            print(f"Saved checkpoint to {ckpt_file}")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train conditional diffusion (CFG) for mapping A_obs -> W.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory with network_*.pt files")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channel_multipliers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_cfg_v2")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="diffusion-inference")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    train(
        dataset_dir=args.dataset_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        T=args.T,
        schedule=args.schedule,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        cond_drop_prob=args.cond_drop_prob,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        log_interval=args.log_interval,
        device=args.device,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

