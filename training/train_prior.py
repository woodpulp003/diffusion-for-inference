# -*- coding: utf-8 -*-
"""
Training script for unconditional diffusion prior over weight matrices.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from models.diffusion_prior import UNetPrior, DiffusionPrior
from data.generators.build_dataset import load_network, list_networks


class WeightsDataset(Dataset):
    """Dataset that loads weight matrices from .pt files."""
    
    def __init__(self, dataset_dir: str):
        """
        Args:
            dataset_dir: Directory containing network_*.pt files.
        """
        self.dataset_dir = Path(dataset_dir)
        self.network_files = sorted(self.dataset_dir.glob("network_*.pt"))
        
        if len(self.network_files) == 0:
            raise ValueError(f"No network files found in {dataset_dir}")
        
        print(f"Loaded {len(self.network_files)} networks from {dataset_dir}")
    
    def __len__(self) -> int:
        return len(self.network_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            W: [N, N] weight matrix.
        """
        network_data = load_network(str(self.network_files[idx]))
        W = network_data['W']  # [N, N]
        
        # Ensure float32
        W = W.float()
        
        return W


def train(
    dataset_dir: str,
    num_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    T: int = 1000,
    schedule: str = "cosine",
    base_channels: int = 64,
    channel_multipliers: tuple = (1, 2, 4),
    checkpoint_dir: str = "checkpoints",
    device: str = "cuda",
    save_every: int = 10,
    sample_every: int = 5,
    wandb_enabled: bool = False,
    wandb_project: str = "diffusion-inference",
    wandb_run_name: str = None,
    log_interval: int = 100,
):
    """
    Train unconditional diffusion prior.
    
    Args:
        dataset_dir: Path to dataset directory.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        T: Number of diffusion timesteps.
        schedule: Noise schedule ("linear" or "cosine").
        base_channels: Base channels for UNet.
        channel_multipliers: Channel multipliers for UNet.
        checkpoint_dir: Directory to save checkpoints.
        device: Device to train on.
        save_every: Save checkpoint every N epochs.
        sample_every: Sample and visualize every N epochs.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Initialize wandb
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
                "device": str(device),
            },
        )
        print(f"W&B initialized: project={wandb_project}, run={wandb_run_name}")
    elif wandb_enabled and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without logging.")
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = WeightsDataset(dataset_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Get N from first sample
    W_sample = dataset[0]
    N = W_sample.shape[0]
    print(f"Network size: N = {N}")
    
    # Create model
    unet = UNetPrior(
        in_channels=1,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
    ).to(device)
    
    diffusion = DiffusionPrior(
        unet=unet,
        T=T,
        schedule=schedule,
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(diffusion.parameters(), lr=lr)
    
    # Training loop
    global_step = 0
    
    for epoch in range(num_epochs):
        diffusion.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, W in enumerate(pbar):
            # W is [B, N, N], reshape to [B, 1, N, N]
            W = W.unsqueeze(1).to(device)  # [B, 1, N, N]
            
            # Compute loss
            loss = diffusion.p_losses(W)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to wandb
            if wandb_enabled and WANDB_AVAILABLE and global_step % log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                })
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
        # Log epoch metrics to wandb
        if wandb_enabled and WANDB_AVAILABLE:
            wandb.log({
                "epoch/loss": avg_loss,
                "epoch/epoch": epoch + 1,
            })
        
        # Sample and visualize
        if (epoch + 1) % sample_every == 0:
            print("Sampling...")
            diffusion.eval()
            with torch.no_grad():
                W_samples = diffusion.sample(num_samples=4, N=N, device=device)
            
            print(f"  Sampled W shape: {W_samples.shape}")
            sample_mean = W_samples.mean().item()
            sample_std = W_samples.std().item()
            print(f"  Mean: {sample_mean:.4f}, Std: {sample_std:.4f}")
            
            # Compute spectral radii for all samples
            spec_radii = []
            for i in range(W_samples.shape[0]):
                W_flat = W_samples[i, 0].cpu()  # [N, N]
                eigenvalues = torch.linalg.eigvals(W_flat)
                spec_radius = torch.abs(eigenvalues).max().item()
                spec_radii.append(spec_radius)
            
            avg_spec_radius = sum(spec_radii) / len(spec_radii)
            print(f"  Spectral radius (avg): {avg_spec_radius:.4f}")
            
            # Log sample statistics to wandb
            if wandb_enabled and WANDB_AVAILABLE:
                wandb.log({
                    "samples/mean": sample_mean,
                    "samples/std": sample_std,
                    "samples/spectral_radius_avg": avg_spec_radius,
                    "samples/spectral_radius_min": min(spec_radii),
                    "samples/spectral_radius_max": max(spec_radii),
                    "epoch/epoch": epoch + 1,
                })
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            checkpoint_file = checkpoint_path / f"prior_diffusion_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'N': N,
                'T': T,
                'schedule': schedule,
                'betas': diffusion.betas,
            }, checkpoint_file)
            print(f"Saved checkpoint: {checkpoint_file}")
    
    print("Training complete!")
    
    # Finish wandb run
    if wandb_enabled and WANDB_AVAILABLE:
        wandb.finish()


def test_sampling(checkpoint_file: str, num_samples: int = 10, device: str = "cuda"):
    """
    Test sampling from trained model.
    
    Args:
        checkpoint_file: Path to checkpoint file.
        num_samples: Number of samples to generate.
        device: Device to run on.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device)
    N = checkpoint['N']
    T = checkpoint['T']
    schedule = checkpoint['schedule']
    
    print(f"Loading checkpoint: {checkpoint_file}")
    print(f"  N = {N}, T = {T}, schedule = {schedule}")
    
    # Create model
    unet = UNetPrior(in_channels=1, base_channels=64, channel_multipliers=(1, 2, 4))
    diffusion = DiffusionPrior(unet=unet, T=T, schedule=schedule)
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    diffusion.to(device)
    
    # Sample
    print(f"\nSampling {num_samples} weight matrices...")
    W_samples = diffusion.sample(num_samples, N, device)
    
    print(f"\nSample Statistics:")
    print(f"  Shape: {W_samples.shape}")
    print(f"  Mean: {W_samples.mean().item():.6f}")
    print(f"  Std: {W_samples.std().item():.6f}")
    print(f"  Min: {W_samples.min().item():.6f}")
    print(f"  Max: {W_samples.max().item():.6f}")
    
    # Compute spectral radii
    print(f"\nSpectral Radii:")
    for i in range(min(num_samples, 5)):
        W_flat = W_samples[i, 0].cpu()  # [N, N]
        eigenvalues = torch.linalg.eigvals(W_flat)
        spec_radius = torch.abs(eigenvalues).max().item()
        print(f"  Sample {i}: {spec_radius:.4f}")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train unconditional diffusion prior")
    
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--T", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"], help="Noise schedule")
    parser.add_argument("--base_channels", type=int, default=64, help="UNet base channels")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--sample_every", type=int, default=5, help="Sample every N epochs")
    
    parser.add_argument("--test", action="store_true", help="Test sampling from checkpoint")
    parser.add_argument("--checkpoint_file", type=str, help="Checkpoint file for testing")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples for testing")
    
    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="diffusion-inference", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--log-interval", type=int, default=100, help="Log to wandb every N steps")
    
    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    if args.test:
        if not args.checkpoint_file:
            raise ValueError("--checkpoint_file required for testing")
        test_sampling(args.checkpoint_file, args.num_samples, args.device)
    else:
        train(
            dataset_dir=args.dataset_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            T=args.T,
            schedule=args.schedule,
            base_channels=args.base_channels,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            save_every=args.save_every,
            sample_every=args.sample_every,
            wandb_enabled=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            log_interval=args.log_interval,
        )

