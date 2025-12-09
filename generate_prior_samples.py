# -*- coding: utf-8 -*-
"""
Generate weight matrices from the trained prior diffusion model.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from models.diffusion_prior import UNetPrior, DiffusionPrior


def generate_prior_samples(
    checkpoint_file: str,
    num_samples: int = 50,
    output_file: str = "prior_samples.pt",
    device: str = "cuda",
):
    """
    Generate weight matrices from the trained prior diffusion model.
    
    Args:
        checkpoint_file: Path to trained checkpoint.
        num_samples: Number of matrices to generate.
        output_file: Path to save generated matrices.
        device: Device to use (cuda or cpu).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"\nLoading checkpoint: {checkpoint_file}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device)
    N = checkpoint.get('N', 50)  # Default to 50 if not in checkpoint
    T = checkpoint.get('T', 1000)
    schedule = checkpoint.get('schedule', 'cosine')
    
    print(f"  N = {N}, T = {T}, schedule = {schedule}")
    
    # Create model
    print("\nInitializing model...")
    unet = UNetPrior(in_channels=1, base_channels=64, channel_multipliers=(1, 2, 4))
    diffusion = DiffusionPrior(unet=unet, T=T, schedule=schedule)
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    diffusion.to(device)
    diffusion.eval()
    
    print(f"\nGenerating {num_samples} weight matrices (full quality, {T} steps each)...")
    
    # Sample in batches to avoid memory issues
    batch_size = 32
    W_list = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(0, num_samples, batch_size):
            batch_size_actual = min(batch_size, num_samples - batch_idx)
            batch_num = batch_idx // batch_size + 1
            
            print(f"  Batch {batch_num}/{num_batches} ({batch_size_actual} samples)...", end=' ', flush=True)
            
            # Sample: returns [batch_size, 1, N, N]
            W_samples = diffusion.sample(batch_size_actual, N, device, verbose=(batch_num == 1))
            
            # Squeeze channel dimension: [batch_size, N, N]
            W_samples = W_samples.squeeze(1)
            W_list.append(W_samples.cpu())
            
            print("✓")
    
    # Concatenate all samples
    W_all = torch.cat(W_list, dim=0)  # [num_samples, N, N]
    
    print(f"\nGenerated matrices shape: {W_all.shape}")
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(W_all, output_path)
    
    print(f"\nSaved {num_samples} generated matrices to: {output_path}")
    print("=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    
    return W_all


def create_arg_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate weight matrices from trained prior diffusion model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="checkpoints/prior_diffusion_epoch_100.pt",
        help="Path to trained diffusion model checkpoint.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of matrices to generate.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="prior_samples.pt",
        help="Path to save generated matrices.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for generation.",
    )
    
    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    generate_prior_samples(
        checkpoint_file=args.checkpoint_file,
        num_samples=args.num_samples,
        output_file=args.output_file,
        device=args.device,
    )

