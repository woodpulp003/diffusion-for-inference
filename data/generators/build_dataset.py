# -*- coding: utf-8 -*-
"""
Dataset Builder for Neuro-Diffusion Inference.

Usage:
    python build_dataset.py --out_dir data/raw/my_dataset --num_networks 1000 --N 50 --g 1.5
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from typing import Optional
from tqdm import tqdm

from generate_weights import sample_weight_matrix
from simulate_activity import generate_activity_trials


def build_dataset(
    out_dir: str,
    num_networks: int,
    N: int,
    g: float,
    num_trials: int,
    T: int,
    dt: float,
    tau: float,
    I: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    h0_std: float = 1.0,
    ei_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Path:
    """
    Build dataset of (W, activity) pairs.
    
    Saves each network as network_XXXX.pt containing:
        {'W': [N,N], 'activities': List[[T,N]], 'network_idx': int}
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'num_networks': num_networks, 'N': N, 'g': g,
        'num_trials': num_trials, 'T': T, 'dt': dt, 'tau': tau,
        'noise_std': noise_std, 'h0_std': h0_std,
        'ei_ratio': ei_ratio, 'seed': seed,
    }
    torch.save(metadata, out_path / 'metadata.pt')
    
    if verbose:
        print(f"Building dataset: {num_networks} networks, N={N}, g={g}, {num_trials} trials/network")
    
    iterator = tqdm(range(num_networks), desc="Generating") if verbose else range(num_networks)
    
    for i in iterator:
        W = sample_weight_matrix(N=N, g=g, ei_ratio=ei_ratio)
        activities = generate_activity_trials(W, num_trials, T, dt, tau, I, noise_std, h0_std)
        
        network_data = {
            'W': W.detach().clone(),
            'activities': [a.detach().clone() for a in activities],
            'network_idx': i,
        }
        torch.save(network_data, out_path / f"network_{i:04d}.pt")
    
    if verbose:
        print(f"Dataset saved to {out_path}")
    
    return out_path


def load_network(filepath: str) -> dict:
    """Load a network from .pt file."""
    return torch.load(filepath)


def load_metadata(dataset_dir: str) -> dict:
    """Load dataset metadata."""
    return torch.load(Path(dataset_dir) / 'metadata.pt')


def list_networks(dataset_dir: str) -> list:
    """List all network files in dataset directory."""
    return sorted([str(f) for f in Path(dataset_dir).glob("network_*.pt")])


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic neural network dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_networks", type=int, required=True, help="Number of networks")
    parser.add_argument("--N", type=int, default=50, help="Neurons per network")
    parser.add_argument("--g", type=float, default=1.5, help="Gain (spectral radius)")
    parser.add_argument("--ei_ratio", type=float, default=None, help="E/I ratio for Dale's law")
    parser.add_argument("--num_trials", type=int, default=10, help="Trials per network")
    parser.add_argument("--T", type=int, default=100, help="Timesteps per trial")
    parser.add_argument("--dt", type=float, default=0.1, help="Euler step size")
    parser.add_argument("--tau", type=float, default=1.0, help="Time constant")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Process noise std")
    parser.add_argument("--h0_std", type=float, default=1.0, help="Initial condition std")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    build_dataset(
        out_dir=args.out_dir,
        num_networks=args.num_networks,
        N=args.N,
        g=args.g,
        num_trials=args.num_trials,
        T=args.T,
        dt=args.dt,
        tau=args.tau,
        noise_std=args.noise_std,
        h0_std=args.h0_std,
        ei_ratio=args.ei_ratio,
        seed=args.seed,
        verbose=not args.quiet,
    )
