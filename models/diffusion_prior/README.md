# Phase 3: Unconditional Diffusion Prior

This module implements an unconditional DDPM (Denoising Diffusion Probabilistic Model) over synaptic weight matrices W [N, N].

## Components

### 1. `unet_prior.py` - UNet Architecture
- **UNetPrior**: UNet model that treats weight matrices as single-channel images [1, N, N]
- Time-step conditioning via sinusoidal embeddings and FiLM (Feature-wise Linear Modulation)
- Downsampling → mid block → upsampling with skip connections

### 2. `noise_schedule.py` - Noise Schedule Utilities
- **make_beta_schedule**: Creates beta schedules (linear or cosine)
- **compute_alpha_cumprod**: Computes alpha and cumulative alpha products
- **q_sample**: Forward diffusion process (adds noise to clean W)

### 3. `prior_diffusion.py` - DDPM Wrapper
- **DiffusionPrior**: Main diffusion model class
  - `p_losses()`: Training loss (MSE between predicted and actual noise)
  - `p_sample()`: Single reverse diffusion step
  - `p_sample_loop()`: Full sampling loop
  - `sample()`: High-level sampling interface

## Usage

### Training

```bash
python training/train_prior.py \
    --dataset_dir data/raw/train_dataset \
    --num_epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --T 1000 \
    --schedule cosine \
    --checkpoint_dir checkpoints
```

### Testing/Sampling

```bash
python training/train_prior.py \
    --test \
    --checkpoint_file checkpoints/prior_diffusion_epoch_100.pt \
    --num_samples 10
```

## Model Architecture

- **Input**: Weight matrix W [N, N] → reshaped to [1, N, N]
- **Output**: Predicted noise [1, N, N]
- **Timesteps**: T = 1000 (default)
- **Noise Schedule**: Cosine (default) or linear
- **UNet**: Base channels = 64, multipliers = (1, 2, 4)

## Checkpoint Format

Saved checkpoints contain:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `N`: Network size
- `T`: Number of timesteps
- `schedule`: Noise schedule type
- `betas`: Beta schedule values
- `epoch`: Training epoch
- `loss`: Average loss

