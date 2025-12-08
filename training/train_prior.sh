#!/bin/bash
#SBATCH -p mit_normal_gpu   
#SBATCH --gres=gpu:l40s:1 
#SBATCH -t 240
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=10GB
#SBATCH --job-name=diffusion_prior

# Navigate to project directory
cd /home/sahil003/diffusion_inference

echo "======== Diffusion Prior Training ========"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU requested: 1 x L40s"
echo "Start time: $(date)"
echo ""

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
export PATH="/home/sahil003/.pyenv/versions/miniconda3-3.10-23.10.0-1/bin:$PATH"
source /home/sahil003/.pyenv/versions/miniconda3-3.10-23.10.0-1/etc/profile.d/conda.sh
conda activate megagem

# Use conda environment (don't activate .venv which has old Python/PyTorch)
# The conda environment should have the correct Python and PyTorch versions

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check and upgrade PyTorch if needed
echo "Checking PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    # Check if PyTorch supports sm_89 (L40S)
    if hasattr(torch.cuda, 'get_device_capability'):
        cap = torch.cuda.get_device_capability(0)
        print(f'GPU compute capability: {cap[0]}.{cap[1]}')
        if cap[0] < 8:
            print('WARNING: PyTorch may not support L40S (sm_89). Consider upgrading.')
"
echo ""

# Test GPU availability
echo "Testing GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
echo ""

echo "======== Starting Diffusion Prior Training ========"
echo "Model: Unconditional DDPM over weight matrices"
echo "Architecture: UNet with time conditioning"
echo "W&B Project: diffusion-inference"
echo ""

# Install wandb if not available
if ! python -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb --quiet
fi

# Run training with wandb
python training/train_prior.py \
    --dataset_dir data/raw/train_dataset \
    --num_epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --T 1000 \
    --schedule cosine \
    --base_channels 64 \
    --checkpoint_dir checkpoints \
    --device cuda \
    --save_every 10 \
    --sample_every 5 \
    --wandb \
    --wandb-project diffusion-inference \
    --wandb-run-name "prior_diffusion_$(date +%Y%m%d_%H%M%S)" \
    --log-interval 100

echo ""
echo "======== Training Completed ========"
echo "End time: $(date)"
echo "Checkpoints saved to: checkpoints/"

