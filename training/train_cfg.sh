#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 360           # 6 hours
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=40GB
#SBATCH --job-name=diffusion_cfg

cd /home/sahil003/diffusion_inference

echo "======== Conditional Diffusion (CFG) Training ========"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU requested: 1 x H100"
echo "Start time: $(date)"
echo ""

# Environment setup (matches prior training script)
export PATH="/home/sahil003/.pyenv/versions/miniconda3-3.10-23.10.0-1/bin:$PATH"
source /home/sahil003/.pyenv/versions/miniconda3-3.10-23.10.0-1/etc/profile.d/conda.sh
conda activate megagem

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

echo "Testing GPU availability..."
python - <<'PY'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory / 1e9:.2f} GB")
PY
echo ""

echo "======== Launching CFG Training ========"
python training/train_cfg.py \
    --dataset_dir data/raw/train_dataset \
    --num_epochs 1000 \
    --batch_size 8 \
    --lr 1e-5 \
    --T 1000 \
    --schedule cosine \
    --base_channels 64 \
    --channel_multipliers 1 2 4 \
    --d_model 128 \
    --num_heads 4 \
    --num_layers 2 \
    --cond_drop_prob 0.1 \
    --checkpoint_dir checkpoints_cfg_v2 \
    --save_every 10 \
    --log_interval 100 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --device cuda \
    --wandb \
    --wandb-project diffusion-inference \
    --wandb-run-name "cfg_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "======== Training Completed ========"
echo "End time: $(date)"
echo "Checkpoints saved to: checkpoints_cfg/"

