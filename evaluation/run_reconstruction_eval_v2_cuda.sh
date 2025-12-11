#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 240           # 4 hours
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=24GB
#SBATCH --job-name=cfg_recon_eval_v2

cd /home/sahil003/diffusion_inference

echo "======== CFG Reconstruction Evaluation v2 (GPU, guidance_scale=3.0) ========"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU requested: 1 x L40s"
echo "Start time: $(date)"
echo ""

# Environment setup
export PATH="/home/sahil003/.pyenv/versions/miniconda3-3.10-23.10.0-1/bin:$PATH"
source /home/sahil003/.pyenv/versions/miniconda3-3.10-23.10.0-1/etc/profile.d/conda.sh
conda activate megagem

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Find latest checkpoints
COND_CHECKPOINT=$(ls -t checkpoints_cfg_v2/cfg_epoch_*.pt 2>/dev/null | head -1)
PRIOR_CHECKPOINT=$(ls -t checkpoints/prior_diffusion_epoch_*.pt 2>/dev/null | head -1)

if [ -z "$COND_CHECKPOINT" ]; then
    echo "Error: No CFG checkpoint found in checkpoints_cfg_v2/"
    exit 1
fi

PRIOR_ARG=""
if [ -n "$PRIOR_CHECKPOINT" ]; then
    PRIOR_ARG="--prior_checkpoint $PRIOR_CHECKPOINT"
fi

echo "Using CFG checkpoint: $COND_CHECKPOINT"
if [ -n "$PRIOR_CHECKPOINT" ]; then
    echo "Using prior checkpoint: $PRIOR_CHECKPOINT"
else
    echo "No prior checkpoint found; running without baseline."
fi
echo ""

# Run evaluation on GPU with guidance_scale=3.0 (default in v2 script)
python evaluation/eval_cfg_reconstruction_v2.py \
    --cond_checkpoint "$COND_CHECKPOINT" \
    $PRIOR_ARG \
    --dataset_dir data/raw/train_dataset \
    --num_eval 50 \
    --num_samples 4 \
    --guidance_scale 3.0 \
    --simulate_activity \
    --device cuda

echo ""
echo "======== Evaluation Completed ========"
echo "End time: $(date)"
echo "Results saved to: evaluation/reconstruction_results_v2/"


