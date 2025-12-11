#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 240           # 4 hours
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=24GB
#SBATCH --job-name=dps_trainer

cd /home/sahil003/diffusion_inference

echo "======== DPS Trainer (CFG warm-start + refinement) ========"
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

# User parameters (edit as needed)
COND_CKPT=${COND_CKPT:-checkpoints_cfg_v2/cfg_epoch_1000.pt}
DATASET_DIR=${DATASET_DIR:-data/raw/train_dataset}
OUT_DIR=${OUT_DIR:-evaluation/dps_results}
NUM_EVAL=${NUM_EVAL:-10}
NUM_STEPS=${NUM_STEPS:-300}
LR=${LR:-1e-3}
LAMBDA_PRIOR=${LAMBDA_PRIOR:-0.01}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.5}
WANDB=${WANDB:-"--wandb"}
WANDB_PROJECT=${WANDB_PROJECT:-"diffusion-inference"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"dps_$(date +%Y%m%d_%H%M%S)"}

echo "Using CFG checkpoint: $COND_CKPT"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUT_DIR"
echo "num_eval=$NUM_EVAL, num_steps=$NUM_STEPS, lr=$LR, lambda_prior=$LAMBDA_PRIOR"
echo ""

python inference/dps_trainer.py \
    --cond_checkpoint "$COND_CKPT" \
    --dataset_dir "$DATASET_DIR" \
    --out_dir "$OUT_DIR" \
    --num_eval "$NUM_EVAL" \
    --num_steps "$NUM_STEPS" \
    --lr "$LR" \
    --lambda_prior "$LAMBDA_PRIOR" \
    $WANDB \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --device cuda

echo ""
echo "======== DPS Trainer Completed ========"
echo "End time: $(date)"

