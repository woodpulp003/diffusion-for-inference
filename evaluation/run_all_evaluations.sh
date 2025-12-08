#!/bin/bash
# Evaluate all three models: base prior, CFG v1, and CFG v2

set -e  # Exit on error

NUM_EVAL=50
DATASET_DIR="data/raw/train_dataset"
DEVICE="cuda"

# Find latest checkpoints
PRIOR_CHECKPOINT=$(ls -t checkpoints/prior_diffusion_epoch_*.pt 2>/dev/null | head -1)
V1_CHECKPOINT=$(ls -t checkpoints_cfg/cfg_epoch_*.pt 2>/dev/null | head -1)
V2_CHECKPOINT=$(ls -t checkpoints_cfg_v2/cfg_epoch_*.pt 2>/dev/null | head -1)

if [ -z "$PRIOR_CHECKPOINT" ]; then
    echo "Error: No prior checkpoint found in checkpoints/"
    exit 1
fi

if [ -z "$V1_CHECKPOINT" ]; then
    echo "Error: No v1 checkpoint found in checkpoints_cfg/"
    exit 1
fi

if [ -z "$V2_CHECKPOINT" ]; then
    echo "Error: No v2 checkpoint found in checkpoints_cfg_v2/"
    exit 1
fi

echo "================================================================================"
echo "Evaluating All Three Models with $NUM_EVAL samples each"
echo "================================================================================"
echo ""
echo "Prior checkpoint: $PRIOR_CHECKPOINT"
echo "CFG v1 checkpoint: $V1_CHECKPOINT"
echo "CFG v2 checkpoint: $V2_CHECKPOINT"
echo ""

# 1. Evaluate Base Diffusion Prior
echo "================================================================================"
echo "1. Evaluating Base Diffusion Prior (Unconditional)"
echo "================================================================================"
python evaluation/eval_prior.py \
    --checkpoint_file "$PRIOR_CHECKPOINT" \
    --dataset_dir "$DATASET_DIR" \
    --num_eval $NUM_EVAL \
    --device $DEVICE

echo ""
echo ""

# 2. Evaluate CFG v1
echo "================================================================================"
echo "2. Evaluating CFG v1 (Conditional)"
echo "================================================================================"
python evaluation/eval_cfg.py \
    --checkpoint_file "$V1_CHECKPOINT" \
    --dataset_dir "$DATASET_DIR" \
    --num_eval $NUM_EVAL \
    --guidance_scale 1.5 \
    --device $DEVICE

echo ""
echo ""

# 3. Evaluate CFG v2
echo "================================================================================"
echo "3. Evaluating CFG v2 (Conditional)"
echo "================================================================================"
python evaluation/eval_cfg.py \
    --checkpoint_file "$V2_CHECKPOINT" \
    --dataset_dir "$DATASET_DIR" \
    --num_eval $NUM_EVAL \
    --guidance_scale 1.5 \
    --device $DEVICE

echo ""
echo ""
echo "================================================================================"
echo "All Evaluations Complete!"
echo "================================================================================"

