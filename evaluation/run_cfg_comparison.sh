#!/bin/bash
# Compare CFG v1 and v2 checkpoints

# Find latest checkpoints
V1_CHECKPOINT=$(ls -t checkpoints_cfg/cfg_epoch_*.pt 2>/dev/null | head -1)
V2_CHECKPOINT=$(ls -t checkpoints_cfg_v2/cfg_epoch_*.pt 2>/dev/null | head -1)

if [ -z "$V1_CHECKPOINT" ]; then
    echo "Error: No v1 checkpoint found in checkpoints_cfg/"
    exit 1
fi

if [ -z "$V2_CHECKPOINT" ]; then
    echo "Error: No v2 checkpoint found in checkpoints_cfg_v2/"
    exit 1
fi

echo "Using v1 checkpoint: $V1_CHECKPOINT"
echo "Using v2 checkpoint: $V2_CHECKPOINT"
echo ""

python evaluation/eval_cfg_comparison.py \
    --checkpoint_v1 "$V1_CHECKPOINT" \
    --checkpoint_v2 "$V2_CHECKPOINT" \
    --dataset_dir data/raw/train_dataset \
    --num_eval 256 \
    --guidance_scale 1.5 \
    --device cuda

