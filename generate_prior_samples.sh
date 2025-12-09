#!/bin/bash
# Bash script to generate 50 matrices from the prior model

# Set default values
CHECKPOINT_FILE="${CHECKPOINT_FILE:-checkpoints/prior_diffusion_epoch_100.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
OUTPUT_FILE="${OUTPUT_FILE:-prior_samples.pt}"
DEVICE="${DEVICE:-cuda}"

# Print configuration
echo "=========================================="
echo "Generating Prior Model Samples"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_FILE"
echo "Number of samples: $NUM_SAMPLES"
echo "Output file: $OUTPUT_FILE"
echo "Device: $DEVICE"
echo "=========================================="
echo ""

# Run the Python script
python3 generate_prior_samples.py \
    --checkpoint_file "$CHECKPOINT_FILE" \
    --num_samples "$NUM_SAMPLES" \
    --output_file "$OUTPUT_FILE" \
    --device "$DEVICE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Successfully generated $NUM_SAMPLES matrices!"
    echo "Output saved to: $OUTPUT_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Error: Generation failed!"
    echo "=========================================="
    exit 1
fi

