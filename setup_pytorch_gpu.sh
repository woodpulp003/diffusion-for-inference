#!/bin/bash
# Script to install/upgrade PyTorch with CUDA support for L40S GPU
# Run this in your conda environment

echo "Installing PyTorch with CUDA 11.8+ support for L40S GPU..."

# Activate conda environment
export PATH="/home/sahil003/.pyenv/versions/miniconda3-3.10-23.10.0-1/bin:$PATH"
source /home/sahil003/.pyenv/versions/miniconda3-3.10-23.10.0-1/etc/profile.d/conda.sh
conda activate megagem

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install PyTorch 2.0+ with CUDA 11.8 or 12.1
# For CUDA 11.8:
echo "Installing PyTorch 2.0+ with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1 (if available on cluster):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    if torch.cuda.device_count() > 0:
        print(f'GPU name: {torch.cuda.get_device_name(0)}')
        # Test a simple CUDA operation
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = x @ y
        print('✓ CUDA operations working!')
"

echo ""
echo "Installation complete!"
echo "Now install other dependencies:"
echo "  pip install wandb tqdm numpy matplotlib"

