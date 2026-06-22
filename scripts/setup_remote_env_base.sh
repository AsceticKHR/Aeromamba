#!/bin/bash
# setup_remote_env_base.sh - Script to configure the Aeromamba training environment in the remote server's base env.

set -e

# Export CUDA paths for compilation
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "=========================================================="
echo "Starting Aeromamba environment setup on remote server (base env)..."
echo "=========================================================="

# 1. Initialize Conda
if [ -z "$CONDA_EXE" ]; then
    if [ -f "/opt/conda/bin/conda" ]; then
        CONDA_PATH="/opt/conda/bin/conda"
    else
        CONDA_PATH=$(which conda || true)
    fi
else
    CONDA_PATH="$CONDA_EXE"
fi

if [ -z "$CONDA_PATH" ]; then
    echo "Using system pip/python directly."
    PIP_CMD="pip"
    PYTHON_CMD="python"
else
    echo "Using Conda path: $CONDA_PATH"
    eval "$($CONDA_PATH shell.bash hook)"
    conda activate base
    PIP_CMD="/opt/conda/bin/pip"
    PYTHON_CMD="/opt/conda/bin/python"
fi

# 2. Configure Mirrors
echo "Configuring pip and huggingface mirrors for China network access..."
$PIP_CMD config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT="https://hf-mirror.com"
if ! grep -q "HF_ENDPOINT" ~/.bashrc; then
    echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
fi

# 3. Downgrade NumPy and Setuptools for PyTorch 2.1.2 compatibility
echo "Downgrading numpy (<2) and setuptools (<70) for PyTorch compatibility..."
$PIP_CMD install "numpy<2" "setuptools<70"

# 4. Install build tools
echo "Installing ninja and packaging..."
$PIP_CMD install ninja packaging

# 5. Install causal-conv1d and mamba-ssm (compatible with HF MambaForCausalLM)
echo "Installing causal-conv1d and mamba-ssm..."
$PIP_CMD install causal-conv1d==1.1.3.post1 --no-build-isolation || $PIP_CMD install causal-conv1d --no-build-isolation
$PIP_CMD install mamba-ssm==1.1.3.post1 --no-build-isolation || $PIP_CMD install mamba-ssm==1.2.2 --no-build-isolation || $PIP_CMD install mamba-ssm --no-build-isolation

# 6. Install requirements from requirements.txt (skipping torch/torchvision which are pre-installed)
echo "Installing other project dependencies..."
$PIP_CMD install transformers>=4.39.0 peft>=0.9.0 timm>=0.9.16 flask>=3.0.0 pillow>=10.0.0 safetensors>=0.4.0

# 7. Verification
echo "=========================================================="
echo "Verifying installation in base environment..."
$PYTHON_CMD -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))
"

$PYTHON_CMD -c "
try:
    import causal_conv1d
    import mamba_ssm
    print('causal-conv1d successfully imported!')
    print('mamba-ssm successfully imported!')
    print('Environment setup COMPLETED successfully!')
except ImportError as e:
    print('Warning: Mamba kernels failed to import:', e)
"
echo "=========================================================="
