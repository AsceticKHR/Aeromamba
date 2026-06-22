#!/bin/bash
# setup_remote_env.sh - Script to configure the Aeromamba training environment on the remote server.
# Run this script on your new remote server.

set -e  # Exit immediately if a command exits with a non-zero status

ENV_NAME="mamba2"
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.1.1"

echo "=========================================================="
echo "Starting Aeromamba environment setup on remote server..."
echo "=========================================================="

# 1. Initialize Conda
if [ -z "$CONDA_EXE" ]; then
    if [ -f "/root/miniconda3/bin/conda" ]; then
        CONDA_PATH="/root/miniconda3/bin/conda"
    elif [ -f "/root/anaconda3/bin/conda" ]; then
        CONDA_PATH="/root/anaconda3/bin/conda"
    elif [ -f "/opt/conda/bin/conda" ]; then
        CONDA_PATH="/opt/conda/bin/conda"
    else
        CONDA_PATH=$(which conda || true)
    fi
else
    CONDA_PATH="$CONDA_EXE"
fi

if [ -z "$CONDA_PATH" ]; then
    echo "Error: Conda is not installed or not found. Please install Miniconda/Anaconda first."
    exit 1
fi

echo "Using Conda path: $CONDA_PATH"
eval "$($CONDA_PATH shell.bash hook)"

# 2. Create Conda Environment
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Activating it..."
    conda activate "$ENV_NAME"
else
    echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    conda activate "$ENV_NAME"
fi

# 3. Configure Mirrors for Faster Downloading (China Mirroring)
echo "Configuring pip and huggingface mirrors for China network access..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT="https://hf-mirror.com"
if ! grep -q "HF_ENDPOINT" ~/.bashrc; then
    echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
fi

# 4. Install PyTorch with CUDA
echo "Installing PyTorch $PYTORCH_VERSION with CUDA 12.1..."
pip install torch==${PYTORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Fix NumPy and Setuptools Version Compatibility Issues
# NumPy 2.x is incompatible with PyTorch 2.1 C extensions. Setuptools >= 70 drops pkg_resources needed by PyTorch.
echo "Installing compatible version of numpy (< 2) and setuptools (< 70)..."
pip install "numpy<2" "setuptools<70"

# 6. Install Compilation Prerequisites (Essential for compiling Mamba extensions)
echo "Installing build prerequisites (ninja, packaging)..."
pip install ninja packaging

# 7. Install causal-conv1d and mamba-ssm
echo "Installing causal-conv1d and mamba-ssm..."
# We use --no-build-isolation to avoid dependency errors during pip compilation
pip install causal-conv1d==1.1.3.post1 --no-build-isolation || pip install causal-conv1d --no-build-isolation
pip install mamba-ssm==1.1.3 --no-build-isolation || pip install mamba-ssm --no-build-isolation

# 8. Install other requirements
echo "Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    pip install transformers>=4.39.0 peft>=0.9.0 timm>=0.9.16 flask>=3.0.0 pillow>=10.0.0 safetensors>=0.4.0
fi

# Extra check: ensure numpy is downgraded to < 2.0 (requirements might have pulled newer numpy)
pip install "numpy<2"

# 9. Verify installation
echo "=========================================================="
echo "Verifying installation..."
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))
"

python -c "
try:
    import causal_conv1d
    import mamba_ssm
    print('causal-conv1d successfully imported!')
    print('mamba-ssm successfully imported!')
    print('Environment setup COMPLETED successfully!')
except ImportError as e:
    print('Warning: Mamba kernels failed to import:', e)
    print('Sequential fallback mode will be active.')
"
echo "=========================================================="
